"""Explicit MOSS text encoding with a fitted Qwen3.5 feature projection."""
from pathlib import Path

import torch
from safetensors import safe_open

import folder_paths

ADAPTER_FOLDER = "moss_encoder_adapters"
ADAPTER_FORMAT = "potatoforge_moss_encoder_projection_v1"

# Keep configured model paths and allow a local adapter without copying into ComfyUI.
if ADAPTER_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[ADAPTER_FOLDER] = ([], {".safetensors"})
folder_paths.add_model_folder_path(ADAPTER_FOLDER, str(Path(folder_paths.models_dir) / ADAPTER_FOLDER))
folder_paths.add_model_folder_path(ADAPTER_FOLDER, str(Path(__file__).resolve().parent / "adapters"))


def load_adapter(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != ADAPTER_FORMAT:
            raise ValueError("Select a MOSS encoder projection adapter, not a LoRA or text encoder.")
        if any(metadata.get(key) != "last_normalized" for key in ("source_layer", "target_layer")):
            raise ValueError("This node requires final normalized source and target features.")
        shapes = {"weight": (2560, 4096), "mean_in": (2560,), "std_in": (2560,),
                  "mean_out": (4096,), "std_out": (4096,)}
        if set(handle.keys()) != set(shapes):
            raise ValueError("The adapter has missing or unsupported tensors.")
        for key, shape in shapes.items():
            if tuple(handle.get_slice(key).get_shape()) != shape:
                raise ValueError(f"Adapter tensor {key} must have shape {shape}.")
        adapter = {key: handle.get_tensor(key) for key in shapes}
    for key, value in adapter.items():
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Adapter tensor {key} must contain finite floating-point values.")
    if any(not bool((adapter[key] > 0).all()) for key in ("std_in", "std_out")):
        raise ValueError("Adapter standard deviations must be positive.")
    return adapter


def project(features, adapter):
    if features.ndim != 3 or features.shape[-1] != 2560:
        raise ValueError("Expected Qwen3.5-4B conditioning with 2560 features per token.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("The text encoder returned non-finite features.")
    # Match the pilot's CPU FP32 calculation and FP16 output rounding.
    features = features.to(device="cpu", dtype=torch.float32)
    result = ((features - adapter["mean_in"]) / adapter["std_in"]) @ adapter["weight"].float()
    result = (result * adapter["std_out"] + adapter["mean_out"]).half().float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("The projected conditioning contains non-finite values.")
    return result


class MOSSProjectedCLIP:
    """Keep the adapter on this CLIP branch, including across downstream clones."""

    def __init__(self, clip, adapter, tokenizer):
        self._clip = clip
        self._adapter = adapter
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._clip, name)

    def clone(self, *args, **kwargs):
        return type(self)(self._clip.clone(*args, **kwargs), self._adapter, self._tokenizer)

    def tokenize(self, text, return_word_ids=False, **kwargs):
        options = {**self._clip.tokenizer_options, **kwargs.get("tokenizer_options", {})}
        if options:
            kwargs["tokenizer_options"] = options
        tokens = self._tokenizer.tokenize_with_weights(text, return_word_ids=return_word_ids, **kwargs)
        return {self._clip.cond_stage_model.clip_name: tokens["qwen35_9b"]}

    def _encode(self, method, *args, **kwargs):
        stage = self._clip.cond_stage_model
        inner = getattr(stage, stage.clip)
        defaults, layer_idx = inner.options_default, self._clip.layer_idx
        # Comfy resets options and reapplies them when setting the execution device.
        try:
            inner.options_default = ("last", None, defaults[2])
            self._clip.clip_layer(None)
            return method(*args, **kwargs)
        finally:
            inner.options_default = defaults
            self._clip.clip_layer(layer_idx)
            stage.reset_clip_options()

    @torch.inference_mode()
    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        encoded = self._encode(self._clip.encode_from_tokens, tokens,
                               return_pooled=return_pooled, return_dict=True)
        encoded["cond"] = project(encoded["cond"], self._adapter)
        if return_dict:
            return encoded
        if return_pooled:
            return encoded["cond"], encoded["pooled_output"]
        return encoded["cond"]

    @torch.inference_mode()
    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        conditions = self._encode(self._clip.encode_from_tokens_scheduled, tokens,
                                  unprojected=unprojected, add_dict={} if add_dict is None else add_dict,
                                  show_pbar=show_pbar)
        return [[project(features, self._adapter), metadata] for features, metadata in conditions]


class MOSSSmallerEncoderAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "adapter_name": (folder_paths.get_filename_list(ADAPTER_FOLDER),),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/MOSS"
    DESCRIPTION = "Prepare Qwen3.5-4B with a selectable MOSS adapter before normal CLIP Text Encode."

    def apply(self, clip, adapter_name):
        if isinstance(clip, MOSSProjectedCLIP):
            raise ValueError("This CLIP already has a MOSS encoder adapter. Use one adapter per CLIP branch.")
        if getattr(clip.cond_stage_model, "clip_name", None) != "qwen35_4b":
            raise ValueError("Load Qwen3.5-4B as the CLIP input. Qwen3-VL-4B and the original 9B are unsupported.")
        adapter = load_adapter(folder_paths.get_full_path_or_raise(ADAPTER_FOLDER, adapter_name))
        from comfy.text_encoders.moss_image import MOSSImageTokenizer

        return (MOSSProjectedCLIP(clip.clone(), adapter, MOSSImageTokenizer()),)


NODE_CLASS_MAPPINGS = {"MOSSSmallerEncoderAdapter": MOSSSmallerEncoderAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSSmallerEncoderAdapter": "MOSS Smaller Encoder Adapter"}
