"""Run with an existing Python that has Torch and safetensors; no ComfyUI needed."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
folders = types.ModuleType("folder_paths")
folders.models_dir = str(ROOT / "models")
folders.folder_names_and_paths = {}
folders.add_model_folder_path = lambda name, path: folders.folder_names_and_paths[name][0].append(path)
folders.get_filename_list = lambda name: ["pilot.safetensors"]
folders.get_full_path_or_raise = lambda name, file: file
spec = importlib.util.spec_from_file_location("moss_nodes_under_test", ROOT / "nodes.py")
node = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"folder_paths": folders}):
    spec.loader.exec_module(node)


def model():
    result = {"weight": torch.zeros(2560, 4096, dtype=torch.float16),
              "mean_in": torch.ones(2560), "std_in": torch.full((2560,), 2.),
              "mean_out": torch.full((4096,), 5.), "std_out": torch.full((4096,), 4.)}
    result["weight"][0, 0] = 2
    return result


META = {"format": node.ADAPTER_FORMAT, "source_layer": "last_normalized", "target_layer": "last_normalized"}
TOKENS = {"qwen35_4b": [[(11, 1.), (12, 1.)]]}


class NodeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="moss-node-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "adapter.safetensors"
        self.adapter = model()
        save_file(self.adapter, self.path, metadata=META)

    def apply(self, clip):
        tokenizer = types.ModuleType("comfy.text_encoders.moss_image")
        tokenizer.MOSSImageTokenizer = FakeTokenizer
        with patch.dict(sys.modules, {"comfy.text_encoders.moss_image": tokenizer}):
            return node.MOSSSmallerEncoderAdapter().apply(clip, str(self.path))[0]

    def assert_restored(self, clip):
        self.assertEqual(clip.layer_idx, -2)
        self.assertEqual(clip.cond_stage_model.encoder.layer, "hidden")
        self.assertEqual(clip.cond_stage_model.encoder.options_default, ("hidden", -2, False))

    def test_node_returns_clip_and_normal_encode_projects(self):
        clip = FakeClip()
        adapted = self.apply(clip)
        self.assertEqual(node.MOSSSmallerEncoderAdapter.RETURN_TYPES, ("CLIP",))
        self.assertNotIn("text", node.MOSSSmallerEncoderAdapter.INPUT_TYPES()["required"])
        self.assertIsNot(adapted._clip, clip)
        tokens = adapted.tokenize("hello")
        self.assertEqual(tokens, TOKENS)
        cond, extra = adapted.encode_from_tokens_scheduled(tokens)[0]
        self.assertEqual(tuple(cond.shape), (1, 2, 4096))
        self.assertEqual(cond.dtype, torch.float32)
        self.assertEqual(cond[0, 0, 0].item(), 13.)
        self.assertEqual(cond[0, 0, 1].item(), 5.)
        self.assertTrue(torch.equal(extra["attention_mask"], torch.ones(1, 2)))
        self.assert_restored(clip)
        self.assert_restored(adapted)

    def test_direct_encoding_return_modes(self):
        adapted = self.apply(FakeClip())
        cond = adapted.encode_from_tokens(TOKENS)
        pooled_cond, pooled = adapted.encode_from_tokens(TOKENS, return_pooled="unprojected")
        as_dict = adapted.encode_from_tokens(TOKENS, return_dict=True)
        self.assertEqual(tuple(cond.shape), (1, 2, 4096))
        self.assertTrue(torch.equal(cond, pooled_cond))
        self.assertTrue(torch.equal(cond, as_dict["cond"]))
        self.assertIsNone(pooled)
        self.assertEqual(adapted._clip.last_return_pooled, False)
        self.assert_restored(adapted)

    def test_clone_preserves_adapter_and_upstream_lora_patches(self):
        clip = FakeClip()
        clip.patcher.patches["upstream_lora"] = .8
        adapted = self.apply(clip)
        cloned = adapted.clone(disable_dynamic=True)
        self.assertIsInstance(cloned, node.MOSSProjectedCLIP)
        self.assertIsNot(cloned._clip, adapted._clip)
        self.assertEqual(cloned.patcher.patches, {"upstream_lora": .8})
        cloned.patcher.patches["later_patch"] = .5
        self.assertNotIn("later_patch", clip.patcher.patches)
        self.assertNotIn("later_patch", adapted.patcher.patches)
        self.assertEqual(cloned.tokenize("hello", return_word_ids=True, tokenizer_options={"extra": 2}), TOKENS)
        self.assertEqual(cloned._tokenizer.last_options, (True, {"base": 1, "extra": 2}))
        self.assertEqual(cloned.encode_from_tokens(TOKENS).shape[-1], 4096)

    def test_scheduled_entries_keep_metadata(self):
        adapted = self.apply(FakeClip())
        adapted._clip.schedule = True
        result = adapted.encode_from_tokens_scheduled(TOKENS, unprojected=True,
                                                      add_dict={"custom": "kept"}, show_pbar=False)
        self.assertEqual(len(result), 2)
        for index, (cond, metadata) in enumerate(result):
            self.assertEqual(cond.shape[-1], 4096)
            self.assertEqual(metadata["clip_start_percent"], index * .5)
            self.assertEqual(metadata["clip_end_percent"], (index + 1) * .5)
            self.assertEqual(metadata["custom"], "kept")
            self.assertEqual(metadata["hooks"], "existing_hooks")
        self.assert_restored(adapted)

    def test_exact_pilot_arithmetic(self):
        gen = torch.Generator().manual_seed(23)
        x = torch.randn(1, 11, 2560, generator=gen)
        self.adapter["weight"] = (torch.randn(2560, 4096, generator=gen) * .01).half()
        expected = ((x.float() - self.adapter["mean_in"]) / self.adapter["std_in"]) @ self.adapter["weight"].float()
        expected = (expected * self.adapter["std_out"] + self.adapter["mean_out"]).half().float()
        self.assertTrue(torch.equal(node.project(x, self.adapter), expected))

    def test_rejects_incompatible_files(self):
        for metadata in ({}, {**META, "format": "lora"}, {**META, "source_layer": "penultimate"}):
            with self.subTest(metadata=metadata):
                save_file(self.adapter, self.path, metadata=metadata)
                with self.assertRaises(ValueError):
                    node.load_adapter(self.path)
        for key, value in (("weight", torch.zeros(4, 9)), ("std_in", torch.zeros(2560)),
                           ("mean_out", torch.full((4096,), float("nan")))):
            with self.subTest(key=key):
                save_file({**self.adapter, key: value}, self.path, metadata=META)
                with self.assertRaises(ValueError):
                    node.load_adapter(self.path)
        save_file({**self.adapter, "extra": torch.zeros(1)}, self.path, metadata=META)
        with self.assertRaises(ValueError):
            node.load_adapter(self.path)

    def test_rejects_wrong_features_encoder_and_double_adapter(self):
        for x in (torch.zeros(1, 2, 4096), torch.zeros(2, 2560), torch.full((1, 2, 2560), float("nan"))):
            with self.assertRaises(ValueError):
                node.project(x, self.adapter)
        for name in ("qwen35_9b", "qwen3_4b", "qwen35_2b"):
            clip = FakeClip()
            clip.cond_stage_model.clip_name = name
            with self.assertRaises(ValueError):
                self.apply(clip)
        with self.assertRaisesRegex(ValueError, "already has"):
            self.apply(self.apply(FakeClip()))

    def test_encoder_options_reset_after_failure(self):
        clip = FakeClip(fail=True)
        adapted = self.apply(clip)
        for method in (adapted.encode_from_tokens, adapted.encode_from_tokens_scheduled):
            with self.assertRaisesRegex(RuntimeError, "encoder failure"):
                method(TOKENS)
            self.assert_restored(clip)
            self.assert_restored(adapted)


class FakeTokenizer:
    def tokenize_with_weights(self, text, return_word_ids=False, tokenizer_options=None):
        assert text == "hello"
        self.last_options = (return_word_ids, tokenizer_options)
        return {"qwen35_9b": TOKENS["qwen35_4b"]}


class FakeClip:
    def __init__(self, fail=False):
        self.layer_idx = -2
        self.fail = fail
        self.schedule = False
        self.tokenizer_options = {"base": 1}
        self.patcher = types.SimpleNamespace(patches={})
        inner = types.SimpleNamespace(layer="hidden", options_default=("hidden", -2, False))
        self.cond_stage_model = types.SimpleNamespace(clip_name="qwen35_4b", clip="encoder", encoder=inner)
        self.cond_stage_model.reset_clip_options = lambda: setattr(inner, "layer", "hidden")

    def clone(self, *args, **kwargs):
        cloned = copy.copy(self)
        cloned.patcher = copy.deepcopy(self.patcher)
        return cloned

    def clip_layer(self, index):
        self.layer_idx = index

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        assert tokens == TOKENS and return_dict and self.layer_idx is None
        assert self.cond_stage_model.encoder.options_default == ("last", None, False)
        self.cond_stage_model.encoder.layer = "last"
        self.last_return_pooled = return_pooled
        if self.fail:
            raise RuntimeError("encoder failure")
        return {"cond": torch.full((1, 2, 2560), 3.), "attention_mask": torch.ones(1, 2), "pooled_output": None}

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        encoded = self.encode_from_tokens(tokens, return_dict=True)
        features = encoded.pop("cond")
        return [[features.clone(), {**encoded, **add_dict, "clip_start_percent": i * .5,
                                   "clip_end_percent": (i + 1) * .5, "hooks": "existing_hooks"}]
                for i in range(2 if self.schedule else 1)]


if __name__ == "__main__":
    unittest.main()
