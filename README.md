# MOSS Image Smaller Encoder

Use **Qwen3.5-4B** instead of MOSS Image's original Qwen3.5-9B text encoder in ComfyUI. Select a compatible projection adapter in a visible node, then write your prompts in the normal **CLIP Text Encode** node.

## Credits

This project was inspired by [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) by [nicolab28](https://github.com/nicolab28), which demonstrated replacing a larger text encoder with a smaller encoder and a learned feature projection. Thanks to nicolab28 for sharing the approach, calibration methods and experiments that motivated this MOSS Image adaptation.

## How this project is built

This project is **vibe coded-ish**: I use AI to help research, write code and run experiments, while I review the proposed steps, verify the work and inspect the results as development progresses. It's an AI-assisted project with hands-on human oversight, and the testing limits are documented below.

## Image comparison

Compare the original 9B encoder on the left with the projected 4B encoder on the right. Each pair uses the same prompt, seed and generation settings. These previews use the original 640 x 960 PNGs; click an image to open it at full resolution.

### Portrait

Seed **2026101801**

| Original Qwen3.5-9B                                                                                                                 | Projected Qwen3.5-4B                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [![portrait, original 9B, seed 2026101801](images/portrait_2026101801_original_9b.png)](images/portrait_2026101801_original_9b.png) | [![portrait, projected 4B, seed 2026101801](images/portrait_2026101801_projected_4b.png)](images/portrait_2026101801_projected_4b.png) |

<details>
<summary>Compare the second seed (2026101802)</summary>

| Original Qwen3.5-9B                                                                                                                 | Projected Qwen3.5-4B                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [![portrait, original 9B, seed 2026101802](images/portrait_2026101802_original_9b.png)](images/portrait_2026101802_original_9b.png) | [![portrait, projected 4B, seed 2026101802](images/portrait_2026101802_projected_4b.png)](images/portrait_2026101802_projected_4b.png) |

</details>

### Watercolor

Seed **2026101801**

| Original Qwen3.5-9B                                                                                                                       | Projected Qwen3.5-4B                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [![watercolor, original 9B, seed 2026101801](images/watercolor_2026101801_original_9b.png)](images/watercolor_2026101801_original_9b.png) | [![watercolor, projected 4B, seed 2026101801](images/watercolor_2026101801_projected_4b.png)](images/watercolor_2026101801_projected_4b.png) |

<details>
<summary>Compare the second seed (2026101802)</summary>

| Original Qwen3.5-9B                                                                                                                       | Projected Qwen3.5-4B                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [![watercolor, original 9B, seed 2026101802](images/watercolor_2026101802_original_9b.png)](images/watercolor_2026101802_original_9b.png) | [![watercolor, projected 4B, seed 2026101802](images/watercolor_2026101802_projected_4b.png)](images/watercolor_2026101802_projected_4b.png) |

</details>

<details>
<summary>Placement comparisons</summary>

Seed **2026101801**

| Original Qwen3.5-9B                                                                                                                    | Projected Qwen3.5-4B                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [![placement, original 9B, seed 2026101801](images/placement_2026101801_original_9b.png)](images/placement_2026101801_original_9b.png) | [![placement, projected 4B, seed 2026101801](images/placement_2026101801_projected_4b.png)](images/placement_2026101801_projected_4b.png) |

<details>
<summary>Compare the second seed (2026101802)</summary>

| Original Qwen3.5-9B                                                                                                                    | Projected Qwen3.5-4B                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [![placement, original 9B, seed 2026101802](images/placement_2026101802_original_9b.png)](images/placement_2026101802_original_9b.png) | [![placement, projected 4B, seed 2026101802](images/placement_2026101802_projected_4b.png)](images/placement_2026101802_projected_4b.png) |

</details>

</details>

<details>
<summary>Lettering comparisons</summary>

Seed **2026101801**

| Original Qwen3.5-9B                                                                                                                    | Projected Qwen3.5-4B                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [![lettering, original 9B, seed 2026101801](images/lettering_2026101801_original_9b.png)](images/lettering_2026101801_original_9b.png) | [![lettering, projected 4B, seed 2026101801](images/lettering_2026101801_projected_4b.png)](images/lettering_2026101801_projected_4b.png) |

<details>
<summary>Compare the second seed (2026101802)</summary>

| Original Qwen3.5-9B                                                                                                                    | Projected Qwen3.5-4B                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [![lettering, original 9B, seed 2026101802](images/lettering_2026101802_original_9b.png)](images/lettering_2026101802_original_9b.png) | [![lettering, projected 4B, seed 2026101802](images/lettering_2026101802_projected_4b.png)](images/lettering_2026101802_projected_4b.png) |

</details>

</details>

<details>
<summary>View the complete overview grid</summary>

[![All four prompts and both seeds, original 9B versus projected 4B](images/encoder-comparison.png)](images/encoder-comparison.png)

</details>

## What you need

- A working MOSS Image workflow and a ComfyUI installation that supports MOSS Image and Qwen3.5.
- A Qwen3.5-4B text encoder. The tested checkpoint is [Qwen3.5-4B INT8 ConvRot](https://huggingface.co/beycanai/Qwen-LM/resolve/main/Qwen3.5-4B_int8_convrot.safetensors?download=true). Your ComfyUI installation must support loading its quantization format.
- A compatible MOSS projection adapter, such as `moss_qwen35_4b_to_9b.safetensors` (approximately 21 MB). You need a separate copy of this file; adapter weights are not included in the Git repository.

Choose **Qwen3.5-4B**, not Qwen3-VL-4B. They are different model families.

## Install

1. Place this repository in `ComfyUI/custom_nodes/moss-image-smaller-encoder`.
2. Place your Qwen3.5-4B checkpoint in `ComfyUI/models/text_encoders/`.
3. Place your projection adapter in `ComfyUI/models/moss_encoder_adapters/`. You can also use the `adapters/` folder inside this repository.
4. Restart ComfyUI.

The node uses ComfyUI's existing Torch, safetensors and tokenizer dependencies. No additional packages are required.

## Connect your workflow

The included [workflow.json](workflow.json) uses the [MOSS Image Base 8-step DMD LoRA by Lodestones](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd). Download the LoRA into `ComfyUI/models/loras/` and select it in **Load LoRA**; the workflow references it as `moss_image_8_step.safetensors`.

**Load CLIP -> Load LoRA (optional) -> MOSS Smaller Encoder Adapter -> CLIP Text Encode -> KSampler**

1. In **Load CLIP**, select your Qwen3.5-4B checkpoint and set **type** to **moss_image**.
2. Add **MOSS Smaller Encoder Adapter** from `conditioning/MOSS`.
3. Connect the CLIP output of **Load CLIP** to the adapter node. If you use **Load LoRA**, connect its CLIP output instead.
4. Select your projection file in **adapter_name**.
5. Connect the adapter's CLIP output to the normal **CLIP Text Encode** node. Enter your prompt there.
6. Connect the text encoder's CONDITIONING output to KSampler's positive input.

Keep your MOSS diffusion model, VAE and sampling setup. When using Load LoRA, connect its MODEL output along your usual model path to KSampler.

For a CFG 1 workflow, connect the positive conditioning through **ConditioningZeroOut** to KSampler's negative input. To supply a separate negative prompt, connect another CLIP Text Encode node to the same adapted CLIP; higher CFG and explicit negative prompting have not been evaluated.

## Compatibility

Use a projection adapter made for **Qwen3.5-4B -> MOSS's Qwen3.5-9B feature space**. The node accepts `.safetensors` files with the `potatoforge_moss_encoder_projection_v1` format and final normalized 2560-to-4096 features. LoRA files and ClipProj adapters cannot be selected as substitutes.

You can keep compatible LoRAs before the adapter node. The adapter preserves their patches when cloning CLIP, but it does not convert a 9B text-encoder LoRA into a 4B LoRA. Real text-encoder LoRAs and scheduled encoder hooks have not been evaluated.

Use text-only prompts with the standard CLIP Text Encode node. MOSS treats prompt-weight syntax such as `(word:1.2)` as literal text. Image inputs and specialized nodes that bypass CLIP's encoding methods are unsupported.

## What to expect

The adapter is experimental. The initial fit used 200 training prompts, 50 validation prompts and 50 held-out test prompts. Eight image comparisons covered four English prompts and two seeds at 640 x 960, 8 steps, Euler/simple, CFG 1, with the MOSS 8-step LoRA at strength 0.8. The projected encoder preserved the main subjects and styles in these examples, but compositions and incidental details changed.

Expect differences from the original 9B encoder. Quality with other quantizations, longer prompts, adult content, editing and other generation settings has not been established.

The CLIP-to-CLIP node has been checked in the real ComfyUI runtime: standard CLIP Text Encode reproduced the four comparison prompts' saved conditioning exactly.

## How it works

The adapter node prepares a cloned CLIP branch with MOSS's prompt template and final normalized encoder output. When CLIP Text Encode runs, the adapter converts each token's 2560 features into the 4096 features MOSS expects. Projection runs on CPU in FP32, with FP16 output rounding. The encoder and diffusion-model weights stay unchanged, and you can still use the original CLIP branch separately.

## Troubleshooting

- **The adapter list is empty:** check the adapter's folder and `.safetensors` extension, then restart ComfyUI.
- **The node asks for Qwen3.5-4B:** check the selected text encoder. Qwen3-VL-4B and the original 9B checkpoint are unsupported inputs.
- **The adapter format or shape is rejected:** use a compatible MOSS projection file. Renaming another adapter or LoRA will not convert it.
- **ComfyUI cannot load the encoder checkpoint:** check that your installation supports the checkpoint's quantization format. This node applies the projection after encoder loading.
