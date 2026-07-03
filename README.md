# GAP SCAIL-2 Character Replacement for ComfyUI

**by [Geekatplay Studio](https://www.youtube.com/@geekatplay)** — tutorials on this and other
AI video workflows on the YouTube channel: **[@geekatplay](https://www.youtube.com/@geekatplay)**

Replace **1–6 people** in a driving video of **any length** with your own characters
(humans, toys, robots, creatures) using the SCAIL-2 (Wan 2.1) model — with one queue press.

The stock SCAIL-2 template handles a single 81-frame segment with one character and requires
manually duplicating subgraphs for every extra segment. This node pack automates all of it:

- **Any-length videos** — automatic chunking (81 frames / 5-frame overlap by default), all
  chunks generated and stitched in one run, with the original audio.
- **Multiple characters** — up to 6 reference images, each bound to a tracked person via
  SCAIL-2's identity color palette (1=blue, 2=red, 3=green, 4=magenta, 5=cyan, 6=yellow).
- **Dynamic prompts** — per-chunk prompt resolution from a frame-range schedule, plus
  automatic injection of character descriptions based on who is actually visible in each
  chunk (detected from the SAM3 mask colors).
- **Footage analysis** — pre-flight report: how many characters the video needs, on which
  frames each enters/leaves, and an auto-generated schedule template you just fill in.
- **Two-phase execution** — phase 1 runs only analysis (generation skipped automatically),
  phase 2 renders. No node muting needed.
- **Scene-cut handling** — hard cuts are detected; chunk boundaries snap to them and the
  frame anchor resets at each cut (no ghosting across cuts).
- **Crash-safe checkpoints** — finished chunks are saved to disk; resume a crashed render,
  or re-render a single bad chunk with a new seed and splice it back in.
- **Memory controls** — chunk length, tiled VAE decode, VRAM cache flushing between chunks.
- **Color drift correction** — each chunk is Reinhard color-matched to the previous one via
  the shared overlap frames.

## Requirements

- ComfyUI with SCAIL-2 support (`comfy_extras/nodes_scail.py`, PR#14373 or newer).
- Models (same as the stock SCAIL-2 template):

| Type | File | Folder |
|---|---|---|
| Diffusion | [wan2.1_14B_SCAIL_2_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors) (16.5 GB, recommended) | `models/diffusion_models` |
| LoRA | [wan2.1_SCAIL_2_DPO_lora_bf16.safetensors](https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors) | `models/loras` |
| LoRA | [lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors) | `models/loras` |
| VAE | [Wan2_1_VAE_bf16.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors) | `models/vae` |
| Text encoder | [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | `models/text_encoders` |
| CLIP vision | [clip_vision_h.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) | `models/clip_vision` |
| SAM3 | [sam3.1_multiplex_fp16.safetensors](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | `models/checkpoints` |

> The fp16 diffusion model (30.5 GB) stages ~31 GB into system RAM and can crash machines
> with long videos loaded — use the fp8 version unless you have plenty of free RAM.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/GeekatplayStudio/wan-scail2-gap.git ComfyUI_gap_character_replacement
```

Restart ComfyUI, then open `example_workflows/GAP_SCAIL2_long_multi_character.json`.
The workflow is fully annotated — numbered groups ①–⑧ with a note in each explaining
exactly what to set.

## Quick start (two phases)

1. Load the driving video (②) and your character images (③); set resolution once (①).
2. Give each character image its **SUBJECT text** — what SAM3 should segment: `person`,
   `pink plush teddy bear`, `robot`…
3. Queue with the **PHASE node** on `1 - analyze only`. Generation is skipped; you get:
   - **MASK CHECK video** (④) — who got which identity color,
   - **REF MASK preview** (③) — each character's silhouette in its color,
   - **FOOTAGE ANALYSIS** (⑧) — character count, entry/exit frames, and a ready-made
     `prompt_schedule` template.
4. Fill the prompts on the generator (⑥ — cheat-sheet note right beside it), switch PHASE to
   `2 - generate video`, queue. Analysis is cached; generation starts immediately.
5. The final video (with audio) and a per-chunk report land in ⑦.

## Nodes

| Node | Purpose |
|---|---|
| **GAP SCAIL-2 Long Video** | The orchestrator: chunking, anchoring, per-chunk dynamic prompts, scene cuts, color match, checkpoints, stitching. Live progress on the node. |
| **GAP Multi-Character Reference** | Builds the multi-identity reference batch from up to 6 character images + masks. |
| **GAP Character Extra View** | Appends an extra reference view (back view / close-up) for one character — chain freely. |
| **GAP Character Timeline** | Footage analysis: who appears when + auto-generated schedule template. |
| **GAP SCAIL-2 Chunk Planner** | Dry-run: chunk boundaries and exact per-chunk prompts without loading the diffusion model. |
| **GAP Phase Gate** | The 1=analyze / 2=generate switch (uses ExecutionBlocker — no muting). |

## New render vs resume (cache_mode)

- **`new render`** (default) — every queue renders fresh. Finished chunks are still
  checkpointed to `output/gap_scail2_cache/<cache_id>/` as the render progresses.
- **`resume`** — continue a crashed/interrupted render from its checkpoints. Also required
  for **`chunk_rerender`**: set e.g. `3` (or `2,5` / `4-6`) plus a new seed to regenerate only
  those chunks; `rerender_cascade` also regenerates everything after them for perfect
  continuity. If the video or settings changed, resume detects the mismatch and safely starts
  a new render.
- **`disabled`** — no disk cache at all.

## Non-human characters (toy bear, robot, creature…)

If the output "takes the colors but keeps a human body":

1. Set that character's **SUBJECT text** to what the image contains (e.g. `pink plush teddy
   bear`) and confirm the **REF MASK preview** shows its full silhouette — in replacement mode
   the reference is composited through this mask, so an empty mask means the model never sees
   the character.
2. Describe it explicitly in **character_prompts** (`1: a fluffy pink plush teddy bear with
   round ears and stubby arms`).
3. **Disable turbo** — cfg 1 barely follows prompts for drastic changes: bypass the distill
   LoRA and set **steps 40, cfg 5**.
4. **Loosen the pose**: `pose_strength ≈ 0.7`, `pose_end ≈ 0.8` (lower for very different
   body proportions).

## Prompting strategy

- **base_prompt** — scene, style, camera; put `{characters}` where descriptions belong.
- **character_prompts** — one line per character (`1: description`); injected only in chunks
  where that character is on screen, so entering/leaving characters are handled automatically.
- **prompt_schedule** — frame-range actions (`0-151: they dance`); generate the skeleton with
  the footage analysis and fill in the blanks. Empty ranges fall back to `base_prompt`.

## Memory guide

- `chunk_length` is the VRAM knob — 81 is the trained value, drop to 65/49 when tight.
- `vae_decode = tiled` removes the decode VRAM spike at higher resolutions.
- System RAM: ~11 GB per 1000 source frames at 896×512 (frames + masks), plus the model.
- Wrong-person fixes, remapping, and troubleshooting live in the workflow notes.

---

© Geekatplay Studio — [youtube.com/@geekatplay](https://www.youtube.com/@geekatplay)
