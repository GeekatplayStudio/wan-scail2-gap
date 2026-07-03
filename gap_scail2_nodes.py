"""GAP SCAIL-2 long-video multi-character replacement nodes.

by Geekatplay Studio - https://www.youtube.com/@geekatplay
https://github.com/GeekatplayStudio/wan-scail2-gap

Automates what the stock SCAIL-2 template does manually: chunked generation of
arbitrarily long videos (with previous-frame anchoring between chunks), multiple
reference characters, and per-chunk dynamic prompts driven by a frame schedule
and/or by which identities are actually visible in the SAM3 mask video.

Requires a ComfyUI build with comfy_extras.nodes_scail (WanSCAILToVideo / PR#14373).
"""

import hashlib
import json
import logging
import math
import os
import re

import torch
import torch.nn.functional as F

import nodes
import comfy.samplers
import comfy.utils
import comfy.model_management

log = logging.getLogger("GAP.SCAIL2")


def _progress_text(text, unique_id):
    """Show live status text on the node in the UI (best effort)."""
    if unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(text, unique_id)
    except Exception:
        pass

# Palette must match comfy_extras.nodes_scail.DEFAULT_PALETTE — SCAIL-2 was
# trained on these exact colors, identity i == palette[i].
PALETTE = [
    (0.0, 0.0, 1.0),  # 1 blue
    (1.0, 0.0, 0.0),  # 2 red
    (0.0, 1.0, 0.0),  # 3 green
    (1.0, 0.0, 1.0),  # 4 magenta
    (0.0, 1.0, 1.0),  # 5 cyan
    (1.0, 1.0, 0.0),  # 6 yellow
]
PALETTE_NAMES = ["blue", "red", "green", "magenta", "cyan", "yellow"]

_ON_THRESH = 225.0 / 255.0  # same threshold nodes_scail uses to read mask colors
MAX_CHARACTERS = 6


def _get_scail():
    try:
        from comfy_extras.nodes_scail import WanSCAILToVideo
        return WanSCAILToVideo
    except ImportError as e:
        raise RuntimeError(
            "comfy_extras.nodes_scail not found. Update ComfyUI to a version that "
            "includes SCAIL-2 support (PR#14373)."
        ) from e


def _four_n_plus_1(n):
    return ((max(int(n), 1) - 1) // 4) * 4 + 1


def _plan_chunks(total_frames, chunk_length, overlap):
    """Chunk plan mirroring WanSCAILToVideo offset bookkeeping.

    Returns a list of (source_start, length, new_frames) tuples where
    source_start is the pose-video frame the chunk begins at, and new_frames is
    how many non-overlap frames the chunk contributes to the stitched output.
    """
    chunks = []
    offset = 0
    first = True
    while True:
        eff = offset if first else max(0, offset - overlap)
        remaining = total_frames - eff
        if remaining <= 0:
            break
        if not first and remaining <= overlap:
            break
        length = min(chunk_length, _four_n_plus_1(remaining))
        if not first and length <= overlap:
            break
        chunks.append((eff, length, length if first else length - overlap))
        offset = eff + length
        first = False
    return chunks


def _detect_cuts(frames, threshold=0.3, min_shot=9, batch=256):
    """Hard-cut detection: frame indices (excluding 0) that start a new shot.

    Mean absolute difference between consecutive frames, downscaled to 64x64.
    A cut needs a score >= threshold and both neighboring shots >= min_shot."""
    total = frames.shape[0]
    if total < 2 or threshold <= 0:
        return []
    scores = []
    prev_tail = None
    for i in range(0, total, batch):
        blk = frames[i:i + batch, ..., :3].movedim(-1, 1).float()
        small = F.interpolate(blk, size=(64, 64), mode="area")
        merged = small if prev_tail is None else torch.cat([prev_tail, small], dim=0)
        if merged.shape[0] > 1:
            scores.append((merged[1:] - merged[:-1]).abs().mean(dim=(1, 2, 3)))
        prev_tail = small[-1:]
    diff = torch.cat(scores)  # diff[t-1] = change going into frame t
    cuts = []
    last = 0
    for t in range(1, total):
        if diff[t - 1].item() >= threshold and t - last >= min_shot and total - t >= min_shot:
            cuts.append(t)
            last = t
    return cuts


def _plan_chunks_ex(total_frames, chunk_length, overlap, cuts=()):
    """Shot-aware chunk plan. Each shot (between cuts) is chunked independently;
    the first chunk of a shot is unanchored (no previous-frame conditioning
    across a camera cut). Returns a list of dicts."""
    bounds = [0] + sorted({c for c in cuts if 0 < c < total_frames}) + [total_frames]
    out = []
    for shot in range(len(bounds) - 1):
        a, b = bounds[shot], bounds[shot + 1]
        for j, (s, length, new) in enumerate(_plan_chunks(b - a, chunk_length, overlap)):
            out.append({
                "start": a + s, "length": length, "new": new,
                "anchored": j > 0, "shot": shot, "shot_start": a, "shot_len": b - a,
            })
    return out


def _parse_schedule(text):
    """Parse 'start-end: prompt' / 'start: prompt' lines into [start, end, prompt]."""
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, sep, prompt = line.partition(":")
        if not sep:
            continue
        head, prompt = head.strip(), prompt.strip()
        try:
            if "-" in head:
                a, b = head.split("-", 1)
                start = int(a) if a.strip() else 0
                end = int(b) if b.strip() else None
            else:
                start, end = int(head), None
        except ValueError:
            continue
        entries.append([start, end, prompt])
    entries.sort(key=lambda e: e[0])
    for i, e in enumerate(entries):
        if e[1] is None:
            e[1] = entries[i + 1][0] - 1 if i + 1 < len(entries) else 10 ** 9
    return entries


def _parse_character_prompts(text):
    """Parse '1: description' lines (1-based character index) into {0-based: desc}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, sep, desc = line.partition(":")
        if not sep:
            continue
        head, desc = head.strip().lower(), desc.strip()
        idx = None
        if head.isdigit():
            idx = int(head) - 1
        elif head in PALETTE_NAMES:
            idx = PALETTE_NAMES.index(head)
        if idx is not None and 0 <= idx < MAX_CHARACTERS and desc:
            out[idx] = desc
    return out


def _resolve_schedule(entries, base_prompt, midpoint):
    for start, end, prompt in entries:
        # empty prompt = unfilled template marker -> fall back to base_prompt
        if start <= midpoint <= end and prompt:
            return prompt
    return base_prompt


def _identity_combos(mask_frames):
    """Per-identity boolean masks [T,H,W] for each palette color."""
    on = mask_frames[..., :3] > _ON_THRESH
    R, G, B = on[..., 0], on[..., 1], on[..., 2]
    return [
        (~R) & (~G) & B,  # blue
        R & (~G) & (~B),  # red
        (~R) & G & (~B),  # green
        R & (~G) & B,     # magenta
        (~R) & G & B,     # cyan
        R & G & (~B),     # yellow
    ]


def _detect_identities(mask_frames, presence_threshold):
    """Which palette identities appear in a colored SCAIL-2 mask chunk.

    presence_threshold is the pixel fraction an identity must cover in its
    best frame of the chunk.
    """
    if mask_frames.shape[0] == 0:
        return []
    present = []
    for idx, m in enumerate(_identity_combos(mask_frames)):
        peak = m.flatten(1).float().mean(dim=1).max().item()
        if peak >= presence_threshold:
            present.append(idx)
    return present


def _presence_matrix(mask_video, presence_threshold, batch=64):
    """Per-frame identity presence over the whole mask video -> bool tensor [T, 6].

    Processed in frame batches to keep the transient boolean masks small."""
    rows = []
    for i in range(0, mask_video.shape[0], batch):
        combos = _identity_combos(mask_video[i:i + batch])
        rows.append(torch.stack(
            [c.flatten(1).float().mean(dim=1) >= presence_threshold for c in combos], dim=1))
    return torch.cat(rows, dim=0)


def _intervals(present, gap_tolerance, min_duration):
    """Bool-per-frame -> list of (start, end) inclusive intervals; gaps up to
    gap_tolerance frames are bridged, intervals shorter than min_duration dropped."""
    raw = []
    start = None
    for t, p in enumerate(list(present) + [False]):
        if p and start is None:
            start = t
        elif not p and start is not None:
            raw.append([start, t - 1])
            start = None
    merged = []
    for iv in raw:
        if merged and iv[0] - merged[-1][1] - 1 <= gap_tolerance:
            merged[-1][1] = iv[1]
        else:
            merged.append(iv)
    return [(a, b) for a, b in merged if b - a + 1 >= min_duration]


def _fmt_range(a, b, fps):
    if fps and fps > 0:
        return f"{a}-{b} ({a / fps:.1f}s-{(b + 1) / fps:.1f}s)"
    return f"{a}-{b}"


def _analyze_timeline(mask_video, presence_threshold, gap_tolerance, min_duration, fps=0.0):
    """Analyze a colored mask video: which characters appear, when they
    enter/leave, and a fill-in-the-action schedule template.

    Returns (timeline_text, schedule_template_text, character_count)."""
    total = mask_video.shape[0]
    matrix = _presence_matrix(mask_video, presence_threshold)
    char_intervals = {}
    for c in range(MAX_CHARACTERS):
        ivs = _intervals(matrix[:, c].tolist(), gap_tolerance, min_duration)
        if ivs:
            char_intervals[c] = ivs

    n_chars = len(char_intervals)
    lines = [
        "GAP character timeline",
        f"frames analyzed: {total}" + (f" @ {fps:.6g} fps ({total / fps:.1f}s)" if fps and fps > 0 else ""),
        f"characters detected: {n_chars} -> you need {n_chars} reference image(s)",
        "",
    ]
    for c, ivs in char_intervals.items():
        spans = ", ".join(_fmt_range(a, b, fps) for a, b in ivs)
        lines.append(f"character {c + 1} ({PALETTE_NAMES[c]}): frames {spans}")
    if not char_intervals:
        lines.append("no characters detected - check presence_threshold / SAM3 text prompt")
    timeline = "\n".join(lines)

    # smoothed per-frame visible set -> scene segments where the cast changes
    smooth = torch.zeros(total, MAX_CHARACTERS, dtype=torch.bool)
    for c, ivs in char_intervals.items():
        for a, b in ivs:
            smooth[a:b + 1, c] = True
    segments = []  # (start, end, cast tuple)
    for t in range(total):
        cast = tuple(i for i in range(MAX_CHARACTERS) if smooth[t, i])
        if segments and segments[-1][2] == cast:
            segments[-1][1] = t
        else:
            segments.append([t, t, cast])
    # absorb blips shorter than min_duration into the previous segment
    cleaned = []
    for seg in segments:
        if cleaned and seg[1] - seg[0] + 1 < min_duration:
            cleaned[-1][1] = seg[1]
        else:
            cleaned.append(seg)

    tpl = [
        "# Auto-generated schedule template - fill the action after each range.",
        "# Lines starting with # are ignored; ranges left empty fall back to base_prompt.",
    ]
    for a, b, cast in cleaned:
        if cast:
            who = " + ".join(f"character {i + 1} ({PALETTE_NAMES[i]})" for i in cast)
        else:
            who = "nobody"
        tpl.append(f"# frames {_fmt_range(a, b, fps)} | visible: {who}")
        tpl.append(f"{a}-{b}: ")
    template = "\n".join(tpl)
    return timeline, template, n_chars


def _compose_prompt(scheduled_prompt, char_prompts, identities):
    descriptions = [char_prompts[i] for i in identities if i in char_prompts]
    joined = " ".join(descriptions)
    if "{characters}" in scheduled_prompt:
        return scheduled_prompt.replace("{characters}", joined).strip()
    if joined:
        return (scheduled_prompt + " " + joined).strip()
    return scheduled_prompt


def _match_colors(target, src_anchor, dst_anchor, mode):
    """Reinhard-style mean/std transfer: map src_anchor stats onto dst_anchor
    stats and apply that transform to the whole target chunk. Anchors are the
    overlap frames both chunks generated, so this cancels inter-chunk drift."""
    if mode == "disabled":
        return target
    use_lab = mode == "lab"
    kornia = None
    if use_lab:
        try:
            import kornia  # noqa: F401 — core ComfyUI ships it (ColorTransfer node)
            import kornia.color
        except ImportError:
            use_lab = False

    def to_space(x):
        x = x[..., :3].movedim(-1, 1).float()
        return kornia.color.rgb_to_lab(x) if use_lab else x

    t = to_space(target)
    s = to_space(src_anchor)
    d = to_space(dst_anchor)
    s_mean = s.mean(dim=(0, 2, 3), keepdim=True)
    s_std = s.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    d_mean = d.mean(dim=(0, 2, 3), keepdim=True)
    d_std = d.std(dim=(0, 2, 3), keepdim=True)
    out = (t - s_mean) / s_std * d_std + d_mean
    if use_lab:
        out = kornia.color.lab_to_rgb(out)
    return out.clamp(0.0, 1.0).movedim(1, -1)


def _decode_frames(vae, latent_samples, mode):
    """VAE decode a chunk; 'tiled' trades speed for much lower VRAM."""
    if mode == "tiled":
        tile_size, overlap, temporal_size, temporal_overlap = 512, 64, 64, 8
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None
        compression = vae.spacial_compression_decode()
        frames = vae.decode_tiled(
            latent_samples, tile_x=tile_size // compression, tile_y=tile_size // compression,
            overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
    else:
        frames = vae.decode(latent_samples)
    if frames.ndim == 5:
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    return frames.cpu().float()


def _encode(clip, text, cache):
    if text not in cache:
        tokens = clip.tokenize(text)
        cache[text] = clip.encode_from_tokens_scheduled(tokens)
    return cache[text]


def _build_chunk_report(chunks, per_chunk_info, total_frames, chunk_length, overlap, cuts=()):
    n_shots = (chunks[-1]["shot"] + 1) if chunks else 1
    lines = [
        "GAP SCAIL-2 long video plan",
        f"source frames: {total_frames} | chunk length: {chunk_length} | overlap: {overlap}",
        f"chunks: {len(chunks)} | shots: {n_shots} | output frames: {sum(c['new'] for c in chunks)}",
    ]
    if cuts:
        lines.append(f"scene cuts at frames: {', '.join(str(c) for c in cuts)}")
    lines.append("")
    for i, (ch, info) in enumerate(zip(chunks, per_chunk_info)):
        ids, prompt = info[0], info[1]
        origin = f" | {info[2]}" if len(info) > 2 and info[2] else ""
        shot_txt = f" [shot {ch['shot'] + 1}{'' if ch['anchored'] else ' start'}]" if n_shots > 1 else ""
        id_txt = ", ".join(f"{j + 1}({PALETTE_NAMES[j]})" for j in ids) if ids else "none detected"
        lines.append(f"chunk {i + 1}{shot_txt}: source frames {ch['start']}-{ch['start'] + ch['length'] - 1} (+{ch['new']} new){origin}")
        lines.append(f"  characters: {id_txt}")
        lines.append(f"  prompt: {prompt}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# chunk cache (crash-safe resume + single-chunk re-render)
# ---------------------------------------------------------------------------

def _cache_dir(cache_id):
    import folder_paths
    safe = re.sub(r"[^\w\-.]", "_", cache_id.strip()) or "default"
    d = os.path.join(folder_paths.get_output_directory(), "gap_scail2_cache", safe)
    os.makedirs(d, exist_ok=True)
    return d


def _fingerprint(total_frames, width, height, chunk_length, overlap, replacement_mode, cuts=()):
    """Structural settings that make cached chunks (in)compatible. Prompts and
    seeds are deliberately excluded so single chunks can be re-rendered."""
    key = f"{total_frames}|{width}|{height}|{chunk_length}|{overlap}|{replacement_mode}"
    key += "|cuts:" + ",".join(str(c) for c in cuts)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _chunk_file(cache_dir, index):
    return os.path.join(cache_dir, f"chunk_{index + 1:03d}.pt")


def _clear_cache(cache_dir):
    for f in os.listdir(cache_dir):
        if f.startswith("chunk_") and (f.endswith(".pt") or f.endswith(".tmp")):
            os.remove(os.path.join(cache_dir, f))
    mp = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(mp):
        os.remove(mp)


def _load_manifest(cache_dir):
    p = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.warning("unreadable cache manifest at %s - ignoring", p)
    return None


def _save_manifest(cache_dir, manifest):
    p = os.path.join(cache_dir, "manifest.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, p)


def _save_chunk(cache_dir, index, frames):
    path = _chunk_file(cache_dir, index)
    tmp = path + ".tmp"
    torch.save(frames.half().contiguous(), tmp)
    os.replace(tmp, path)


def _load_chunk(cache_dir, index):
    return torch.load(_chunk_file(cache_dir, index), map_location="cpu", weights_only=True).float()


def _parse_chunk_list(text, n_chunks):
    """'3', '2,5', '4-6' (1-based) -> set of 0-based chunk indices."""
    out = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
            else:
                lo = hi = int(part)
        except ValueError:
            continue
        for i in range(lo, hi + 1):
            if 1 <= i <= n_chunks:
                out.add(i - 1)
    return out


def _plan_actions(n_chunks, done, rerender, cascade):
    """Per chunk: 'load' from cache or 'generate'. Cascade regenerates
    everything from the earliest re-rendered chunk onward."""
    cascade_from = min(rerender) if (rerender and cascade) else None
    actions = []
    for i in range(n_chunks):
        force = i in rerender or (cascade_from is not None and i >= cascade_from)
        actions.append("generate" if force or i not in done else "load")
    return actions


class GAPSCAIL2LongVideo:
    """One-queue-press long-video SCAIL-2 character replacement.

    Slices the driving video into overlapping chunks, generates each chunk with
    WanSCAILToVideo + sampling, anchors every chunk on the previous one, adjusts
    the prompt per chunk, and stitches the result.
    """

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SCAIL-2 model with LoRAs + ModelSamplingSD3 applied."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pose_video": ("IMAGE", {"tooltip": "ALL frames of the driving video (already resized)."}),
                "pose_video_mask": ("IMAGE", {"tooltip": "Full-length colored mask video from SCAIL2ColoredMask."}),
                "reference_image": ("IMAGE", {"tooltip": "Reference character batch (e.g. from GAP Multi-Character Reference)."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "Matching colored reference masks."}),
                "base_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Scene description used when no schedule entry matches. Use {characters} to place character descriptions."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "One line per character: '1: description'. 1=blue, 2=red, 3=green, 4=magenta, 5=cyan, 6=yellow. Appended automatically when that character is visible in the chunk."}),
                "prompt_schedule": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional per-frame-range prompts, one per line: '0-152: prompt' or '153: prompt'. Chunk midpoint picks the entry; falls back to base_prompt."}),
                "width": ("INT", {"default": 896, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "chunk_length": ("INT", {"default": 81, "min": 9, "max": 321, "step": 4, "tooltip": "Frames per chunk (4n+1). SCAIL-2 was trained at 81. Lower = less VRAM."}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 33, "step": 4, "tooltip": "Anchor frames carried into the next chunk (4n+1). SCAIL-2 trained at 5."}),
                "replacement_mode": ("BOOLEAN", {"default": True, "tooltip": "True = replace tracked people, False = animation mode."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Per-chunk seed = seed + chunk index."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "6 with the distill LoRA (turbo), ~40 without."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "1.0 with the distill LoRA (turbo), ~5 without."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "color_match": (["lab", "rgb", "disabled"], {"default": "lab", "tooltip": "Match each chunk's colors to the previous chunk via the shared overlap frames (fights drift)."}),
                "auto_character_prompts": ("BOOLEAN", {"default": True, "tooltip": "Detect which characters are visible per chunk from the mask colors and inject their descriptions."}),
                "presence_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005, "tooltip": "Min pixel fraction (best frame of the chunk) for a character to count as present."}),
                "cache_mode": (["new render", "resume", "disabled"], {"default": "new render", "tooltip": "new render: fresh generation, previous chunks of this cache_id are cleared (finished chunks are still checkpointed as you go). resume: continue a crashed/interrupted render from its checkpoints - also required for chunk_rerender. disabled: no disk cache at all. Cache lives in output/gap_scail2_cache/<cache_id> (~220 MB per 81-frame chunk at 896x512)."}),
                "cache_id": ("STRING", {"default": "default", "tooltip": "Cache folder name - use a different id per video/project."}),
                "chunk_rerender": ("STRING", {"default": "", "tooltip": "Re-generate specific cached chunks (1-based): '3', '2,5', '4-6'. Change the seed for a different take. The next chunk keeps its old anchor, so a subtle seam is possible - enable rerender_cascade for perfect continuity."}),
                "rerender_cascade": ("BOOLEAN", {"default": False, "tooltip": "Also regenerate every chunk after the earliest re-rendered one (perfect continuity, more compute)."}),
                "detect_scene_cuts": ("BOOLEAN", {"default": True, "tooltip": "Detect hard camera cuts in the driving video, align chunk boundaries to them and reset the previous-frame anchor at each cut (prevents ghosting across cuts)."}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.01, "tooltip": "Mean frame-difference (0-1, at 64x64) that counts as a hard cut. Lower = more sensitive."}),
                "vae_decode": (["standard", "tiled"], {"default": "standard", "tooltip": "tiled: decode each chunk in tiles - slower but much less VRAM at high resolutions."}),
                "pad_to_source_length": ("BOOLEAN", {"default": True, "tooltip": "Repeat each shot's last frame to exactly match the source frame count (keeps audio in sync; 4n+1 rounding otherwise drops up to a few frames per shot)."}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def generate(self, model, clip, vae, pose_video, pose_video_mask, reference_image,
                 reference_image_mask, base_prompt, negative_prompt, character_prompts,
                 prompt_schedule, width, height, chunk_length, overlap, replacement_mode,
                 pose_strength, pose_start, pose_end, seed, steps, cfg, sampler_name,
                 scheduler, color_match, auto_character_prompts, presence_threshold,
                 cache_mode="new render", cache_id="default", chunk_rerender="",
                 rerender_cascade=False, detect_scene_cuts=True, scene_cut_threshold=0.3,
                 vae_decode="standard", pad_to_source_length=True,
                 clip_vision_output=None, unique_id=None):
        WanSCAILToVideo = _get_scail()

        chunk_length = _four_n_plus_1(chunk_length)
        overlap = _four_n_plus_1(overlap)
        if overlap >= chunk_length:
            raise ValueError(f"overlap ({overlap}) must be smaller than chunk_length ({chunk_length})")

        total_frames = pose_video.shape[0]
        cuts = _detect_cuts(pose_video, scene_cut_threshold) if detect_scene_cuts else []
        if cuts:
            log.info("scene cuts detected at frames: %s", cuts)
        chunks = _plan_chunks_ex(total_frames, chunk_length, overlap, cuts)
        if not chunks:
            raise ValueError("pose_video has no frames to process")
        n_chunks = len(chunks)

        # ---- chunk cache: new render vs resume vs disabled ----
        use_cache = cache_mode != "disabled"
        rerender = _parse_chunk_list(chunk_rerender, n_chunks)
        cache_path = None
        manifest = None
        done = set()
        if use_cache:
            cache_path = _cache_dir(cache_id)
            fp = _fingerprint(total_frames, width, height, chunk_length, overlap, replacement_mode,
                              cuts=cuts)
            manifest = _load_manifest(cache_path)
            stale = manifest is not None and manifest.get("fingerprint") != fp
            if cache_mode == "new render" or stale:
                if stale and cache_mode == "resume":
                    log.warning("cache '%s' belongs to a different job (video/settings changed) - "
                                "starting a new render instead of resuming", cache_id)
                _clear_cache(cache_path)
                manifest = None
            if manifest is None:
                manifest = {"fingerprint": fp, "chunks": {}}
            if cache_mode == "resume":
                for k, meta in manifest.get("chunks", {}).items():
                    i = int(k)
                    if (i < n_chunks and meta.get("done")
                            and meta.get("start") == chunks[i]["start"]
                            and meta.get("length") == chunks[i]["length"]
                            and meta.get("anchored", i > 0) == chunks[i]["anchored"]
                            and os.path.exists(_chunk_file(cache_path, i))):
                        done.add(i)
        if rerender and cache_mode != "resume":
            log.warning("chunk_rerender requires cache_mode=resume - ignoring it")
            rerender = set()

        actions = _plan_actions(n_chunks, done, rerender, rerender_cascade)
        n_cached = actions.count("load")

        schedule = _parse_schedule(prompt_schedule)
        char_prompts = _parse_character_prompts(character_prompts)
        enc_cache = {}
        negative_cond = None  # encoded lazily, only if something actually generates

        out_expected = sum(c["new"] for c in chunks)
        n_shots = chunks[-1]["shot"] + 1
        log.info("plan: %d source frames -> %d chunk(s) of %d frames (overlap %d), %d shot(s), %d output frames",
                 total_frames, n_chunks, chunk_length, overlap, n_shots, out_expected)
        if use_cache:
            log.info("cache '%s' (%s): reusing %d cached chunk(s), generating %d",
                     cache_id, cache_path, n_cached, n_chunks - n_cached)
        for i, ch in enumerate(chunks):
            log.info("  chunk %d/%d [shot %d]: source frames %d-%d (+%d new) [%s]",
                     i + 1, n_chunks, ch["shot"] + 1, ch["start"], ch["start"] + ch["length"] - 1,
                     ch["new"], actions[i])
        _progress_text(
            f"planned {n_chunks} chunk(s) / {n_shots} shot(s): {n_cached} cached, {n_chunks - n_cached} to generate",
            unique_id)

        pbar = comfy.utils.ProgressBar(n_chunks)
        out_segments = []
        seg_shots = []
        per_chunk_info = []
        prev_frames = None

        for chunk_index, ch in enumerate(chunks):
            comfy.model_management.throw_exception_if_processing_interrupted()
            start, length, anchored = ch["start"], ch["length"], ch["anchored"]
            if not anchored:
                prev_frames = None  # never anchor across a scene cut

            midpoint = start + length // 2
            identities = []
            if auto_character_prompts:
                identities = _detect_identities(pose_video_mask[start:start + length], presence_threshold)
            scheduled = _resolve_schedule(schedule, base_prompt, midpoint)
            prompt = _compose_prompt(scheduled, char_prompts, identities) if auto_character_prompts else scheduled

            if actions[chunk_index] == "load":
                frames = _load_chunk(cache_path, chunk_index)
                meta = manifest["chunks"].get(str(chunk_index), {})
                cached_prompt = meta.get("prompt", prompt)
                cached_seed = meta.get("seed")
                origin = "cached" + (f" (seed {cached_seed})" if cached_seed is not None else "")
                per_chunk_info.append((identities, cached_prompt, origin))
                _progress_text(f"chunk {chunk_index + 1}/{n_chunks}: cached", unique_id)
                log.info("chunk %d/%d: loaded from cache (%d frames)", chunk_index + 1, n_chunks, frames.shape[0])
            else:
                chunk_seed = seed + chunk_index
                id_txt = ",".join(str(i + 1) for i in identities) if identities else "-"
                _progress_text(
                    f"chunk {chunk_index + 1}/{n_chunks} | frames {start}-{start + length - 1} | characters: {id_txt}",
                    unique_id)
                log.info("chunk %d/%d: generating frames %d-%d | seed %d | characters: %s | prompt: %s",
                         chunk_index + 1, n_chunks, start, start + length - 1, chunk_seed, id_txt, prompt)

                if negative_cond is None:
                    negative_cond = _encode(clip, negative_prompt, enc_cache)
                positive_cond = _encode(clip, prompt, enc_cache)

                offset_in = start + (overlap if anchored else 0)
                ret = WanSCAILToVideo.execute(
                    positive_cond, negative_cond, vae, width, height, length, 1,
                    pose_strength, pose_start, pose_end, offset_in, overlap,
                    replacement_mode=replacement_mode,
                    reference_image=reference_image,
                    clip_vision_output=clip_vision_output,
                    pose_video=pose_video,
                    pose_video_mask=pose_video_mask,
                    reference_image_mask=reference_image_mask,
                    previous_frames=prev_frames if anchored else None,
                )
                positive_c, negative_c, latent, _ = ret.args

                samples = nodes.common_ksampler(
                    model, chunk_seed, steps, cfg, sampler_name, scheduler,
                    positive_c, negative_c, latent, denoise=1.0)[0]

                frames = _decode_frames(vae, samples["samples"], vae_decode)

                if anchored and prev_frames is not None:
                    frames = _match_colors(frames, frames[:overlap], prev_frames[-overlap:], color_match)
                per_chunk_info.append((identities, prompt, f"generated (seed {chunk_seed})"))

                if use_cache:
                    _save_chunk(cache_path, chunk_index, frames)
                    manifest["chunks"][str(chunk_index)] = {
                        "start": start, "length": length, "seed": chunk_seed,
                        "prompt": prompt, "anchored": anchored, "done": True,
                    }
                    _save_manifest(cache_path, manifest)

                del samples, latent, ret, positive_c, negative_c
                comfy.model_management.soft_empty_cache()

            out_segments.append(frames if not anchored else frames[overlap:])
            seg_shots.append(ch["shot"])
            prev_frames = frames
            pbar.update(1)

        # stitch per shot; pad each shot to its source length to keep A/V sync
        shot_lens = {ch["shot"]: ch["shot_len"] for ch in chunks}
        parts = []
        for shot in range(n_shots):
            segs = [s for s, sid in zip(out_segments, seg_shots) if sid == shot]
            if not segs:
                continue
            seg = torch.cat(segs, dim=0)
            target = shot_lens[shot]
            if pad_to_source_length and seg.shape[0] < target:
                pad = seg[-1:].repeat(target - seg.shape[0], 1, 1, 1)
                seg = torch.cat([seg, pad], dim=0)
            parts.append(seg)
        result = torch.cat(parts, dim=0)

        report = _build_chunk_report(chunks, per_chunk_info, total_frames, chunk_length, overlap, cuts=cuts)
        if use_cache:
            report = report.replace("\n\n", f"\ncache: '{cache_id}' | reused {n_cached} | generated {n_chunks - n_cached}\n\n", 1)
        _progress_text(f"done: {n_chunks} chunk(s) ({n_cached} cached), {result.shape[0]} frames", unique_id)
        log.info("done: %d chunks (%d cached), %d output frames", n_chunks, n_cached, result.shape[0])
        return (result, report)


def _prep_ref(image, mask, width, height, color_idx, bg_value, device):
    """Resize one reference image + mask to generation size; render the mask in
    the identity's palette color on the mode-appropriate background."""
    img = comfy.utils.common_upscale(
        image[:1, :, :, :3].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1).to(device)
    if mask is None:
        m = torch.ones((1, height, width), device=device)
    else:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # same scale + center-crop geometry as the image, so mask stays aligned
        m = comfy.utils.common_upscale(
            mask[:1].unsqueeze(1).float().to(device), width, height, "nearest-exact", "center").squeeze(1)
    color = torch.tensor(PALETTE[color_idx], device=device).view(1, 1, 1, 3)
    bg = torch.full((1, height, width, 3), bg_value, device=device)
    colored = torch.where((m > 0.5).unsqueeze(-1), color.expand(1, height, width, 3), bg)
    return img, colored


class GAPMultiCharacterReference:
    """Build the SCAIL-2 multi-identity reference batch from separate character
    images. Character N gets palette color N; in the driving video, identity
    colors are assigned by SCAIL2ColoredMask sort order (default left_to_right:
    character 1 replaces the person appearing first/leftmost)."""

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("reference_image", "reference_image_mask", "character_count")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "width": ("INT", {"default": 896, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
            "replacement_mode": ("BOOLEAN", {"default": True, "tooltip": "Must match the orchestrator/SCAIL2ColoredMask setting. Controls mask background color."}),
            "image_1": ("IMAGE",),
        }
        optional = {"mask_1": ("MASK",)}
        for i in range(2, MAX_CHARACTERS + 1):
            optional[f"image_{i}"] = ("IMAGE",)
            optional[f"mask_{i}"] = ("MASK",)
        return {"required": required, "optional": optional}

    def build(self, width, height, replacement_mode, image_1, **kwargs):
        device = comfy.model_management.intermediate_device()
        bg_value = 0.0 if replacement_mode else 1.0  # nodes_scail: ref bg black in replacement mode

        images, colored_masks = [], []
        for i in range(1, MAX_CHARACTERS + 1):
            img = image_1 if i == 1 else kwargs.get(f"image_{i}")
            if img is None:
                continue
            ref, colored = _prep_ref(img, kwargs.get(f"mask_{i}"), width, height, len(images), bg_value, device)
            images.append(ref)
            colored_masks.append(colored)

        return (torch.cat(images, dim=0), torch.cat(colored_masks, dim=0), len(images))


class GAPCharacterExtraView:
    """Append an extra reference view (back view, close-up, different outfit
    angle) for one character. SCAIL-2 uses additional same-color reference
    images to strengthen identity when the person turns or gets small in frame.
    Chain several of these after GAP Multi-Character Reference."""

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("reference_image", "reference_image_mask")
    FUNCTION = "append_view"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {"tooltip": "Batch from GAP Multi-Character Reference (or a previous Extra View)."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "Matching colored mask batch."}),
                "image": ("IMAGE", {"tooltip": "The extra view of the character."}),
                "character": ("INT", {"default": 1, "min": 1, "max": MAX_CHARACTERS, "tooltip": "Which character this view belongs to (1=blue, 2=red, ...). Must match the character's slot in the reference builder."}),
                "replacement_mode": ("BOOLEAN", {"default": True, "tooltip": "Must match the reference builder setting."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Subject mask for the extra view (e.g. from SAM3 Detect). Full image if omitted."}),
            },
        }

    def append_view(self, reference_image, reference_image_mask, image, character, replacement_mode, mask=None):
        device = comfy.model_management.intermediate_device()
        height, width = reference_image.shape[1], reference_image.shape[2]
        bg_value = 0.0 if replacement_mode else 1.0
        img, colored = _prep_ref(image, mask, width, height, character - 1, bg_value, device)
        return (
            torch.cat([reference_image.to(device), img], dim=0),
            torch.cat([reference_image_mask.to(device), colored], dim=0),
        )


class GAPPhaseGate:
    """Two-phase execution without muting nodes.

    Phase 1: everything upstream (tracking, mask check, footage analysis) runs;
    the generator and all its downstream nodes are skipped silently.
    Phase 2: frames pass through and generation runs. Analysis results are
    cached by ComfyUI, so the phase-2 queue goes straight to generation."""

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "gate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "phase": (["1 - analyze only", "2 - generate video"], {"default": "1 - analyze only", "tooltip": "Phase 1: queue runs tracking + mask check + footage analysis, generation is skipped. Review the outputs, fill your prompts, switch to phase 2 and queue again."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def gate(self, frames, phase, unique_id=None):
        if phase.startswith("1"):
            from comfy_execution.graph_utils import ExecutionBlocker
            _progress_text("PHASE 1: analysis only — review mask check + timeline, fill prompts, then set phase 2", unique_id)
            log.info("phase gate: analysis only - generation skipped")
            return (ExecutionBlocker(None),)
        _progress_text("PHASE 2: generating", unique_id)
        return (frames,)


class GAPSCAIL2Planner:
    """Dry-run preview: chunk boundaries and the exact prompt each chunk will
    use, without touching the diffusion model."""

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("plan", "chunk_count")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_length": ("INT", {"default": 81, "min": 9, "max": 321, "step": 4}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 33, "step": 4}),
                "base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_prompts": ("STRING", {"multiline": True, "default": ""}),
                "prompt_schedule": ("STRING", {"multiline": True, "default": ""}),
                "presence_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005}),
                "detect_scene_cuts": ("BOOLEAN", {"default": True, "tooltip": "Needs video_frames wired; detects hard cuts and plans shots exactly like the orchestrator."}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "pose_video_mask": ("IMAGE", {"tooltip": "Colored mask video; enables character presence detection."}),
                "video_frames": ("IMAGE", {"tooltip": "Driving video frames; enables scene-cut detection (and frame count when no mask is wired)."}),
            },
        }

    def plan(self, chunk_length, overlap, base_prompt, character_prompts, prompt_schedule,
             presence_threshold, detect_scene_cuts=True, scene_cut_threshold=0.3,
             pose_video_mask=None, video_frames=None):
        source = pose_video_mask if pose_video_mask is not None else video_frames
        if source is None:
            raise ValueError("Wire either pose_video_mask or video_frames so the planner knows the frame count")
        chunk_length = _four_n_plus_1(chunk_length)
        overlap = _four_n_plus_1(overlap)
        total_frames = source.shape[0]
        cuts = _detect_cuts(video_frames, scene_cut_threshold) if (detect_scene_cuts and video_frames is not None) else []
        chunks = _plan_chunks_ex(total_frames, chunk_length, overlap, cuts)
        schedule = _parse_schedule(prompt_schedule)
        char_prompts = _parse_character_prompts(character_prompts)

        per_chunk_info = []
        for ch in chunks:
            start, length = ch["start"], ch["length"]
            identities = []
            if pose_video_mask is not None:
                identities = _detect_identities(pose_video_mask[start:start + length], presence_threshold)
            scheduled = _resolve_schedule(schedule, base_prompt, start + length // 2)
            per_chunk_info.append((identities, _compose_prompt(scheduled, char_prompts, identities)))

        report = _build_chunk_report(chunks, per_chunk_info, total_frames, chunk_length, overlap, cuts=cuts)
        return (report, len(chunks))


class GAPCharacterTimeline:
    """Pre-analyze the driving footage: how many characters appear, on which
    frames they enter/leave, and a ready-to-fill prompt_schedule template with
    a marker for every stretch where the visible cast changes."""

    CATEGORY = "GAP/SCAIL2"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("timeline", "schedule_template", "character_count")
    FUNCTION = "analyze"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_video_mask": ("IMAGE", {"tooltip": "Full-length colored mask video from SCAIL2ColoredMask."}),
                "presence_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005, "tooltip": "Min pixel fraction of a frame for a character to count as present."}),
                "gap_tolerance": ("INT", {"default": 12, "min": 0, "max": 10000, "tooltip": "Bridge disappearances shorter than this many frames (occlusions, tracking dropouts)."}),
                "min_duration": ("INT", {"default": 8, "min": 1, "max": 10000, "tooltip": "Ignore appearances/cast changes shorter than this many frames."}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 480.0, "step": 0.01, "tooltip": "Wire from GetVideoComponents to also show times in seconds (0 = frames only)."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def analyze(self, pose_video_mask, presence_threshold, gap_tolerance, min_duration, fps=0.0, unique_id=None):
        timeline, template, n_chars = _analyze_timeline(
            pose_video_mask, presence_threshold, gap_tolerance, min_duration, fps)
        _progress_text(f"{n_chars} character(s) detected", unique_id)
        log.info("timeline:\n%s", timeline)
        return (timeline, template, n_chars)


NODE_CLASS_MAPPINGS = {
    "GAPSCAIL2LongVideo": GAPSCAIL2LongVideo,
    "GAPMultiCharacterReference": GAPMultiCharacterReference,
    "GAPCharacterExtraView": GAPCharacterExtraView,
    "GAPSCAIL2Planner": GAPSCAIL2Planner,
    "GAPCharacterTimeline": GAPCharacterTimeline,
    "GAPPhaseGate": GAPPhaseGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GAPSCAIL2LongVideo": "GAP SCAIL-2 Long Video (multi-character)",
    "GAPMultiCharacterReference": "GAP Multi-Character Reference",
    "GAPCharacterExtraView": "GAP Character Extra View",
    "GAPSCAIL2Planner": "GAP SCAIL-2 Chunk Planner",
    "GAPCharacterTimeline": "GAP Character Timeline (footage analysis)",
    "GAPPhaseGate": "GAP Phase Gate (1=analyze / 2=generate)",
}
