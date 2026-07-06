#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_brain_perception.py

Inference for a trained BrainPerceptionModel (Qwen3-VL + <SEG> + SAM):
load a checkpoint, take one image + a question, generate the text reply, and — if the
model emits a <SEG> token — decode the corresponding segmentation mask.

Pipeline (mirrors the training forward, but with autoregressive generation):
  1. build the chat prompt (system + user[image + question]) and process it
  2. VLM.generate(...) -> text reply (may contain <SEG>)
  3. re-run ONE forward over prompt+generated with output_hidden_states=True to get the
     hidden state at each <SEG> position (generation returns ids, not hidden states)
  4. seg_projection(<SEG> hidden) -> SAM sparse prompt; SAM decodes it against the
     3-channel images_seg -> binary mask(s)

Run:
    PYTHONPATH=$PWD python tools/infer_brain_perception.py \
        --ckpt runs/exp1/best.pt \
        --image /path/to/slice_t1n.png \
        --question "Segment the tumor core." \
        --out runs/exp1/infer_out

    # multi-modal SAM input (3 modalities, matching training's 3-channel images_seg):
    #   --seg_images t1n.png t1c.png t2f.png
    # if --seg_images is omitted, --image is replicated to 3 channels.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.brain_perception_model import (
    BrainPerceptionModel,
    BrainPerceptionModelConfig,
    Qwen3VLSegCollator,
)
from src.dataset.brain_perception_dataset import (
    _default_seg_norm,
    _load_gray_u8,
    _resize_np,
)

SYSTEM_PROMPT = Qwen3VLSegCollator.system_prompt  # exact string used in training

PRECISION_TO_TORCH_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a local training checkpoint.

    The trainer stores optimizer/scheduler state as well as tensors, so use
    weights_only=False when the installed PyTorch exposes that argument.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_precision(requested: str | None, train_args: Dict[str, Any], device: torch.device) -> str:
    precision = requested or train_args.get("precision") or ("bf16" if device.type == "cuda" else "fp32")
    if precision not in PRECISION_TO_TORCH_DTYPE:
        print(f"[precision][warn] unsupported checkpoint precision {precision!r}; using bf16")
        precision = "bf16"
    if device.type != "cuda" and precision != "fp32":
        print(f"[precision][warn] {precision} inference is only enabled on CUDA here; using fp32 on {device.type}")
        precision = "fp32"
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[precision][warn] CUDA device does not report bf16 support; using fp32")
        precision = "fp32"
    return precision


def amp_context(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cast_processor_tensor(key: str, value: Any, device: torch.device, vlm_dtype: torch.dtype | None = None) -> Any:
    if not hasattr(value, "to"):
        return value
    if key.startswith("pixel_values") and vlm_dtype is not None:
        return value.to(device=device, dtype=vlm_dtype)
    return value.to(device)


# ------------------------------------------------------------------
# SAM-branch image (3-channel, same normalization as training)
# ------------------------------------------------------------------
def build_seg_image_infer(paths: List[str], size: int, n_channels: int) -> torch.Tensor:
    """Load up to n_channels grayscale images, resize to (size,size), ImageNet-normalize,
    and stack to (n_channels, size, size). If fewer paths than channels are given, the
    last one is replicated (e.g. a single image -> 3 identical channels)."""
    chans = []
    for p in paths[:n_channels]:
        g = _load_gray_u8(Path(p))
        g = _resize_np(g, size, nearest=False).astype(np.float32)
        chans.append(g)
    if not chans:
        raise ValueError("no seg image paths provided")
    while len(chans) < n_channels:
        chans.append(chans[-1].copy())
    img = np.stack(chans[:n_channels], axis=0)  # (C, H, W) in 0..255
    mean_l, std_l = _default_seg_norm(n_channels)
    mean = np.array(mean_l, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(std_l, dtype=np.float32).reshape(-1, 1, 1)
    img = (img / 255.0 - mean) / std
    return torch.from_numpy(img).float()


# ------------------------------------------------------------------
# Model construction + checkpoint load
# ------------------------------------------------------------------
def build_model(args, device) -> BrainPerceptionModel:
    ckpt = load_checkpoint(args.ckpt)
    ta = ckpt.get("args", {}) or {}   # training argparse vars, if saved
    args.precision = resolve_precision(args.precision, ta, device)

    def pick(cli, key, default):
        return cli if cli is not None else ta.get(key, default)

    cfg = BrainPerceptionModelConfig(
        vlm_name_or_path=pick(args.vlm, "vlm", BrainPerceptionModelConfig.vlm_name_or_path),
        vlm_source=pick(args.vlm_source, "vlm_source", BrainPerceptionModelConfig.vlm_source),
        vlm_revision=pick(args.vlm_revision, "vlm_revision", None),
        vlm_cache_dir=pick(args.vlm_cache_dir, "vlm_cache_dir", None),
        sam_version=pick(args.sam_version, "sam_version", BrainPerceptionModelConfig.sam_version),
        sam_name_or_path=pick(args.sam, "sam", BrainPerceptionModelConfig.sam_name_or_path),
        seg_image_size=pick(args.seg_size, "seg_size", 1024),
        mask_size=pick(args.mask_size, "mask_size", 1024),
        torch_dtype=PRECISION_TO_TORCH_DTYPE[args.precision],
    )
    print(f"[cfg] vlm={cfg.vlm_source}:{cfg.vlm_name_or_path} sam={cfg.sam_version}:{cfg.sam_name_or_path} "
          f"seg_size={cfg.seg_image_size} mask_size={cfg.mask_size} "
          f"seg_in_ch={cfg.seg_in_channels} precision={args.precision}")

    model = BrainPerceptionModel(cfg)
    state = ckpt.get("trainable_state_dict") or ckpt.get("state_dict") or ckpt
    info = model.load_state_dict(state, strict=False)
    # strict=False: 'missing' = the frozen base weights (expected, not in the ckpt);
    # 'unexpected' should be ~0 — anything here means a key/name mismatch worth checking.
    print(f"[ckpt] loaded {len(state)} tensor entries "
          f"(step {ckpt.get('step','?')}, best_dice {ckpt.get('best_val_dice','?')}); "
          f"unexpected={len(info.unexpected_keys)}")
    if info.unexpected_keys:
        print(f"[ckpt][warn] unexpected keys (first few): {info.unexpected_keys[:5]}")
    return model.eval().to(device)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
@torch.no_grad()
def run(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args, device)
    cfg = model.cfg
    processor = model.processor
    vlm_dtype = next(model.vlm.parameters()).dtype
    sam_dtype = next(model.sam_head.parameters()).dtype

    # ---- 1. build + process the prompt (same shape as training's collator) ----
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": args.image},
                                     {"type": "text", "text": args.question}]},
    ]
    from qwen_vl_utils import process_vision_info  # ships with Qwen-VL
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    enc = processor(text=[text], images=image_inputs, return_tensors="pt",
                    return_mm_token_type_ids=True)
    enc = {k: cast_processor_tensor(k, v, device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    # ---- 2. images_seg for the SAM branch (3-ch, matching training) ----
    seg_paths = args.seg_images if args.seg_images else [args.image]
    images_seg = build_seg_image_infer(seg_paths, cfg.seg_image_size,
                                        cfg.seg_in_channels).unsqueeze(0).to(device)

    # ---- 3. generate the text reply ----
    gen_kwargs = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids"):
        if key in enc:
            gen_kwargs[key] = cast_processor_tensor(key, enc[key], device, vlm_dtype)
    with amp_context(args.precision, device):
        gen = model.vlm.generate(**gen_kwargs)          # (1, prompt_len + new)
    new_ids = gen[0, prompt_len:]
    reply = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    reply_raw = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
    print("\n=== reply ===")
    print(reply)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reply.txt").write_text(reply_raw, encoding="utf-8")

    # ---- 4. re-run one forward to get the <SEG> hidden states ----
    full_ids = gen                                       # (1, L_full)
    fwd_kwargs = dict(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        output_hidden_states=True,
        return_dict=True,
    )
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                "video_grid_thw", "second_per_grid_ts"):
        if key in enc:
            fwd_kwargs[key] = cast_processor_tensor(key, enc[key], device, vlm_dtype)
    if "mm_token_type_ids" in enc:
        # Reuse the processor's exact image-token mask (avoids a fragile image_token_id
        # re-lookup that could silently yield all-zeros -> wrong M-RoPE -> wrong mask).
        # The generated tail contains no image tokens, so zero-pad it to the full length.
        pad = full_ids.shape[1] - prompt_len
        fwd_kwargs["mm_token_type_ids"] = F.pad(enc["mm_token_type_ids"], (0, pad), value=0)
    with amp_context(args.precision, device):
        out = model.vlm(**fwd_kwargs)
    hidden = out.hidden_states[-1]                        # (1, L_full, D)
    seg_positions = torch.zeros_like(full_ids, dtype=torch.bool)
    seg_positions[:, prompt_len:] = full_ids[:, prompt_len:] == model.seg_token_id
    seg_embeds = hidden[seg_positions]                   # (n_seg, D)
    n_seg = int(seg_embeds.shape[0])
    print(f"[seg] <SEG> tokens emitted: {n_seg}")
    if n_seg == 0:
        print("[seg] no segmentation requested by the model — text-only reply.")
        print(f"[out] wrote reply.txt to {out_dir}")
        return

    # ---- 5. <SEG> -> SAM prompt -> mask ----
    with amp_context(args.precision, device):
        sparse = model.seg_projection(seg_embeds.to(model.seg_projection[0].weight.dtype))
        sparse = sparse.unsqueeze(1)                     # (n_seg, 1, sam_prompt_dim)
        feats = model.sam_head.encode_image(images_seg.to(sam_dtype))
        image_embed = feats["image_embed"]               # (1, c, h, w)
        idx = torch.zeros(n_seg, dtype=torch.long, device=device)   # all <SEG> -> the single image
        feats_exp = {
            "image_embed": image_embed[idx],
            "high_res_feats": ([f[idx] for f in feats["high_res_feats"]]
                               if feats.get("high_res_feats") is not None else None),
        }
        low_res = model.sam_head.decode(feats_exp, sparse.to(image_embed.dtype))  # (n_seg,1,low,low)
        pred = F.interpolate(low_res.float(), size=(cfg.mask_size, cfg.mask_size),
                             mode="bilinear", align_corners=False).squeeze(1)      # (n_seg,H,W)
    masks = (pred.sigmoid() > args.mask_threshold).cpu().numpy().astype(np.uint8)  # (n_seg,H,W)

    # ---- 6. save masks (at the original image resolution) + overlays ----
    base_img = Image.open(args.image).convert("RGB")
    W, H = base_img.size
    for i in range(n_seg):
        m = Image.fromarray(masks[i] * 255).resize((W, H), Image.NEAREST)
        m.save(out_dir / f"mask_{i}.png")
        # red overlay for a quick visual check
        overlay = np.array(base_img).copy()
        mm = np.array(m) > 127
        overlay[mm] = (0.5 * overlay[mm] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(out_dir / f"overlay_{i}.png")
    print(f"[out] wrote {n_seg} mask(s) + overlay(s) and reply.txt to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to a trained checkpoint (best.pt / last.pt)")
    ap.add_argument("--image", required=True, help="image shown to the VLM (the question is about it)")
    ap.add_argument("--question", required=True, help="the user question / instruction")
    ap.add_argument("--out", default="infer_out", help="output dir for masks/overlays/reply")
    ap.add_argument("--seg_images", nargs="*", default=None,
                    help="optional N modality images for the SAM branch (default: replicate --image)")
    # overrides (default: taken from the checkpoint's saved training args)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--vlm_source", choices=["huggingface", "modelscope"], default=None)
    ap.add_argument("--vlm_revision", default=None)
    ap.add_argument("--vlm_cache_dir", default=None)
    ap.add_argument("--sam_version", choices=["sam2", "sam3"], default=None)
    ap.add_argument("--sam", default=None)
    ap.add_argument("--seg_size", type=int, default=None)
    ap.add_argument("--mask_size", type=int, default=None)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None,
                    help="inference precision (default: checkpoint precision, or bf16 on CUDA / fp32 on CPU)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
