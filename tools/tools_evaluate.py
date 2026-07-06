#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from tools.infer_brain_perception import (
    build_model,
    build_seg_image_infer,
    amp_context,
    cast_processor_tensor,
    SYSTEM_PROMPT,
)


def load_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def select_items(items, n_per_task=None, max_samples=None):
    if n_per_task is None:
        return items[:max_samples] if max_samples else items

    buckets = {}
    selected = []
    for item in items:
        task = item.get("task_type", "unknown")
        cnt = buckets.get(task, 0)
        if cnt < n_per_task:
            selected.append(item)
            buckets[task] = cnt + 1
    if max_samples:
        selected = selected[:max_samples]
    return selected


def derive_seg_images(image_path):
    """
    训练验证时默认使用前 3 个模态：t1n, t1c, t2w。
    如果某个模态文件不存在，就自动退回用已有图像复制通道。
    """
    p = str(image_path)
    mods = ["t1n", "t1c", "t2w"]
    paths = []

    for mod in mods:
        q = re.sub(r"-(t1n|t1c|t2w|t2f)_", f"-{mod}_", p)
        if Path(q).exists():
            paths.append(q)

    if not paths:
        paths = [p]

    return paths


def require_file(path, kind):
    """Raise a clear FileNotFoundError that prints the missing path, so a bad
    image/mask path in the manifest is easy to locate instead of surfacing as a
    cryptic error deep inside the image processor."""
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return path


def target_size_wh(image_path, gt_mask_path, cfg):
    """Pick the (W, H) size to save the predicted mask at. Prefer the GT mask size
    so the prediction lines up with scoring; if the GT is unavailable, fall back to
    the input image size; finally to the model's mask_size. The GT mask is NOT
    required for inference — it is only used here to choose an output resolution."""
    for p in (gt_mask_path, image_path):
        if p and Path(p).exists():
            with Image.open(p) as im:
                return im.size  # (W, H)
    return (cfg.mask_size, cfg.mask_size)


def make_messages(image_path, question, answer=None):
    """Build the chat messages. When ``answer`` is given (teacher forcing) the
    assistant turn is appended; when it is None (generate mode) only the system
    and user turns are returned so the model produces the answer itself."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]
    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })
    return messages


@torch.no_grad()
def infer_one(model, item, args, device):
    from qwen_vl_utils import process_vision_info

    cfg = model.cfg
    processor = model.processor
    vlm_dtype = next(model.vlm.parameters()).dtype
    sam_dtype = next(model.sam_head.parameters()).dtype

    image_path = item["image_path"]
    question = item.get("prompt_for_model") or item.get("prompt") or item.get("question")
    gt_mask_path = item["gt_mask_path"]

    # The input image is required for inference; fail early with its exact path.
    require_file(image_path, "image_path")
    # The GT mask is only needed for scoring (done separately), so it is optional
    # here: a missing GT does NOT stop us from producing a prediction.

    reply = None

    if args.mode == "teacher_force":
        # Teacher forcing: feed the canonical answer (with one <SEG>) and read the
        # hidden state at the <SEG> position — measures mask quality in isolation.
        messages = make_messages(image_path, question, args.teacher_answer)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        enc = processor(text=[text], images=image_inputs, return_tensors="pt",
                        return_mm_token_type_ids=True)
        enc = {k: cast_processor_tensor(k, v, device, vlm_dtype) for k, v in enc.items()}

        fwd_kwargs = dict(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            output_hidden_states=True, return_dict=True,
        )
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                    "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids"):
            if key in enc:
                fwd_kwargs[key] = cast_processor_tensor(key, enc[key], device, vlm_dtype)

        with amp_context(args.precision, device):
            out = model.vlm(**fwd_kwargs)
        hidden = out.hidden_states[-1]
        seg_positions = enc["input_ids"] == model.seg_token_id
        seg_embeds = hidden[seg_positions]

    else:
        # Generate mode: the model produces its own answer and must emit <SEG>
        # itself; we then re-run one forward over prompt+generated tokens and take
        # the hidden state at the generated <SEG>. Measures the end-to-end system.
        messages = make_messages(image_path, question, answer=None)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        enc = processor(text=[text], images=image_inputs, return_tensors="pt",
                        return_mm_token_type_ids=True)
        enc = {k: cast_processor_tensor(k, v, device, vlm_dtype) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=False,
        )
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                    "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids"):
            if key in enc:
                gen_kwargs[key] = cast_processor_tensor(key, enc[key], device, vlm_dtype)
        with amp_context(args.precision, device):
            gen = model.vlm.generate(**gen_kwargs)          # (1, prompt_len + new)
        reply = processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=False)

        # re-run a single forward over the full sequence to get <SEG> hidden states
        full_ids = gen
        fwd_kwargs = dict(
            input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
            output_hidden_states=True, return_dict=True,
        )
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                    "video_grid_thw", "second_per_grid_ts"):
            if key in enc:
                fwd_kwargs[key] = cast_processor_tensor(key, enc[key], device, vlm_dtype)
        if "mm_token_type_ids" in enc:
            # image tokens live only in the prompt; zero-pad the generated tail
            pad = full_ids.shape[1] - prompt_len
            fwd_kwargs["mm_token_type_ids"] = F.pad(enc["mm_token_type_ids"], (0, pad), value=0)
        with amp_context(args.precision, device):
            out = model.vlm(**fwd_kwargs)
        hidden = out.hidden_states[-1]
        seg_positions = torch.zeros_like(full_ids, dtype=torch.bool)
        seg_positions[:, prompt_len:] = full_ids[:, prompt_len:] == model.seg_token_id
        seg_embeds = hidden[seg_positions]

    n_seg = int(seg_embeds.shape[0])

    if n_seg == 0:
        # teacher_force: should not happen (guarded); generate: model chose not to
        # segment -> an empty mask is saved by the caller (correctly scores as a miss).
        return {
            "status": "no_seg",
            "n_seg": 0,
            "mask": None,
            "seg_images": [],
            "reply": reply,
        }

    # use the first <SEG> (teacher forcing has exactly one; generate may emit >1)
    seg_embeds = seg_embeds[:1]

    seg_paths = derive_seg_images(image_path)

    images_seg = build_seg_image_infer(
        seg_paths,
        cfg.seg_image_size,
        cfg.seg_in_channels,
    ).unsqueeze(0).to(device)

    with amp_context(args.precision, device):
        sparse = model.seg_projection(seg_embeds.to(model.seg_projection[0].weight.dtype))
        sparse = sparse.unsqueeze(1)

        feats = model.sam_head.encode_image(images_seg.to(sam_dtype))
        image_embed = feats["image_embed"]

        idx = torch.zeros(1, dtype=torch.long, device=device)

        feats_exp = {
            "image_embed": image_embed[idx],
            "high_res_feats": (
                [f[idx] for f in feats["high_res_feats"]]
                if feats.get("high_res_feats") is not None
                else None
            ),
        }

        low_res = model.sam_head.decode(feats_exp, sparse.to(image_embed.dtype))
        pred = F.interpolate(
            low_res.float(),
            size=(cfg.mask_size, cfg.mask_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    mask_1024 = (pred.sigmoid()[0] > args.mask_threshold).cpu().numpy().astype(np.uint8)

    # Save at the GT mask size when available (so it lines up with scoring), else
    # fall back to the input image size. GT is not required to produce a prediction.
    W, H = target_size_wh(image_path, gt_mask_path, cfg)
    mask = Image.fromarray(mask_1024 * 255).resize((W, H), Image.NEAREST)
    mask = np.array(mask).astype(np.uint8)

    return {
        "status": "ok",
        "n_seg": n_seg,
        "mask": mask,
        "seg_images": seg_paths,
        "reply": reply,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--vlm", required=True)
    ap.add_argument("--vlm_source", choices=["huggingface", "modelscope"], default="huggingface")
    ap.add_argument("--vlm_revision", default=None)
    ap.add_argument("--vlm_cache_dir", default=None)

    ap.add_argument("--sam_version", choices=["sam2", "sam3"], default="sam2")
    ap.add_argument("--sam", required=True)

    ap.add_argument("--seg_size", type=int, default=None)
    ap.add_argument("--mask_size", type=int, default=None)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default=None)

    ap.add_argument("--mode", choices=["teacher_force", "generate"], default="teacher_force",
                    help="teacher_force: feed the canonical answer with <SEG> and score the "
                         "mask in isolation. generate: the model produces its own answer and "
                         "must emit <SEG> itself (end-to-end).")
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--teacher_answer", default="好的，<SEG>，记为实例1。",
                    help="Assistant answer injected in teacher_force mode (must contain <SEG>).")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max tokens generated per sample in --mode generate.")

    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--n_per_task", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    mask_dir = out_root / "masks"
    log_dir = out_root / "logs"
    mask_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = out_root / "predictions.jsonl"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model_args = SimpleNamespace(
        ckpt=args.ckpt,
        vlm=args.vlm,
        vlm_source=args.vlm_source,
        vlm_revision=args.vlm_revision,
        vlm_cache_dir=args.vlm_cache_dir,
        sam_version=args.sam_version,
        sam=args.sam,
        seg_size=args.seg_size,
        mask_size=args.mask_size,
        precision=args.precision,
        device=str(device),
    )

    print("[INFO] Loading model...")
    model = build_model(model_args, device)
    model.eval()

    # build_model 会解析 precision，这里同步回来
    args.precision = model_args.precision

    # --- guard: both modes depend on <SEG> being a single known token ---
    # If it is missing/unknown, apply_chat_template would split it, no <SEG> hidden
    # state would be gathered, every mask would come out empty, and Dice ~ 0 with no
    # error raised. Fail fast here instead.
    seg_token = model.cfg.seg_token
    unk_id = getattr(model.processor.tokenizer, "unk_token_id", None)
    if model.seg_token_id is None or (unk_id is not None and model.seg_token_id == unk_id):
        raise RuntimeError(
            f"Segmentation token {seg_token!r} is not registered in the tokenizer "
            f"(seg_token_id={model.seg_token_id}). No <SEG> hidden state could be "
            "gathered. Check the checkpoint / VLM load."
        )
    if args.mode == "teacher_force" and seg_token not in args.teacher_answer:
        raise RuntimeError(
            f"--teacher_answer does not contain the segmentation token {seg_token!r}: "
            f"{args.teacher_answer!r}. Teacher forcing needs exactly one {seg_token}."
        )
    print(f"[INFO] mode={args.mode} seg_token={seg_token!r} id={model.seg_token_id}")

    if not Path(args.manifest).exists():
        raise FileNotFoundError(f"manifest file not found: {args.manifest}")
    items = load_manifest(args.manifest)
    items = select_items(items, n_per_task=args.n_per_task, max_samples=args.max_samples)

    print("[INFO] selected samples:", len(items))
    print("[INFO] out_root:", out_root)

    done = set()
    if args.resume and pred_jsonl.exists():
        with pred_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["sample_id"])
                except Exception:
                    pass
        print("[INFO] resume enabled, existing records:", len(done))

    t0 = time.time()
    n_ok = 0
    n_failed = 0
    n_no_seg = 0

    # Append only when resuming; otherwise start a fresh predictions.jsonl so repeated
    # runs don't accumulate duplicate records.
    write_mode = "a" if args.resume else "w"
    with pred_jsonl.open(write_mode, encoding="utf-8") as fw:
        for i, item in enumerate(tqdm(items), 1):
            sample_id = item["sample_id"]

            if sample_id in done:
                continue

            out_mask_path = mask_dir / f"{sample_id}.png"
            item['image_path'] = item['image_path'].replace("/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/images", "/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/images")
            item['gt_mask_path'] = item['gt_mask_path'].replace('/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/multiround_dataset/masks', '/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/multiround_dataset/masks')

            try:
                result = infer_one(model, item, args, device)

                if result["status"] == "ok":
                    Image.fromarray(result["mask"]).save(out_mask_path)
                    n_ok += 1
                elif result["status"] == "no_seg":
                    # Model emitted no <SEG>; save an empty mask sized from the GT
                    # (or the input image if the GT is unavailable).
                    W, H = target_size_wh(item.get("image_path"), item.get("gt_mask_path"), model.cfg)
                    empty = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
                    empty.save(out_mask_path)
                    n_no_seg += 1
                else:
                    n_failed += 1

                rec = {
                    "sample_id": sample_id,
                    "mode": args.mode,
                    "status": result["status"],
                    "n_seg": result["n_seg"],
                    "reply": result.get("reply"),
                    "pred_mask_path": str(out_mask_path),
                    "image_path": item["image_path"],
                    "gt_mask_path": item["gt_mask_path"],
                    "prompt_for_model": item.get("prompt_for_model", ""),
                    "task_type": item.get("task_type", ""),
                    "target_type": item.get("target_type", ""),
                    "target_name": item.get("target_name", ""),
                    "seg_images": result.get("seg_images", []),
                    "error": "",
                }

            except Exception as e:
                n_failed += 1
                # Surface the failure (and any missing path) live, not just in the jsonl.
                print(f"[FAILED] sample_id={sample_id}: {e}", flush=True)
                rec = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "n_seg": 0,
                    "pred_mask_path": "",
                    "image_path": item.get("image_path", ""),
                    "gt_mask_path": item.get("gt_mask_path", ""),
                    "prompt_for_model": item.get("prompt_for_model", ""),
                    "task_type": item.get("task_type", ""),
                    "target_type": item.get("target_type", ""),
                    "target_name": item.get("target_name", ""),
                    "seg_images": [],
                    "error": repr(e),
                }

            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()

            if i % args.print_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[PROGRESS] {i}/{len(items)} "
                    f"ok={n_ok} no_seg={n_no_seg} failed={n_failed} "
                    f"elapsed={elapsed:.1f}s speed={i/max(elapsed,1e-6):.3f}/s",
                    flush=True,
                )

            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "mode": args.mode,
        "n_selected": len(items),
        "ok": n_ok,
        "no_seg": n_no_seg,
        "failed": n_failed,
        "elapsed_sec": time.time() - t0,
        "mask_dir": str(mask_dir),
        "pred_jsonl": str(pred_jsonl),
    }

    with (log_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
