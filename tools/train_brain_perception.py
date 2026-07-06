#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_brain_perception.py

Multi-GPU (🤗 accelerate) training loop for BrainPerceptionModel
(Qwen3-VL + <SEG> + SAM 2/3):
  - case-level train/val split (no slice leakage)
  - gradient accumulation (manual, with DDP grad-sync skipped between micro-steps)
  - mixed precision via accelerate (bf16 default / fp16 with managed GradScaler / fp32)
  - cosine LR schedule with warmup
  - periodic validation (loss + Dice score), reduced across ranks
  - checkpointing of TRAINABLE params only (LoRA + projection + SAM decoder) + resume

Single-GPU (unchanged UX) — runs as one accelerate process:
    PYTHONPATH=$PWD python tools/train_brain_perception.py \
        --out runs/exp1 --epochs 3 --batch_size 1 --grad_accum 8 --precision bf16

Multi-GPU on one node (e.g. 8 GPUs) — launch with accelerate or torchrun:
    PYTHONPATH=$PWD accelerate launch --num_processes 8 \
        tools/train_brain_perception.py --out runs/exp1 --batch_size 1 --grad_accum 8
    # or, equivalently:
    PYTHONPATH=$PWD torchrun --nproc_per_node 8 \
        tools/train_brain_perception.py --out runs/exp1 --batch_size 1 --grad_accum 8

Notes for multi-GPU:
  - The global (effective) batch = batch_size * grad_accum * num_processes.
  - --batch_size and --grad_accum are PER-PROCESS values.
  - Prefer --precision bf16. With fp16 the base weights are loaded in fp32 and
    accelerate keeps an fp32 master copy + GradScaler (correct mixed precision);
    that costs more memory than bf16.

    # explicit dataset override:
    PYTHONPATH=$PWD accelerate launch --num_processes 8 \
        tools/train_brain_perception.py \
        --root /mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D \
        --dialogues_rel multiround_dataset/multiround_dialogues.jsonl --out runs/exp1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

import sys
# Make the repo root importable regardless of the machine / working directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.brain_perception_dataset import (
    BrainPerceptionDataConfig,
    BrainPerceptionDataset,
    list_all_case_ids,
    split_cases_by_ratio,
)
from src.models.brain_perception_model import (
    BrainPerceptionModel,
    BrainPerceptionModelConfig,
    Qwen3VLSegCollator,
)

# Map the requested run precision to (weight load dtype, accelerate mixed_precision).
# fp16: load weights in fp32 so accelerate can keep an fp32 master copy + GradScaler
# (loading weights directly in fp16 has no master copy and trains poorly). bf16 has no
# scaler/master-copy requirement, so we load weights in bf16 to save memory.
PRECISION_TO_LOAD_DTYPE = {"bf16": "bfloat16", "fp16": "float32", "fp32": "float32"}
PRECISION_TO_MIXED = {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}


def format_duration(seconds: float) -> str:
    """Compact wall-clock duration string for logs."""
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m{seconds:02d}s"
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _default_dataset_root() -> Optional[str]:
    """Default --root to the generated dataset root from data_pathes, if importable."""
    try:
        from src.dataset.data_pathes import instance_out_dir
        return instance_out_dir
    except Exception:
        return None


def resolve_dialogues_rel(root: str, explicit: Optional[str]) -> str:
    """
    Pick the dialogues jsonl (relative to root) to train on.

    Matches 4_remove_duplicate.py / 5_*: prefer the deduped
    multiround_dialogues.dedup.jsonl when it exists, otherwise fall back to the
    raw multiround_dialogues.jsonl. An explicit path always wins.
    """
    if explicit:
        return explicit
    base = Path(root) / "multiround_dataset"
    dedup_rel = "multiround_dataset/multiround_dialogues.dedup.jsonl"
    raw_rel = "multiround_dataset/multiround_dialogues.jsonl"
    return dedup_rel if (base / "multiround_dialogues.dedup.jsonl").exists() else raw_rel


def build_accelerator(args):
    """Create the Accelerator. find_unused_parameters=True is required because the
    forward has an `n_seg == 0` early-return that leaves the SAM decoder / projection /
    inflated patch-embed without gradients; if one rank takes that path and another
    does not, DDP's all-reduce would otherwise deadlock."""
    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "This trainer now uses 🤗 accelerate for (multi-)GPU training. "
            "Install it with `pip install accelerate`, then launch with "
            "`accelerate launch` or `torchrun`. Original error: " + str(e)
        )
    if args.wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment guard
            raise ImportError(
                "--wandb was set but Weights & Biases is not installed. "
                "Run `pip install wandb` (and `wandb login`), or drop --wandb."
            ) from e
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        mixed_precision=PRECISION_TO_MIXED[args.precision],
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb" if args.wandb else None,
    )


# ============================================================
# Data
# ============================================================

def build_loaders(model, args, accelerator):
    dialogues_rel = resolve_dialogues_rel(args.root, args.dialogues_rel)
    accelerator.print(f"[data] root={args.root}")
    accelerator.print(f"[data] dialogues={dialogues_rel}")
    base = BrainPerceptionDataConfig(
        dataset_root=args.root,
        dialogues_rel=dialogues_rel,
        build_clip_branch=False,           # Qwen processor handles the image
        seg_image_size=args.seg_size,
        mask_size=args.mask_size,
        max_rounds=2,
        # Feed SAM exactly the number of channels the model expects (default 3 -> native
        # SAM patch embed, no encoder inflation). Locking these two together prevents a
        # data/model channel mismatch (SAM's 3-ch conv choking on a 4-ch input).
        seg_select_k=model.cfg.seg_in_channels,
    )
    train_cases, val_cases = split_cases_by_ratio(
        list_all_case_ids(base), val_ratio=args.val_ratio, seed=args.seed
    )
    accelerator.print(f"[data] train cases={len(train_cases)}  val cases={len(val_cases)}")
    accelerator.print(f"[data] seg channels={base.seg_select_k} (random modality dropout for train)")

    # Train: random modality subset each step (augmentation). Val: deterministic first-k
    # so validation metrics aren't noised by which modalities happened to be sampled.
    train_cfg = replace(base, keep_case_ids=tuple(train_cases))
    val_cfg = (replace(base, keep_case_ids=tuple(val_cases), seg_random_modalities=False)
               if val_cases else None)

    collator = Qwen3VLSegCollator(processor=model.processor)
    train_ds = BrainPerceptionDataset(train_cfg, tokenizer=None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_cfg is not None:
        val_ds = BrainPerceptionDataset(val_cfg, tokenizer=None)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collator, pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader


# ============================================================
# Checkpoint (trainable params only)
# ============================================================

def trainable_names(model) -> set:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(path: Path, raw_model, optimizer, scheduler, step: int, best: float, args, epoch: int = 0):
    """`raw_model` must be the unwrapped module (accelerator.unwrap_model)."""
    names = trainable_names(raw_model)
    sd = {k: v for k, v in raw_model.state_dict().items() if k in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": sd,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_dice": best,
            "args": vars(args),
        },
        path,
    )
    print(f"[ckpt] saved {path}  (step {step}, {len(sd)} tensors)")


def load_ckpt(path: Path, raw_model, optimizer=None, scheduler=None) -> Dict:
    """`raw_model` must be the unwrapped module (accelerator.unwrap_model)."""
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = raw_model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    print(f"[ckpt] loaded {path}  (missing={len(missing)} unexpected={len(unexpected)})")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, accelerator, max_batches=None) -> Dict[str, float]:
    """Validation loss + soft Dice, summed locally then reduced across ranks so every
    process returns the same global metric. Mixed precision is applied automatically by
    accelerate's wrapped forward, so no explicit autocast is needed here."""
    if val_loader is None:
        return {}
    device = accelerator.device
    model.eval()
    # [sum_loss, sum_lm_loss, sum_dice_loss, count]
    agg = torch.zeros(4, device=device)
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(batch)
        agg[0] += out["loss"].detach()
        agg[1] += out.get("lm_loss", torch.zeros((), device=device))
        agg[2] += out.get("dice_loss", torch.zeros((), device=device))
        agg[3] += 1
    model.train()
    agg = accelerator.reduce(agg, reduction="sum")
    n = max(float(agg[3].item()), 1.0)
    return {
        "val_loss": float(agg[0].item()) / n,
        "val_lm_loss": float(agg[1].item()) / n,
        "val_dice_score": 1.0 - float(agg[2].item()) / n,   # 1 - dice_loss = soft Dice
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    _default_root = _default_dataset_root()
    ap.add_argument(
        "--root",
        default=_default_root,
        required=_default_root is None,
        help="dataset root (default: data_pathes.instance_out_dir, the generated dataset)",
    )
    ap.add_argument(
        "--dialogues_rel",
        default="",
        help="dialogues jsonl relative to --root (default: auto-pick "
             "multiround_dialogues.dedup.jsonl if present, else multiround_dialogues.jsonl)",
    )
    ap.add_argument("--out", default="runs/exp1", help="output dir for checkpoints/logs")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--vlm_source", choices=["huggingface", "modelscope"], default="huggingface",
                    help="where to resolve --vlm from. Use 'modelscope' for ModelScope model ids.")
    ap.add_argument("--vlm_revision", default=None,
                    help="optional VLM revision passed to ModelScope snapshot_download")
    ap.add_argument("--vlm_cache_dir", default=None,
                    help="optional cache directory passed to ModelScope snapshot_download")
    ap.add_argument(
        "--sam_version",
        choices=["sam2", "sam3"],
        default="sam3",
        help="which SAM family to use as the segmentation head (default: sam3)",
    )
    ap.add_argument(
        "--sam",
        default="facebook/sam3",
        help="SAM checkpoint id/dir. Leave at the sam3 default and pass "
             "--sam_version sam2 to auto-use the SAM 2 default checkpoint.",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps cap (0 = use epochs)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seg_size", type=int, default=1024)
    ap.add_argument("--mask_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_max_batches", type=int, default=100,
                    help="validate on at most this many batches per process (0 = full val set). "
                         "The val set here is huge (~300k samples), so a full pass takes hours; "
                         "a capped subset gives a fast, stable Dice estimate.")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", default="", help="path to a checkpoint to resume from")
    # ---- mask loss class-imbalance handling (small lesion vs big background) ----
    ap.add_argument("--mask_bce_mode", choices=["weighted", "focal", "plain"], default="weighted",
                    help="weighted: BCE with foreground pos_weight (default, auto-balanced); "
                         "focal: focal loss; plain: original unweighted BCE")
    ap.add_argument("--bce_pos_weight", type=float, default=None,
                    help="weighted mode: fixed foreground pos_weight; omit for auto (#bg/#fg per batch)")
    # ---- Weights & Biases logging ----
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb_project", default="brain-perception", help="W&B project name")
    ap.add_argument("--wandb_run_name", default=None, help="W&B run name (default: W&B auto-name)")
    ap.add_argument("--wandb_entity", default=None, help="W&B entity/team (default: your default)")
    args = ap.parse_args()

    # ---- accelerate (single- or multi-GPU) ----
    accelerator = build_accelerator(args)
    # Seed every process identically. Accelerate shards a shuffled loader by having all
    # ranks share one sampler permutation and slice it disjointly; differing per-rank RNG
    # can desync that (silent data overlap/gaps) on accelerate builds where the seedable
    # sampler is off, so keep the seed rank-independent.
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    accelerator.print(f"[accel] processes={accelerator.num_processes}  "
                      f"device={accelerator.device}  mixed_precision={accelerator.mixed_precision}")

    # ---- model ----
    # Leave the model on CPU here; accelerator.prepare() moves it to this rank's device.
    model_cfg = BrainPerceptionModelConfig(
        vlm_name_or_path=args.vlm,
        vlm_source=args.vlm_source,
        vlm_revision=args.vlm_revision,
        vlm_cache_dir=args.vlm_cache_dir,
        sam_version=args.sam_version,
        sam_name_or_path=args.sam,
        seg_image_size=args.seg_size,
        mask_size=args.mask_size,
        torch_dtype=PRECISION_TO_LOAD_DTYPE[args.precision],
        mask_bce_mode=args.mask_bce_mode,
        bce_pos_weight=args.bce_pos_weight,
    )
    accelerator.print(f"[model] seg head: {args.sam_version} ({args.sam})  "
                      f"mask_bce_mode={args.mask_bce_mode}")
    accelerator.print(f"[model] VLM source: {args.vlm_source} ({args.vlm})")
    # In distributed launches, let rank 0 resolve/download a ModelScope snapshot first;
    # other ranks then load from the populated cache/local path instead of racing.
    with accelerator.main_process_first():
        model = BrainPerceptionModel(model_cfg)
    accelerator.print(model.trainable_parameter_report())

    # ---- data (loaders are prepared below; build with the raw model's processor) ----
    train_loader, val_loader = build_loaders(model, args, accelerator)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- prepare with accelerate ----
    # device_placement=False for the loaders: the model.forward() already moves every
    # batch tensor to its parameters' device, and it compares num_masks_per_sample on CPU
    # (torch.equal with a .cpu() tensor) — letting accelerate move the batch to GPU would
    # break that guard. Sharding across ranks still happens regardless of device_placement.
    prep = [model, optimizer, train_loader]
    place = [True, True, False]
    if val_loader is not None:
        prep.append(val_loader)
        place.append(False)
    prepared = accelerator.prepare(*prep, device_placement=place)
    model, optimizer, train_loader = prepared[0], prepared[1], prepared[2]
    if val_loader is not None:
        val_loader = prepared[3]
    # Trainable params on the (now device-placed) model, used for grad clipping.
    params = [p for p in model.parameters() if p.requires_grad]

    # ---- schedule (built AFTER prepare so len(train_loader) is the per-rank sharded
    # length; stepped manually once per optimizer update, so it is NOT prepared) ----
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    accelerator.print(f"[sched] total_steps={total_steps}  warmup={warmup_steps}  "
                      f"steps/epoch={steps_per_epoch}  "
                      f"global_batch={args.batch_size * args.grad_accum * accelerator.num_processes}")

    # ---- W&B tracker (init on all ranks; accelerate only inits on the main process) ----
    if args.wandb:
        tracker_config = {
            **vars(args),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_processes": accelerator.num_processes,
            "global_batch": args.batch_size * args.grad_accum * accelerator.num_processes,
        }
        init_kwargs = {"wandb": {}}
        if args.wandb_run_name:
            init_kwargs["wandb"]["name"] = args.wandb_run_name
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        accelerator.init_trackers(args.wandb_project, config=tracker_config, init_kwargs=init_kwargs)
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except Exception:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, s / warmup_steps) * (0.5 * (1 + math.cos(math.pi * s / total_steps))),
        )

    step = 0
    best_val = -1.0
    start_epoch = 0
    if args.resume:
        ck = load_ckpt(Path(args.resume), accelerator.unwrap_model(model), optimizer, scheduler)
        step = ck.get("step", 0)
        best_val = ck.get("best_val_dice", -1.0)
        # Resume from the saved epoch so already-completed epochs are not replayed.
        # (Within-epoch loader position is not restored, so the resumed epoch restarts
        # from its first batch — an accepted approximation.)
        start_epoch = ck.get("epoch", 0)

    def _save(name: str):
        """Barrier + main-process-only checkpoint of the unwrapped model."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_ckpt(out_dir / name, accelerator.unwrap_model(model),
                      optimizer, scheduler, step, best_val, args, epoch=epoch)

    # ---- train loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    t0 = time.time()
    train_t0 = t0
    timer_start_step = step
    done = False
    epoch = start_epoch  # keep `epoch` defined for the final save if the loop never runs
    for epoch in range(start_epoch, args.epochs):
        if done:
            break
        # Reshuffle each epoch (prepared DataLoaderShard won't otherwise re-seed).
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        for batch in train_loader:
            micro += 1
            is_boundary = (micro % args.grad_accum == 0)
            # Skip DDP gradient all-reduce on non-boundary micro-steps (it only needs to
            # happen on the step where we actually update). no_sync is a no-op on 1 GPU.
            sync_ctx = nullcontext() if is_boundary else accelerator.no_sync(model)
            with sync_ctx:
                out = model(batch)
                # Divide by grad_accum ourselves (deterministic across accelerate versions,
                # since we do NOT use accelerator.accumulate's implicit scaling).
                loss = out["loss"] / args.grad_accum
                accelerator.backward(loss)
            if not is_boundary:
                continue

            # ---- optimizer step (boundary only) ----
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            # With fp16 accelerate manages a GradScaler; if it skipped the step (inf/nan
            # grad) don't advance the LR schedule. Always False for bf16/fp32.
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                now = time.time()
                elapsed_s = now - train_t0
                completed_run_steps = max(step - timer_start_step, 1)
                seconds_per_step = elapsed_s / completed_run_steps
                remaining_steps = max(total_steps - step, 0)
                eta_s = seconds_per_step * remaining_steps
                estimated_total_s = elapsed_s + eta_s
                window_s = max(now - t0, 1e-9)
                rec = {
                    "step": step, "epoch": epoch,
                    "loss": float(out["loss"]), "lm_loss": float(out.get("lm_loss", 0.0)),
                    "bce_loss": float(out.get("bce_loss", 0.0)), "dice_loss": float(out.get("dice_loss", 0.0)),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": elapsed_s,
                    "eta_s": eta_s,
                    "estimated_total_s": estimated_total_s,
                    "seconds_per_step": seconds_per_step,
                    # global throughput across all ranks
                    "imgs_per_s": (args.grad_accum * args.batch_size * args.log_every
                                   * accelerator.num_processes / window_s),
                }
                t0 = now
                accelerator.print(f"step {step:5d}/{total_steps} | loss {rec['loss']:.4f} "
                                  f"| lm {rec['lm_loss']:.4f} | dice_loss {rec['dice_loss']:.4f} "
                                  f"| lr {rec['lr']:.2e} "
                                  f"| elapsed {format_duration(elapsed_s)} "
                                  f"| eta {format_duration(eta_s)} "
                                  f"| est_total {format_duration(estimated_total_s)}")
                if accelerator.is_main_process:
                    with log_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                if args.wandb:
                    accelerator.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)

            if step % args.eval_every == 0:
                metrics = evaluate(model, val_loader, accelerator,
                                   max_batches=(args.eval_max_batches or None))
                if metrics:
                    accelerator.print(f"[eval] step {step} | "
                                      + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    if accelerator.is_main_process:
                        with log_path.open("a") as f:
                            f.write(json.dumps({"step": step, **metrics}) + "\n")
                    if args.wandb:
                        accelerator.log({f"val/{k.replace('val_', '')}": v
                                         for k, v in metrics.items()}, step=step)
                    if metrics["val_dice_score"] > best_val:
                        best_val = metrics["val_dice_score"]
                        _save("best.pt")

            if step % args.save_every == 0:
                _save("last.pt")

            if step >= total_steps:
                done = True
                break

    _save("last.pt")
    if args.wandb:
        accelerator.end_training()   # flushes + finishes the W&B run
    accelerator.print("DONE.")


if __name__ == "__main__":
    main()
