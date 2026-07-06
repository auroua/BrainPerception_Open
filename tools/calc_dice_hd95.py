#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import binary_erosion, distance_transform_edt

REGULAR_TASKS = {
    "basic_segmentation",
    "tissue_to_region",
}

HARD_TASKS = {
    "contralateral_same_region",
    "same_side_same_lobe",
    "spatial_named_region",
    "tumor_to_overlapping_region",
}

# The manifest stores training-machine paths (/root/autodl-tmp/...). When scoring on
# another server the files live elsewhere, so remap the paths the same way
# run_evaluation.py does before opening the GT masks. Pass --no_remap to disable.
PATH_REMAPS = [
    ("/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/images",
     "/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/images"),
    ("/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/multiround_dataset/masks",
     "/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/multiround_dataset/masks"),
]


def remap_path(p, enabled=True):
    if not p or not enabled:
        return p
    for old, new in PATH_REMAPS:
        p = p.replace(old, new)
    return p

def load_mask(path):
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0

def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)

def surface(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded

def hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    h, w = gt.shape
    diagonal = float((h ** 2 + w ** 2) ** 0.5)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        return diagonal

    pred_s = surface(pred)
    gt_s = surface(gt)

    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return diagonal

    dt_gt = distance_transform_edt(~gt_s)
    dt_pred = distance_transform_edt(~pred_s)

    d1 = dt_gt[pred_s]
    d2 = dt_pred[gt_s]

    d = np.concatenate([d1, d2])
    if d.size == 0:
        return diagonal

    return float(np.percentile(d, 95))

def summarize(df, group_col=None):
    if group_col is None:
        groups = [("overall", df)]
    else:
        groups = list(df.groupby(group_col))

    rows = []

    for name, g in groups:
        rows.append({
            "group": name,
            "n": len(g),
            "dice_mean": g["dice"].mean(),
            "dice_std": g["dice"].std(ddof=0),
            "hd95_mean_px": g["hd95_px"].mean(),
            "hd95_median_px": g["hd95_px"].median(),
            "empty_pred_rate": g["empty_pred"].mean(),
            "pred_frac_mean": g["pred_frac"].mean(),
            "gt_frac_mean": g["gt_frac"].mean(),
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="/root/autodl-tmp/brainparc_test15k_eval/manifest_explicit_finalround.jsonl")
    parser.add_argument("--pred_mask_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_remap", action="store_true",
                        help="Do not remap the manifest paths (use them as-is).")
    args = parser.parse_args()
    remap = not args.no_remap

    manifest = Path(args.manifest)
    pred_mask_dir = Path(args.pred_mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Remap training-machine paths to this server before opening the GT mask.
            item["gt_mask_path"] = remap_path(item.get("gt_mask_path"), remap)
            if "image_path" in item:
                item["image_path"] = remap_path(item.get("image_path"), remap)
            sid = item["sample_id"]
            pred_path = pred_mask_dir / f"{sid}.png"

            if not pred_path.exists():
                missing.append(sid)
                continue

            pred = load_mask(pred_path)
            gt = load_mask(item["gt_mask_path"])

            if pred.shape != gt.shape:
                pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
                pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST)
                pred = np.array(pred_img) > 0

            task = item.get("task_type", "unknown")
            if task in REGULAR_TASKS:
                difficulty = "regular"
            elif task in HARD_TASKS:
                difficulty = "hard"
            else:
                difficulty = "unknown"

            rows.append({
                "sample_id": sid,
                "task_type": task,
                "difficulty": difficulty,
                "target_type": item.get("target_type", ""),
                "target_name": item.get("target_name", ""),
                "dice": dice(pred, gt),
                "hd95_px": hd95(pred, gt),
                "empty_pred": int(pred.sum() == 0),
                "pred_pixels": int(pred.sum()),
                "gt_pixels": int(gt.sum()),
                "pred_frac": float(pred.sum() / pred.size),
                "gt_frac": float(gt.sum() / gt.size),
                "pred_path": str(pred_path),
                "gt_mask_path": item["gt_mask_path"],
                "prompt": item.get("prompt_for_model", ""),
            })

    df = pd.DataFrame(rows)

    df.to_csv(out_dir / "detail_dice_hd95.csv", index=False, encoding="utf-8-sig")
    summarize(df).to_csv(out_dir / "summary_overall.csv", index=False, encoding="utf-8-sig")
    summarize(df, "task_type").to_csv(out_dir / "summary_by_task.csv", index=False, encoding="utf-8-sig")
    summarize(df, "difficulty").to_csv(out_dir / "summary_by_difficulty.csv", index=False, encoding="utf-8-sig")
    summarize(df, "target_type").to_csv(out_dir / "summary_by_target_type.csv", index=False, encoding="utf-8-sig")

    if missing:
        (out_dir / "missing_predictions.txt").write_text("\n".join(missing), encoding="utf-8")

    print("Evaluation done.")
    print("Evaluated:", len(df))
    print("Missing:", len(missing))
    print("Output:", out_dir)
    print()
    print("Overall:")
    print(summarize(df).to_string(index=False))
    print()
    print("By task:")
    print(summarize(df, "task_type").to_string(index=False))
    print()
    print("By difficulty:")
    print(summarize(df, "difficulty").to_string(index=False))

if __name__ == "__main__":
    main()
