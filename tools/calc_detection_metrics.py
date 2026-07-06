#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert predicted / ground-truth segmentation masks to bounding boxes and score
box grounding metrics.

This script mirrors tools/calc_dice_hd95.py:
  - reads the same evaluation manifest
  - applies the same path remapping
  - expects predicted masks at --pred_mask_dir/{sample_id}.png

For each sample, the predicted and GT masks are converted to tight xyxy boxes.
A sample is recognized when the predicted box overlaps the GT box (IoU > 0).
IoU mean and Acc@tau are computed over recognized samples only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image


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

# Keep this identical to calc_dice_hd95.py so both scorers can be run on the
# same manifest on a different machine.
PATH_REMAPS = [
    ("/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/images",
     "/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/images"),
    ("/root/autodl-tmp/Processed_T1N_T1C_T2W_T2F_BrainParc_2D_step3_teacher6_full/multiround_dataset/masks",
     "/mnt/rna01/chenw/Datasets/BraTS2024/BrainPerception_2D/multiround_dataset/masks"),
]


Box = Tuple[int, int, int, int]  # xyxy, with x2/y2 exclusive


def remap_path(p: Optional[str], enabled: bool = True) -> Optional[str]:
    if not p or not enabled:
        return p
    for old, new in PATH_REMAPS:
        p = p.replace(old, new)
    return p


def load_mask(path: Path, threshold: int = 0) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > threshold


def resize_mask_to(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.array(img) > 0


def mask_to_box(mask: np.ndarray) -> Optional[Box]:
    """Return a tight xyxy box with x2/y2 exclusive, or None for an empty mask."""
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_strict_valid_box(box: Optional[Box], width: int, height: int) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def box_iou(a: Optional[Box], b: Optional[Box]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def box_to_json(box: Optional[Box]) -> str:
    if box is None:
        return ""
    return json.dumps(list(box), ensure_ascii=False)


def difficulty_for_task(task: str) -> str:
    if task in REGULAR_TASKS:
        return "regular"
    if task in HARD_TASKS:
        return "hard"
    return "unknown"


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def summarize(df: pd.DataFrame, group_col: Optional[str] = None, method: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if group_col is None:
        groups: Iterable[Tuple[str, pd.DataFrame]] = [("overall", df)]
    else:
        groups = df.groupby(group_col, dropna=False)

    rows: List[Dict[str, object]] = []
    for name, g in groups:
        n = int(len(g))
        recognized = g[g["recognized"] == 1]
        n_recognized = int(len(recognized))

        if n_recognized > 0:
            iou_mean = float(recognized["bbox_iou"].mean())
            acc_01 = float((recognized["bbox_iou"] >= 0.1).mean())
            acc_03 = float((recognized["bbox_iou"] >= 0.3).mean())
            acc_05 = float((recognized["bbox_iou"] >= 0.5).mean())
        else:
            iou_mean = float("nan")
            acc_01 = float("nan")
            acc_03 = float("nan")
            acc_05 = float("nan")

        rows.append({
            "Method": method,
            "group": name,
            "N": n,
            "No recog.": 1.0 - (n_recognized / n if n else 0.0),
            "Recognized": n_recognized / n if n else float("nan"),
            "Parse succ.": rate(g["parse_success"]),
            "Strict valid bbox": rate(g["strict_valid_bbox"]),
            "OOB": rate(g["oob"]),
            "IoU mean": iou_mean,
            "Acc@0.1": acc_01,
            "Acc@0.3": acc_03,
            "Acc@0.5": acc_05,
            "missing_pred_rate": rate(g["missing_pred"]),
            "empty_pred_rate": rate(g["empty_pred"]),
            "valid_gt_bbox_rate": rate(g["valid_gt_bbox"]),
        })

    return pd.DataFrame(rows)


def format_latex_row(row: pd.Series) -> str:
    method = str(row.get("Method") or "")
    n = int(row["N"])
    fields = [
        method,
        f"{n:,}",
        f"{row['No recog.']:.4f}",
        f"{row['Recognized']:.4f}",
        f"{row['Parse succ.']:.4f}",
        f"{row['Strict valid bbox']:.4f}",
        f"{row['OOB']:.4f}",
        f"{row['IoU mean']:.4f}" if pd.notna(row["IoU mean"]) else "nan",
        f"{row['Acc@0.1']:.4f}" if pd.notna(row["Acc@0.1"]) else "nan",
        f"{row['Acc@0.3']:.4f}" if pd.notna(row["Acc@0.3"]) else "nan",
        f"{row['Acc@0.5']:.4f}" if pd.notna(row["Acc@0.5"]) else "nan",
    ]
    return " & ".join(fields) + r" \\"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str,
                        default="/root/autodl-tmp/brainparc_test15k_eval/manifest_explicit_finalround.jsonl")
    parser.add_argument("--pred_mask_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="Mask-to-Box",
                        help="Method name written to the table-style summary.")
    parser.add_argument("--mask_threshold", type=int, default=0,
                        help="Pixels greater than this value are foreground.")
    parser.add_argument("--no_remap", action="store_true",
                        help="Do not remap the manifest paths (use them as-is).")
    args = parser.parse_args()

    remap = not args.no_remap
    manifest = Path(args.manifest)
    pred_mask_dir = Path(args.pred_mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            sid = item["sample_id"]
            task = item.get("task_type", "unknown")
            gt_mask_path = remap_path(item.get("gt_mask_path"), remap)
            image_path = remap_path(item.get("image_path"), remap)
            pred_path = pred_mask_dir / f"{sid}.png"

            error = ""
            missing_pred = not pred_path.exists()
            if missing_pred:
                missing.append(sid)

            gt = None
            pred = None
            gt_box: Optional[Box] = None
            pred_box: Optional[Box] = None
            height = width = 0

            try:
                if not gt_mask_path:
                    raise FileNotFoundError("missing gt_mask_path in manifest")
                gt = load_mask(Path(gt_mask_path), threshold=args.mask_threshold)
                height, width = gt.shape
                gt_box = mask_to_box(gt)

                if not missing_pred:
                    pred = load_mask(pred_path, threshold=args.mask_threshold)
                    pred = resize_mask_to(pred, gt.shape)
                    pred_box = mask_to_box(pred)
            except Exception as e:
                error = repr(e)

            valid_gt = is_strict_valid_box(gt_box, width, height)
            parse_success = pred_box is not None
            strict_valid = is_strict_valid_box(pred_box, width, height)
            # For mask-derived boxes, OOB should only occur if future callers replace the
            # mask_to_box path with externally parsed boxes.
            oob = bool(parse_success and not strict_valid)
            iou = box_iou(pred_box, gt_box) if valid_gt and strict_valid else 0.0
            recognized = bool(iou > 0.0)

            rows.append({
                "sample_id": sid,
                "task_type": task,
                "difficulty": difficulty_for_task(task),
                "target_type": item.get("target_type", ""),
                "target_name": item.get("target_name", ""),
                "missing_pred": int(missing_pred),
                "parse_success": int(parse_success),
                "strict_valid_bbox": int(strict_valid),
                "oob": int(oob),
                "valid_gt_bbox": int(valid_gt),
                "recognized": int(recognized),
                "bbox_iou": iou,
                "acc_0_1": int(recognized and iou >= 0.1),
                "acc_0_3": int(recognized and iou >= 0.3),
                "acc_0_5": int(recognized and iou >= 0.5),
                "empty_pred": int((pred is None) or (pred.sum() == 0)),
                "pred_pixels": int(pred.sum()) if pred is not None else 0,
                "gt_pixels": int(gt.sum()) if gt is not None else 0,
                "pred_bbox_xyxy": box_to_json(pred_box),
                "gt_bbox_xyxy": box_to_json(gt_box),
                "image_width": width,
                "image_height": height,
                "pred_path": str(pred_path),
                "gt_mask_path": gt_mask_path or "",
                "image_path": image_path or "",
                "prompt": item.get("prompt_for_model", ""),
                "error": error,
            })

    df = pd.DataFrame(rows)
    detail_path = out_dir / "detail_detection_metrics.csv"
    overall_path = out_dir / "summary_detection_overall.csv"
    by_task_path = out_dir / "summary_detection_by_task.csv"
    by_difficulty_path = out_dir / "summary_detection_by_difficulty.csv"
    by_target_type_path = out_dir / "summary_detection_by_target_type.csv"
    table_path = out_dir / "table_detection_metrics.csv"
    latex_row_path = out_dir / "table_detection_latex_row.txt"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    overall = summarize(df, method=args.method)
    by_task = summarize(df, "task_type", method=args.method)
    by_difficulty = summarize(df, "difficulty", method=args.method)
    by_target_type = summarize(df, "target_type", method=args.method)

    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    by_task.to_csv(by_task_path, index=False, encoding="utf-8-sig")
    by_difficulty.to_csv(by_difficulty_path, index=False, encoding="utf-8-sig")
    by_target_type.to_csv(by_target_type_path, index=False, encoding="utf-8-sig")

    table_cols = [
        "Method", "N", "No recog.", "Recognized", "Parse succ.",
        "Strict valid bbox", "OOB", "IoU mean", "Acc@0.1", "Acc@0.3", "Acc@0.5",
    ]
    table = overall[table_cols] if not overall.empty else pd.DataFrame(columns=table_cols)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    if not overall.empty:
        latex_row_path.write_text(format_latex_row(overall.iloc[0]) + "\n", encoding="utf-8")

    if missing:
        (out_dir / "missing_predictions.txt").write_text("\n".join(missing), encoding="utf-8")

    print("Detection evaluation done.")
    print("Samples:", len(df))
    print("Missing predictions:", len(missing))
    print("Output:", out_dir)
    print()
    print("Overall table metrics:")
    print(table.to_string(index=False))
    print()
    print("By task:")
    print(by_task[table_cols + ["group"]].to_string(index=False) if not by_task.empty else "empty")
    print()
    print("LaTeX row:")
    if not overall.empty:
        print(format_latex_row(overall.iloc[0]))


if __name__ == "__main__":
    main()
