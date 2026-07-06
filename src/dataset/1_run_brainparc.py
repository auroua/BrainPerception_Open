from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd


# ============================================================
# 0. 参数配置区：只改这里即可
# ============================================================

# BraTS 原始数据根目录：要求能递归找到 *-t1n.nii.gz
INPUT_ROOT = Path("/root/autodl-tmp/TrainingData")

# BrainParc 全量输出目录：每个病例会输出到 OUT_ROOT / case_id
OUT_ROOT = Path("/root/autodl-tmp/brainparc_full")

# AutoStrip / BrainParc 代码与权重目录
AUTOSTRIP_DIR = Path("/root/autodl-tmp/AutoStrip")
BRAINPARC_DIR = Path("/root/autodl-tmp/BrainParc")

AUTOSTRIP_MODEL = AUTOSTRIP_DIR / "Pretrained_Model" / "AutoBET_Fine.pth.gz"
BRAINPARC_MODEL = BRAINPARC_DIR / "Pretrained_Model" / "BrainParc.pth.gz"
LABEL_XLSX = BRAINPARC_DIR / "Pretrained_Model" / "Label_Cor.xlsx"

# 运行控制
MAX_CASES: Optional[int] = None        # 跑全量设为 None；调试可设 1 / 3 / 10
CASE_IDS: list[str] = [
    # "BraTS-GLI-00005-100",
]
START_INDEX = 0
END_INDEX: Optional[int] = None
SKIP_EXISTING = True

# 是否保留输入 T1 副本。全量建议 False，节省空间。
KEEP_T1_COPY = False

# 是否清理中间文件。全量建议 True。
CLEAN_INTERMEDIATE = True

# 关键：是否把 BrainParc 输出强制重采样回原始 T1N 网格。
# 建议 True，这样下一步切片脚本能稳定对齐 t1n / tissue / dk-struct。
ALIGN_OUTPUT_TO_T1_GRID = True

# 如果重采样后 shape/affine 仍不一致，是否直接报错。
STRICT_FINAL_QC = True

# AutoStrip / BrainParc 参数。保持你之前跑通版本的设置。
AUTOSTRIP_NORM_ORIENTATION = 0
AUTOSTRIP_NORM_SPACING = 0
BRAINPARC_NORM_ORIENTATION = 1
BRAINPARC_NORM_SPACING = 1

# 调外部脚本使用的 python。通常就是 python。
PYTHON_BIN = "python"


# ============================================================
# 1. 基础工具函数
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    """流式运行外部命令，把 stdout/stderr 实时写入 run.log。"""
    append_log(
        log_path,
        "\n" + "=" * 100 + "\n"
        + f"TIME: {now_str()}\n"
        + "CMD: " + " ".join(cmd) + "\n"
        + "CWD: " + str(cwd) + "\n"
        + "=" * 100 + "\n",
    )

    with log_path.open("a", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}:\n"
            f"{' '.join(cmd)}\n"
            f"See log: {log_path}"
        )


def case_id_from_t1n(path: Path) -> str:
    suffix = "-t1n.nii.gz"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected T1N filename: {path}")
    return name[: -len(suffix)]


def find_t1n_files(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.rglob("*-t1n.nii.gz") if p.is_file()])


def affine_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    return bool(np.allclose(a, b, atol=atol))


def load_label_table(label_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(label_xlsx)
    if "BrainParc_index" not in df.columns:
        raise ValueError(f"'BrainParc_index' not found in {label_xlsx}")
    df["BrainParc_index"] = pd.to_numeric(df["BrainParc_index"], errors="coerce")
    df = df.dropna(subset=["BrainParc_index"]).copy()
    df["BrainParc_index"] = df["BrainParc_index"].astype(int)
    return df


# ============================================================
# 2. 输出对齐与标签表
# ============================================================

def resample_label_to_ref_grid(src_path: Path, ref_path: Path, out_path: Path, log_path: Path) -> None:
    """
    将标签图像用最近邻重采样到 ref_path 的空间。
    用于保证 tissue.nii.gz / dk-struct.nii.gz 和原始 t1n 在同一 shape/affine 上。
    """
    src_img = nib.load(str(src_path))
    ref_img = nib.load(str(ref_path))

    append_log(
        log_path,
        "\n[ALIGN]\n"
        f"src={src_path}\n"
        f"ref={ref_path}\n"
        f"src_shape={src_img.shape}, ref_shape={ref_img.shape}\n"
        f"src_affine=\n{src_img.affine}\n"
        f"ref_affine=\n{ref_img.affine}\n",
    )

    same_shape = tuple(src_img.shape[:3]) == tuple(ref_img.shape[:3])
    same_affine = affine_close(src_img.affine, ref_img.affine)

    if same_shape and same_affine:
        data = np.asarray(src_img.dataobj)
        data = np.rint(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.uint16)
        out_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
        out_img.set_data_dtype(np.uint16)
        nib.save(out_img, str(out_path))
        append_log(log_path, f"[ALIGN] already on reference grid, saved: {out_path}\n")
        return

    # order=0: nearest neighbor, 必须用于 label mask
    resampled = resample_from_to(src_img, ref_img, order=0)
    data = np.asarray(resampled.dataobj)
    data = np.rint(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.uint16)

    out_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
    out_img.set_data_dtype(np.uint16)
    nib.save(out_img, str(out_path))

    append_log(
        log_path,
        f"[ALIGN] resampled label to T1 grid and saved: {out_path}\n"
        f"out_shape={out_img.shape}\n"
        f"out_affine=\n{out_img.affine}\n",
    )


def final_qc_against_t1(t1_path: Path, tissue_path: Path, dk_path: Path, log_path: Path) -> dict:
    t1_img = nib.load(str(t1_path))
    tissue_img = nib.load(str(tissue_path))
    dk_img = nib.load(str(dk_path))

    qc = {
        "t1_shape": tuple(map(int, t1_img.shape[:3])),
        "tissue_shape": tuple(map(int, tissue_img.shape[:3])),
        "dk_shape": tuple(map(int, dk_img.shape[:3])),
        "tissue_shape_match": tuple(tissue_img.shape[:3]) == tuple(t1_img.shape[:3]),
        "dk_shape_match": tuple(dk_img.shape[:3]) == tuple(t1_img.shape[:3]),
        "tissue_affine_match": affine_close(tissue_img.affine, t1_img.affine),
        "dk_affine_match": affine_close(dk_img.affine, t1_img.affine),
    }

    append_log(log_path, "\n[FINAL_QC]\n" + str(qc) + "\n")

    if STRICT_FINAL_QC:
        if not (qc["tissue_shape_match"] and qc["dk_shape_match"]):
            raise RuntimeError(f"Final QC failed: shape mismatch. qc={qc}")
        if not (qc["tissue_affine_match"] and qc["dk_affine_match"]):
            raise RuntimeError(f"Final QC failed: affine mismatch. qc={qc}")

    return qc


def build_present_labels_csv(dk_path: Path, label_df: pd.DataFrame, out_csv: Path) -> None:
    """根据 dk-struct.nii.gz 中实际出现的 label，生成 present_labels.csv。"""
    dk_img = nib.load(str(dk_path))
    dk = np.asarray(dk_img.dataobj).astype(np.int32)

    labels = np.unique(dk).astype(np.int64)
    labels = labels[labels > 0]

    rows = []
    for lb in labels.tolist():
        rows.append({"BrainParc_index": int(lb), "present_voxels": int(np.sum(dk == lb))})

    present = pd.DataFrame(rows)
    if len(present) == 0:
        present = pd.DataFrame(columns=["BrainParc_index", "present_voxels"])

    present = present.sort_values("BrainParc_index")
    merged = present.merge(label_df, on="BrainParc_index", how="left")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 3. 断点续跑与清理
# ============================================================

def cleanup_intermediate(case_dir: Path) -> None:
    to_remove = [
        case_dir / "pseudo_brain.nii.gz",
        case_dir / "brain.nii.gz",
        case_dir / "skull_strip.nii.gz",
        case_dir / "brain_edge.nii.gz",
        case_dir / "tissue_raw.nii.gz",
        case_dir / "dk-struct_raw.nii.gz",
    ]

    if not KEEP_T1_COPY:
        to_remove.append(case_dir / "T1.nii.gz")

    for p in to_remove:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def is_case_done(case_dir: Path) -> bool:
    required = [case_dir / "tissue.nii.gz", case_dir / "dk-struct.nii.gz", case_dir / "present_labels.csv"]
    for p in required:
        if not p.exists() or p.stat().st_size <= 0:
            return False
    return True


def select_cases(all_t1n: list[Path]) -> list[Path]:
    if CASE_IDS:
        wanted = set(CASE_IDS)
        chosen = [p for p in all_t1n if case_id_from_t1n(p) in wanted]
        found = {case_id_from_t1n(p) for p in chosen}
        missing = wanted - found
        if missing:
            raise RuntimeError(f"These CASE_IDS were not found: {sorted(missing)}")
        return chosen

    chosen = all_t1n[START_INDEX:END_INDEX]
    if MAX_CASES is not None:
        chosen = chosen[:MAX_CASES]
    return chosen


def save_summary(summary_rows: list[dict], summary_csv: Path) -> None:
    df = pd.DataFrame(summary_rows)
    if "case_id" in df.columns:
        df = df.drop_duplicates(subset=["case_id"], keep="last")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 4. 单病例处理
# ============================================================

def process_one_case(t1n_path: Path, label_df: pd.DataFrame) -> dict:
    case_id = case_id_from_t1n(t1n_path)
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"

    if SKIP_EXISTING and is_case_done(case_dir):
        return {
            "case_id": case_id,
            "status": "skipped_existing",
            "input_t1n": str(t1n_path),
            "output_dir": str(case_dir),
            "tissue_path": str(case_dir / "tissue.nii.gz"),
            "dk_struct_path": str(case_dir / "dk-struct.nii.gz"),
            "present_labels_csv": str(case_dir / "present_labels.csv"),
            "log_path": str(log_path),
            "error": "",
        }

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"TIME: {now_str()}\n")
        f.write(f"Case: {case_id}\n")
        f.write(f"Input: {t1n_path}\n")
        f.write(f"Output: {case_dir}\n")
        f.write(f"ALIGN_OUTPUT_TO_T1_GRID: {ALIGN_OUTPUT_TO_T1_GRID}\n")

    # 1) 准备 T1.nii.gz，作为 BrainParc 输入和最终对齐参考
    t1_copy = case_dir / "T1.nii.gz"
    shutil.copy2(t1n_path, t1_copy)

    pseudo_brain = case_dir / "pseudo_brain.nii.gz"
    brain = case_dir / "brain.nii.gz"
    skull_strip = case_dir / "skull_strip.nii.gz"
    brain_edge = case_dir / "brain_edge.nii.gz"

    tissue = case_dir / "tissue.nii.gz"
    dk_struct = case_dir / "dk-struct.nii.gz"

    if ALIGN_OUTPUT_TO_T1_GRID:
        tissue_raw = case_dir / "tissue_raw.nii.gz"
        dk_struct_raw = case_dir / "dk-struct_raw.nii.gz"
        brainparc_tissue_out = tissue_raw
        brainparc_dk_out = dk_struct_raw
    else:
        tissue_raw = tissue
        dk_struct_raw = dk_struct
        brainparc_tissue_out = tissue
        brainparc_dk_out = dk_struct

    present_labels_csv = case_dir / "present_labels.csv"

    # 2) AutoStrip Step01: pseudo brain extraction
    run_cmd(
        [
            PYTHON_BIN,
            "./Code/Inference/Step01_Pseudo_BrainExtraction.py",
            "--input", str(t1_copy),
            "--output", str(pseudo_brain),
            "--RefImg", "./Atlas/adult/img-T1.nii.gz",
            "--RefSeg", "./Atlas/adult/seg-T1.nii.gz",
        ],
        cwd=AUTOSTRIP_DIR,
        log_path=log_path,
    )

    # 3) AutoStrip Step02: skull stripping
    run_cmd(
        [
            PYTHON_BIN,
            "./Code/Inference/Step02_AutoStrip_Skull_Stripping.py",
            "--model_path", str(AUTOSTRIP_MODEL),
            "--input", str(pseudo_brain),
            "--output_brain", str(brain),
            "--output_brain_mask", str(skull_strip),
            "--norm_orientation", str(AUTOSTRIP_NORM_ORIENTATION),
            "--norm_spacing", str(AUTOSTRIP_NORM_SPACING),
        ],
        cwd=AUTOSTRIP_DIR,
        log_path=log_path,
    )

    # 4) BrainParc Step01: edge extraction
    run_cmd(
        [
            PYTHON_BIN,
            "./Code/Inference/Step01_Intensity_2_Edge.py",
            "--input", str(brain),
            "--output", str(brain_edge),
        ],
        cwd=BRAINPARC_DIR,
        log_path=log_path,
    )

    # 5) BrainParc Step02: tissue + dk-struct
    # 不传 --standard_space，保持你之前跑通版本一致。
    run_cmd(
        [
            PYTHON_BIN,
            "./Code/Inference/Step02_BrainParc.py",
            "--model_path", str(BRAINPARC_MODEL),
            "--input_brain", str(brain),
            "--input_edge", str(brain_edge),
            "--output_tissue", str(brainparc_tissue_out),
            "--output_dk", str(brainparc_dk_out),
            "--norm_orientation", str(BRAINPARC_NORM_ORIENTATION),
            "--norm_spacing", str(BRAINPARC_NORM_SPACING),
        ],
        cwd=BRAINPARC_DIR,
        log_path=log_path,
    )

    # 6) 对齐到原始 T1N 网格，确保下一步切片能稳定对齐
    if ALIGN_OUTPUT_TO_T1_GRID:
        resample_label_to_ref_grid(tissue_raw, t1_copy, tissue, log_path)
        resample_label_to_ref_grid(dk_struct_raw, t1_copy, dk_struct, log_path)

    qc = final_qc_against_t1(t1_copy, tissue, dk_struct, log_path)

    # 7) 导出该病例实际出现的 BrainParc 标签
    build_present_labels_csv(dk_struct, label_df, present_labels_csv)

    # 8) 清理中间文件
    if CLEAN_INTERMEDIATE:
        cleanup_intermediate(case_dir)

    return {
        "case_id": case_id,
        "status": "ok",
        "input_t1n": str(t1n_path),
        "output_dir": str(case_dir),
        "tissue_path": str(tissue),
        "dk_struct_path": str(dk_struct),
        "present_labels_csv": str(present_labels_csv),
        "log_path": str(log_path),
        "t1_shape": str(qc["t1_shape"]),
        "tissue_shape": str(qc["tissue_shape"]),
        "dk_shape": str(qc["dk_shape"]),
        "error": "",
    }


# ============================================================
# 5. 主流程
# ============================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")
    if not AUTOSTRIP_DIR.exists():
        raise FileNotFoundError(f"AUTOSTRIP_DIR not found: {AUTOSTRIP_DIR}")
    if not BRAINPARC_DIR.exists():
        raise FileNotFoundError(f"BRAINPARC_DIR not found: {BRAINPARC_DIR}")
    if not AUTOSTRIP_MODEL.exists():
        raise FileNotFoundError(f"AutoStrip model not found: {AUTOSTRIP_MODEL}")
    if not BRAINPARC_MODEL.exists():
        raise FileNotFoundError(f"BrainParc model not found: {BRAINPARC_MODEL}")
    if not LABEL_XLSX.exists():
        raise FileNotFoundError(f"Label_Cor.xlsx not found: {LABEL_XLSX}")

    label_df = load_label_table(LABEL_XLSX)
    all_t1n = find_t1n_files(INPUT_ROOT)
    if not all_t1n:
        raise RuntimeError(f"No '*-t1n.nii.gz' found under: {INPUT_ROOT}")

    chosen = select_cases(all_t1n)
    if not chosen:
        raise RuntimeError("No cases selected. Check CASE_IDS / START_INDEX / END_INDEX / MAX_CASES.")

    summary_csv = OUT_ROOT / "summary.csv"
    summary_rows: list[dict] = []
    if summary_csv.exists():
        try:
            summary_rows = pd.read_csv(summary_csv).to_dict("records")
        except Exception:
            summary_rows = []

    print("=" * 80)
    print(f"INPUT_ROOT      : {INPUT_ROOT}")
    print(f"OUT_ROOT        : {OUT_ROOT}")
    print(f"Total t1n found : {len(all_t1n)}")
    print(f"Selected cases  : {len(chosen)}")
    print(f"SKIP_EXISTING   : {SKIP_EXISTING}")
    print(f"ALIGN_TO_T1_GRID: {ALIGN_OUTPUT_TO_T1_GRID}")
    print(f"KEEP_T1_COPY    : {KEEP_T1_COPY}")
    print(f"CLEAN_INTER     : {CLEAN_INTERMEDIATE}")
    print("=" * 80)

    selected_csv = OUT_ROOT / "selected_cases.csv"
    pd.DataFrame(
        [
            {"order": i + 1, "case_id": case_id_from_t1n(p), "t1n_path": str(p)}
            for i, p in enumerate(chosen)
        ]
    ).to_csv(selected_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] selected case list saved to: {selected_csv}")

    for i, t1n_path in enumerate(chosen, start=1):
        case_id = case_id_from_t1n(t1n_path)
        print(f"\n[{i}/{len(chosen)}] Processing {case_id} ...", flush=True)

        try:
            row = process_one_case(t1n_path, label_df)
            print(f"  {row['status'].upper()}: {case_id}", flush=True)
        except Exception as e:
            out_dir = OUT_ROOT / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "run.log"
            append_log(
                log_path,
                "\n" + "!" * 100 + "\n"
                + f"FAILED at {now_str()}\n"
                + f"ERROR: {repr(e)}\n"
                + "!" * 100 + "\n",
            )
            row = {
                "case_id": case_id,
                "status": "failed",
                "input_t1n": str(t1n_path),
                "output_dir": str(out_dir),
                "tissue_path": "",
                "dk_struct_path": "",
                "present_labels_csv": "",
                "log_path": str(log_path),
                "error": repr(e),
            }
            print(f"  FAILED: {case_id}", flush=True)
            print(f"  Error: {repr(e)}", flush=True)

        summary_rows.append(row)
        save_summary(summary_rows, summary_csv)

        try:
            tmp_df = pd.DataFrame(summary_rows).drop_duplicates(subset=["case_id"], keep="last")
            print(f"  Current summary: {tmp_df['status'].value_counts().to_dict()}", flush=True)
        except Exception:
            pass

    final_df = pd.DataFrame(summary_rows).drop_duplicates(subset=["case_id"], keep="last")
    final_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Summary saved to: {summary_csv}")
    print(final_df["status"].value_counts())
    print("=" * 80)


if __name__ == "__main__":
    main()
