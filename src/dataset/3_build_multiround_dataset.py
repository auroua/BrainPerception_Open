
"""
build_tables_and_multiround_dataset_full_v5.py

功能：
1. 基于已切好的 2D T1N + BrainParc tissue/region PNG + manifest.jsonl，
   生成四张基础表格：
   - case_level_table.csv
   - slice_level_table.csv
   - region_visible_table.csv
   - tissue_visible_table.csv

2. 基于 3D BrainParc tissue.nii.gz 和 dk-struct.nii.gz，
   生成脑区-组织对应关系表：
   - region_tissue_correspondence_table.csv

3. 生成 full 多轮 reasoning segmentation 数据集：
   - tissue_to_region
   - contralateral_same_region
   - same_side_same_lobe
   - spatial_named_region
   - tumor_to_overlapping_region

v5 改进：
- 明确 BrainParc tissue label：
  1 = CSF / 脑脊液相关区域
  2 = GM / 灰质区域
  3 = WM / 白质区域
- tissue_to_region 使用均衡采样，避免总选同一 tissue
- 新增 spatial_named_region 类型
- spatial / contralateral / same_side 模板只使用 fine_anatomical_region，
  避免 CSF / Cerebral_WM_L / Cerebral_WM_R 进入细脑区推理模板
- tissue_to_region 问句改成“组织属性对应”，避免误解为严格几何包含
- same_side_same_lobe 根据结构类型改写措辞，皮层下结构不再写成“同一脑叶范围”
- spatial_named_region 优先选择同侧脑区；同侧候选不足时再允许跨侧
- 脑区显示名做轻量 prettify，例如 Frontalpole -> Frontal pole
- 新增 tumor_to_overlapping_region：BraTS 肿瘤区域 -> 与其空间重叠面积最大的 BrainParc 细脑区

运行：
    cd /root/autodl-tmp
    python build_tables_and_multiround_dataset_full_v5.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append("/mnt/rna01/chenw/Workspaces/BrainPerception")

from src.dataset.data_pathes import instance_out_dir, instance_seg_root_dir, original_instance_dir


try:
    from multiround_templates import (
        DIALOGUE_CATEGORY_DESCRIPTIONS,
        DIALOGUE_CATEGORY_ORDER,
        DIRECTION_ZH,
        TEMPLATE_POOL_SIZES,
        render_template,
        summarize_template_usage,
    )
except ImportError:
    from .multiround_templates import (
        DIALOGUE_CATEGORY_DESCRIPTIONS,
        DIALOGUE_CATEGORY_ORDER,
        DIRECTION_ZH,
        TEMPLATE_POOL_SIZES,
        render_template,
        summarize_template_usage,
    )


# ============================================================
# 0. 路径和全局参数：按需只改这里
# ============================================================

# PROCESSED_ROOT = Path("/root/autodl-tmp/Processed_T1N_BrainParc_2D_test3_seed2")
PROCESSED_ROOT = Path(instance_out_dir)

MANIFEST_PATH = PROCESSED_ROOT / "manifest.jsonl"

# BRAINPARC_ROOT = Path("/root/autodl-tmp/brainparc_test3")
BRAINPARC_ROOT = Path(instance_seg_root_dir)

# INPUT_ROOT = Path("/root/autodl-tmp/TrainingData")
INPUT_ROOT = Path(original_instance_dir)

TABLE_DIR = PROCESSED_ROOT / "tables"
MULTIROUND_DIR = PROCESSED_ROOT / "multiround_dataset"
MULTIROUND_MASK_DIR = MULTIROUND_DIR / "masks"
MULTIROUND_JSONL = MULTIROUND_DIR / "multiround_dialogues.jsonl"
SUMMARY_JSON = MULTIROUND_DIR / "summary.json"

# 是否清理旧 tables 和 multiround_dataset
CLEAN_OLD_OUTPUTS = True

# 每张 slice 上 label 至少多少像素才算可见
MIN_REGION_AREA_PX = 100
MIN_TISSUE_AREA_PX = 100

# 过滤极小或异常铺满的 region；避免几乎不可见的目标进入多轮训练。
MIN_REGION_AREA_FRAC = 0.0005
MAX_FINE_REGION_AREA_FRAC = 0.35

# bbox 内有效像素占比过低时，通常说明 mask 太碎，不适合作为清晰训练目标。
MIN_REGION_BBOX_FILL_FRAC = 0.05

# 生成 spatial_named_region 时，两个脑区中心点归一化坐标差异阈值
MIN_SPATIAL_DELTA = 0.12

# 生成 spatial_named_region 时，目标区域最少像素
MIN_SPATIAL_REGION_AREA_PX = 100

# 为了避免单张 slice 生成过多样本，同时覆盖更多 region 组合。
MAX_DIALOGUES_PER_SLICE_PER_TYPE = 3
MAX_DIALOGUES_PER_SLICE_TOTAL = 14

# 是否生成单轮直接分割样本。该类别不依赖多轮关系推理，专门增强基础分割能力。
ENABLE_BASIC_SEGMENTATION = True
MAX_BASIC_SEGMENTATION_PER_SLICE = 4

# SegLLM / LISA-style segmentation placeholder token used in assistant answers.
SEG_TOKEN = "<SEG>"

# BrainParc tissue label 真实含义
# 源码依据：
#   csf = UII_Tissue == 1
#   gm  = UII_Tissue == 2
#   wm  = UII_Tissue == 3
TISSUE_LABEL_NAME_ZH = {
    1: "脑脊液相关区域",
    2: "灰质区域",
    3: "白质区域",
}

TISSUE_LABEL_NAME_EN = {
    1: "CSF",
    2: "gray matter",
    3: "white matter",
}


# ============================================================
# 第五类模板：tumor_to_overlapping_region 参数
# ============================================================

# 是否启用第五类“病灶-解剖锚定”模板。
ENABLE_TUMOR_TO_OVERLAPPING_REGION = True

# 第一版最稳：用真正肿瘤相关标签 1/2/3，不包含 4=RC 切除腔。
# BraTS2024 GLI 常用含义：
#   1 = NETC，坏死或非强化肿瘤核心
#   2 = SNFH，水肿或非强化异常信号区域
#   3 = ET，增强肿瘤区域
#   4 = RC，切除腔区域，第一版不纳入“肿瘤区域”union。
TUMOR_UNION_LABELS = (1, 2, 3)
TUMOR_UNION_NAME_ZH = "肿瘤相关异常区域"

# 过滤条件：避免只有几个像素重叠也生成样本。
MIN_TUMOR_AREA_PX = 30
MIN_TUMOR_REGION_OVERLAP_PX = 20
MIN_OVERLAP_FRAC_OF_TUMOR = 0.05


# ============================================================
# 1. 基础工具
# ============================================================

def ensure_clean_dir(path: Path) -> None:
    if path.exists() and CLEAN_OLD_OUTPUTS:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rel_to_root(path_like: str | Path, root: Path = PROCESSED_ROOT) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def safe_literal_eval(x: Any) -> Any:
    if isinstance(x, (list, tuple, dict)):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse_list2(x: Any) -> Optional[Tuple[float, float]]:
    v = safe_literal_eval(x)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None
    return None


def bbox_centroid_from_mask(mask: np.ndarray) -> dict:
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "bbox_xyxy": None,
            "bbox_norm_xyxy": None,
            "centroid_xy": None,
            "centroid_norm_xy": None,
        }

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())

    return {
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_norm_xyxy": [
            float(xmin / max(w, 1)),
            float(ymin / max(h, 1)),
            float(xmax / max(w, 1)),
            float(ymax / max(h, 1)),
        ],
        "centroid_xy": [cx, cy],
        "centroid_norm_xy": [
            float(cx / max(w, 1)),
            float(cy / max(h, 1)),
        ],
    }


def mask_bbox_fill_fraction(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0

    bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if bbox_area <= 0:
        return 0.0
    return float(mask.sum() / bbox_area)


def is_quality_region_mask(mask: np.ndarray, category: str, area_frac: float) -> bool:
    if area_frac < MIN_REGION_AREA_FRAC:
        return False

    if category == "fine_anatomical_region" and area_frac > MAX_FINE_REGION_AREA_FRAC:
        return False

    if mask_bbox_fill_fraction(mask) < MIN_REGION_BBOX_FILL_FRAC:
        return False

    return True


def save_binary_mask(mask: np.ndarray, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask.astype(bool).astype(np.uint8) * 255)
    Image.fromarray(arr).save(out_path, compress_level=0)
    return str(out_path.relative_to(PROCESSED_ROOT))


def read_label_png(rel_path: str) -> np.ndarray:
    p = rel_to_root(rel_path)
    return np.array(Image.open(p))


def load_nifti_int(path: Path) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = img.get_fdata(dtype=np.float32)
    arr = np.rint(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.int32)
    return arr


# ============================================================
# 2. label map 和医学属性推断
# ============================================================

def find_file_recursive(root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        cands = sorted(root.rglob(name))
        if cands:
            return cands[0]
    return None


def find_brainparc_case_dir(case_id: str) -> Optional[Path]:
    direct = BRAINPARC_ROOT / case_id
    if direct.exists() and direct.is_dir():
        return direct
    cands = sorted([p for p in BRAINPARC_ROOT.glob(f"*{case_id}*") if p.is_dir()])
    return cands[0] if cands else None


def load_label_map_from_present_csv(case_bp_dir: Path) -> Dict[int, str]:
    """
    present_labels.csv 一般来自你的 run_brainparc_full_batch 脚本，
    常见列包含 BrainParc_index 和 label/name 类列。
    """
    path = find_file_recursive(case_bp_dir, ["present_labels.csv"])
    if path is None:
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    cols = list(df.columns)

    # id 列
    id_col = None
    for c in cols:
        if c.lower() in ["brainparc_index", "index", "label", "id"]:
            id_col = c
            break
    if id_col is None:
        for c in cols:
            try:
                pd.to_numeric(df[c], errors="raise")
                id_col = c
                break
            except Exception:
                pass

    if id_col is None:
        return {}

    # 名称列：优先选择非数值文本列
    name_col = None
    preferred_keywords = ["name", "label", "structure", "region", "roi", "abbr"]
    for c in cols:
        if c == id_col:
            continue
        if any(k in c.lower() for k in preferred_keywords):
            # 至少有一些文本
            sample = df[c].dropna().astype(str).head(20).tolist()
            if any(re.search(r"[A-Za-z]", x) for x in sample):
                name_col = c
                break

    if name_col is None:
        for c in cols:
            if c == id_col:
                continue
            sample = df[c].dropna().astype(str).head(20).tolist()
            if any(re.search(r"[A-Za-z]", x) for x in sample):
                name_col = c
                break

    if name_col is None:
        return {}

    out = {}
    for _, row in df.iterrows():
        try:
            lab = int(float(row[id_col]))
            name = str(row[name_col]).strip()
            if lab > 0 and name and name.lower() != "nan":
                out[lab] = name
        except Exception:
            continue
    return out


def load_global_label_map(case_ids: Iterable[str]) -> Dict[int, str]:
    """
    从多个 case 的 present_labels.csv 合并 label map。
    """
    mp: Dict[int, str] = {}
    for case_id in case_ids:
        bp_dir = find_brainparc_case_dir(case_id)
        if bp_dir is None:
            continue
        sub = load_label_map_from_present_csv(bp_dir)
        for k, v in sub.items():
            mp.setdefault(int(k), str(v))
    return mp


def infer_left_right_from_region_name(name: str) -> str:
    """
    支持：
    - Precentral_L / Precentral_R
    - Occipital_Lat_L / Occipital_Lat_R
    - Cerebral_WM_L / Cerebral_WM_R
    - ctx-lh-xxx / ctx-rh-xxx
    - left / right / lh / rh
    """
    if name is None:
        return "unknown"

    s = str(name).strip().lower()

    left_patterns = [
        r"(^|[_\-\s])left([_\-\s]|$)",
        r"(^|[_\-\s])lh([_\-\s]|$)",
        r"ctx-lh",
        r"[_\-]l$",
        r"^l[_\-]",
    ]

    right_patterns = [
        r"(^|[_\-\s])right([_\-\s]|$)",
        r"(^|[_\-\s])rh([_\-\s]|$)",
        r"ctx-rh",
        r"[_\-]r$",
        r"^r[_\-]",
    ]

    for pat in left_patterns:
        if re.search(pat, s):
            return "left"
    for pat in right_patterns:
        if re.search(pat, s):
            return "right"

    return "unknown"


def infer_lobe_from_region_name(name: str) -> str:
    if name is None:
        return "unknown"
    s = str(name).lower()

    if any(k in s for k in ["frontal", "precentral", "paracentral", "pars", "orbitofrontal", "front"]):
        return "frontal_lobe"
    if any(k in s for k in ["temporal", "fusiform", "entorhinal", "parahippocampal", "hippocampus", "hippocampal", "bankssts"]):
        return "temporal_lobe"
    if any(k in s for k in ["parietal", "postcentral", "precuneus", "supramarginal"]):
        return "parietal_lobe"
    if any(k in s for k in ["occipital", "lingual", "cuneus", "pericalcarine"]):
        return "occipital_lobe"
    if any(k in s for k in ["insula", "insular"]):
        return "insula"
    if any(k in s for k in ["ventricle", "ventricular", "csf"]):
        return "ventricle_or_csf"
    if any(k in s for k in ["caudate", "putamen", "pallidum", "thalamus", "accumbens", "amygdala"]):
        return "subcortical_or_white_matter"
    if any(k in s for k in ["cerebral_wm", "white_matter", "wm"]):
        return "subcortical_or_white_matter"
    if any(k in s for k in ["brainstem", "cerebellum"]):
        return "brainstem_or_cerebellum"

    return "unknown"


def classify_region_name(name: str) -> str:
    """
    region label 分三类：
    - tissue_like_region：CSF, Cerebral_WM_L/R 等大组织/大结构
    - fine_anatomical_region：左右侧明确的细粒度脑区
    - midline_or_unknown_region：中线或无法判断左右的结构
    """
    if name is None:
        return "unknown"

    s = str(name).strip().lower()

    tissue_like_exact = {"csf", "cerebrospinal_fluid"}
    tissue_like_keywords = [
        "cerebral_wm",
        "white_matter",
        "brainparc 组织",
        "brainparc组织",
    ]

    if s in tissue_like_exact:
        return "tissue_like_region"
    if any(k in s for k in tissue_like_keywords):
        return "tissue_like_region"

    side = infer_left_right_from_region_name(name)
    if side in ["left", "right"]:
        return "fine_anatomical_region"

    return "midline_or_unknown_region"


def strip_lr_suffix(name: str) -> str:
    x = str(name).strip()
    x = re.sub(r"[_\-][LlRr]$", "", x)
    return x


REGION_BASE_REWRITE = {
    "Frontalpole": "Frontal pole",
    "Temporalpole": "Temporal pole",
    "Isthmuscingulate": "Isthmus cingulate",
    "Transversetemporal": "Transverse temporal",
    "Parsopercularis": "Pars opercularis",
    "Parsorbitalis": "Pars orbitalis",
    "Parstriangularis": "Pars triangularis",
    "Pericalcarine": "Pericalcarine",
    "Postcentral": "Postcentral",
    "Precentral": "Precentral",
    "Precuneus": "Precuneus",
    "Supramarginal": "Supramarginal",
    "Parahippocampal": "Parahippocampal",
    "Orbitofrontal": "Orbitofrontal",
    "Cingulum": "Cingulum",
    "VentralDC": "Ventral DC",
}


def prettify_region_base_name(base: str) -> str:
    """
    把 BrainParc 原始 label 名转成更可读的显示名。
    例如：
    Frontalpole -> Frontal pole
    Isthmuscingulate -> Isthmus cingulate
    Frontal_Mid_Caudal -> Frontal Mid Caudal
    VentralDC -> Ventral DC
    """
    x = str(base).strip()
    x = x.replace("_", " ")

    if x in REGION_BASE_REWRITE:
        return REGION_BASE_REWRITE[x]

    parts = x.split()
    parts = [REGION_BASE_REWRITE.get(p, p) for p in parts]
    return " ".join(parts)


def pretty_region_name_zh(name: str) -> str:
    side = infer_left_right_from_region_name(name)
    base_raw = strip_lr_suffix(name)
    base = prettify_region_base_name(base_raw)

    if side == "left":
        return f"左侧 {base} 区域"
    if side == "right":
        return f"右侧 {base} 区域"
    return f"{base} 区域"


def pretty_tissue_name_zh(label: int) -> str:
    return TISSUE_LABEL_NAME_ZH.get(int(label), f"BrainParc 组织标签{int(label)}")


def pretty_tissue_name_en(label: int) -> str:
    return TISSUE_LABEL_NAME_EN.get(int(label), f"BrainParc tissue label {int(label)}")


def contralateral_region_name(name: str) -> Optional[str]:
    """
    Precentral_L -> Precentral_R
    Temporal_Mid_R -> Temporal_Mid_L
    """
    if name is None:
        return None
    s = str(name).strip()
    if re.search(r"[_\-]L$", s):
        return re.sub(r"([_\-])L$", r"\1R", s)
    if re.search(r"[_\-]R$", s):
        return re.sub(r"([_\-])R$", r"\1L", s)
    if re.search(r"[_\-]l$", s):
        return re.sub(r"([_\-])l$", r"\1r", s)
    if re.search(r"[_\-]r$", s):
        return re.sub(r"([_\-])r$", r"\1l", s)
    return None


# ============================================================
# 3. 表格生成
# ============================================================

def build_case_and_slice_tables(manifest_rows: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slice_rows = []
    case_acc = {}

    for d in manifest_rows:
        case_id = d["case"]
        slice_id = d["id"]
        plane = d["plane"]
        slice_index = int(d["slice"])

        bp = d.get("brainparc", {})
        brats_seg = d.get("brats_seg")

        region_stats = bp.get("region_stats", {})
        tissue_stats = bp.get("tissue_stats", {})

        images_dict = get_manifest_images(d)
        image_rel = get_primary_image(d)
        modalities = get_manifest_modalities(d)
        main_modality = str(d.get("main_modality", "t1n"))

        tissue_rel = bp.get("tissue")
        region_rel = bp.get("region")

        slice_rows.append({
            "slice_id": slice_id,
            "case_id": case_id,
            "plane": plane,
            "slice_index": slice_index,
            "t1n_path": images_dict.get("t1n", image_rel),
            "primary_image": image_rel,
            "main_modality": main_modality,
            "modalities_json": json.dumps(modalities, ensure_ascii=False),
            "images_json": json.dumps(images_dict, ensure_ascii=False),
            "brainparc_tissue_path": tissue_rel,
            "brainparc_region_path": region_rel,
            "n_visible_regions": int(region_stats.get("n_visible_labels", 0)),
            "n_visible_tissues": int(tissue_stats.get("n_visible_labels", 0)),
            "has_brats_seg": brats_seg is not None,
            "view_json": json.dumps(d.get("view", {}), ensure_ascii=False),
        })

        if case_id not in case_acc:
            bp_dir = find_brainparc_case_dir(case_id)
            t1n_path = INPUT_ROOT / case_id / f"{case_id}-t1n.nii.gz"
            case_acc[case_id] = {
                "case_id": case_id,
                "t1n_path": str(t1n_path),
                "brainparc_case_dir": str(bp_dir) if bp_dir else "",
                "tissue_path": str(bp_dir / "tissue.nii.gz") if bp_dir else "",
                "dk_struct_path": str(bp_dir / "dk-struct.nii.gz") if bp_dir else "",
                "present_labels_csv": str(bp_dir / "present_labels.csv") if bp_dir else "",
                "n_slices_saved": 0,
                "n_axial": 0,
                "n_coronal": 0,
                "n_sagittal": 0,
            }

        case_acc[case_id]["n_slices_saved"] += 1
        if plane == "axial":
            case_acc[case_id]["n_axial"] += 1
        elif plane == "coronal":
            case_acc[case_id]["n_coronal"] += 1
        elif plane == "sagittal":
            case_acc[case_id]["n_sagittal"] += 1

    case_df = pd.DataFrame(list(case_acc.values())).sort_values("case_id").reset_index(drop=True)
    slice_df = pd.DataFrame(slice_rows).sort_values(["case_id", "plane", "slice_index"]).reset_index(drop=True)
    return case_df, slice_df


def build_visible_tables(manifest_rows: List[dict], label_map: Dict[int, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    tissue_rows = []

    for d in tqdm(manifest_rows, desc="Building visible tables"):
        slice_id = d["id"]
        case_id = d["case"]
        plane = d["plane"]
        slice_index = int(d["slice"])
        bp = d.get("brainparc", {})

        # region
        region_rel = bp.get("region")
        if region_rel:
            region_arr = read_label_png(region_rel).astype(np.int64)
            h, w = region_arr.shape
            labels = sorted([int(x) for x in np.unique(region_arr) if int(x) != 0])

            for lab in labels:
                mask = region_arr == lab
                area = int(mask.sum())
                if area < MIN_REGION_AREA_PX:
                    continue

                name = label_map.get(lab, f"label_{lab}")
                category = classify_region_name(name)
                area_frac = float(area / max(h * w, 1))
                if not is_quality_region_mask(mask, category, area_frac):
                    continue

                geom = bbox_centroid_from_mask(mask)
                side = infer_left_right_from_region_name(name)
                lobe = infer_lobe_from_region_name(name)

                region_rows.append({
                    "slice_id": slice_id,
                    "case_id": case_id,
                    "plane": plane,
                    "slice_index": slice_index,
                    "region_label": lab,
                    "region_name": name,
                    "display_name_zh": pretty_region_name_zh(name),
                    "region_category": category,
                    "area_px": area,
                    "area_frac_of_image": area_frac,
                    "bbox_xyxy": json.dumps(geom["bbox_xyxy"], ensure_ascii=False),
                    "bbox_norm_xyxy": json.dumps(geom["bbox_norm_xyxy"], ensure_ascii=False),
                    "centroid_xy": json.dumps(geom["centroid_xy"], ensure_ascii=False),
                    "centroid_norm_xy": json.dumps(geom["centroid_norm_xy"], ensure_ascii=False),
                    "side": side,
                    "lobe": lobe,
                })

        # tissue
        tissue_rel = bp.get("tissue")
        if tissue_rel:
            tissue_arr = read_label_png(tissue_rel).astype(np.int64)
            h, w = tissue_arr.shape
            labels = sorted([int(x) for x in np.unique(tissue_arr) if int(x) != 0])

            for lab in labels:
                mask = tissue_arr == lab
                area = int(mask.sum())
                if area < MIN_TISSUE_AREA_PX:
                    continue

                geom = bbox_centroid_from_mask(mask)
                tissue_rows.append({
                    "slice_id": slice_id,
                    "case_id": case_id,
                    "plane": plane,
                    "slice_index": slice_index,
                    "tissue_label": lab,
                    "tissue_name_zh": pretty_tissue_name_zh(lab),
                    "tissue_name_en": pretty_tissue_name_en(lab),
                    "area_px": area,
                    "area_frac_of_image": float(area / max(h * w, 1)),
                    "bbox_xyxy": json.dumps(geom["bbox_xyxy"], ensure_ascii=False),
                    "bbox_norm_xyxy": json.dumps(geom["bbox_norm_xyxy"], ensure_ascii=False),
                    "centroid_xy": json.dumps(geom["centroid_xy"], ensure_ascii=False),
                    "centroid_norm_xy": json.dumps(geom["centroid_norm_xy"], ensure_ascii=False),
                })

    region_df = pd.DataFrame(region_rows)
    tissue_df = pd.DataFrame(tissue_rows)
    return region_df, tissue_df


def build_region_tissue_correspondence(case_df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:
    """
    用 3D tissue.nii.gz 和 dk-struct.nii.gz 统计每个 region label 与 tissue label 的重叠。
    """
    acc: Dict[int, Counter] = defaultdict(Counter)
    case_count: Counter = Counter()

    for _, row in tqdm(case_df.iterrows(), total=len(case_df), desc="Building region-tissue correspondence"):
        case_id = row["case_id"]
        bp_dir = find_brainparc_case_dir(case_id)
        if bp_dir is None:
            continue

        tissue_path = bp_dir / "tissue.nii.gz"
        dk_path = bp_dir / "dk-struct.nii.gz"
        if not tissue_path.exists() or not dk_path.exists():
            continue

        tissue = load_nifti_int(tissue_path)
        dk = load_nifti_int(dk_path)

        if tissue.shape != dk.shape:
            print(f"[WARN] shape mismatch for {case_id}: tissue={tissue.shape}, dk={dk.shape}")
            continue

        labels = sorted([int(x) for x in np.unique(dk) if int(x) != 0])
        for lab in labels:
            region_mask = dk == lab
            vals, counts = np.unique(tissue[region_mask], return_counts=True)
            for v, c in zip(vals, counts):
                v = int(v)
                c = int(c)
                if v == 0:
                    continue
                acc[lab][v] += c
            case_count[lab] += 1

    rows = []
    for region_label, cnt in sorted(acc.items()):
        total = int(sum(cnt.values()))
        if total <= 0:
            continue

        dominant_tissue_label, dominant_count = cnt.most_common(1)[0]
        dominant_ratio = float(dominant_count / total)
        region_name = label_map.get(region_label, f"label_{region_label}")

        if dominant_ratio >= 0.80:
            confidence = "high"
        elif dominant_ratio >= 0.60:
            confidence = "medium"
        else:
            confidence = "low"

        rows.append({
            "region_label": int(region_label),
            "region_name": region_name,
            "dominant_tissue_label": int(dominant_tissue_label),
            "dominant_tissue_name_zh": pretty_tissue_name_zh(dominant_tissue_label),
            "dominant_tissue_name_en": pretty_tissue_name_en(dominant_tissue_label),
            "dominant_tissue_ratio": dominant_ratio,
            "region_total_voxels": total,
            "dominant_overlap_voxels": int(dominant_count),
            "all_tissue_overlap_json": json.dumps({str(k): int(v) for k, v in sorted(cnt.items())}, ensure_ascii=False),
            "case_count_present": int(case_count[region_label]),
            "confidence": confidence,
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. 多轮数据集生成辅助
# ============================================================

def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_df = pd.read_csv(TABLE_DIR / "case_level_table.csv")
    slice_df = pd.read_csv(TABLE_DIR / "slice_level_table.csv")
    region_df = pd.read_csv(TABLE_DIR / "region_visible_table.csv")
    tissue_df = pd.read_csv(TABLE_DIR / "tissue_visible_table.csv")
    corr_df = pd.read_csv(TABLE_DIR / "region_tissue_correspondence_table.csv")
    return case_df, slice_df, region_df, tissue_df, corr_df


def get_region_to_tissue_map(corr_df: pd.DataFrame) -> Dict[int, int]:
    mp = {}
    for _, r in corr_df.iterrows():
        try:
            mp[int(r["region_label"])] = int(r["dominant_tissue_label"])
        except Exception:
            continue
    return mp


def get_slice_manifest_map(manifest_rows: List[dict]) -> Dict[str, dict]:
    return {d["id"]: d for d in manifest_rows}


def get_manifest_images(slice_manifest: dict) -> Dict[str, str]:
    """
    兼容单模态和四模态 manifest。

    四模态切片脚本中，manifest 每行包含：
        images: {t1n, t1c, t2w, t2f}
        primary_image: t1n 路径
        modalities: [t1n, t1c, t2w, t2f]

    老版本单模态 manifest 至少包含 images["t1n"]。
    """
    images = slice_manifest.get("images", {})
    if isinstance(images, dict) and images:
        return {str(k): str(v) for k, v in images.items()}

    image = slice_manifest.get("primary_image") or slice_manifest.get("image")
    if image:
        return {"t1n": str(image)}
    return {}


def get_primary_image(slice_manifest: dict) -> str:
    images = get_manifest_images(slice_manifest)
    return str(
        slice_manifest.get("primary_image")
        or images.get("t1n")
        or slice_manifest.get("image")
        or next(iter(images.values()), "")
    )


def get_manifest_modalities(slice_manifest: dict) -> List[str]:
    images = get_manifest_images(slice_manifest)
    mods = slice_manifest.get("modalities")
    if isinstance(mods, list) and mods:
        return [str(x) for x in mods]
    return list(images.keys())


def row_area_px(row: dict | pd.Series) -> float:
    try:
        return float(row.get("area_px", 0))
    except Exception:
        return 0.0


def region_row_has_enough_area(row: dict | pd.Series) -> bool:
    return row_area_px(row) >= MIN_REGION_AREA_PX


def tissue_row_has_enough_area(row: dict | pd.Series) -> bool:
    return row_area_px(row) >= MIN_TISSUE_AREA_PX


def choose_balanced_region(
    region_candidates: List[dict],
    region_usage_counter: Counter,
    excluded_region_labels: Optional[set[int]] = None,
) -> Optional[dict]:
    excluded_region_labels = excluded_region_labels or set()
    region_candidates = [
        r for r in region_candidates
        if (
            int(r["region_label"]) not in excluded_region_labels
            and region_row_has_enough_area(r)
        )
    ]
    if not region_candidates:
        return None
    # 使用次数少优先，面积大次之
    return sorted(
        region_candidates,
        key=lambda r: (
            region_usage_counter[int(r["region_label"])],
            -float(r.get("area_px", 0)),
            str(r.get("region_name", "")),
        )
    )[0]


def choose_balanced_tissue_for_slice(
    tissue_rows: pd.DataFrame,
    fine_region_rows: pd.DataFrame,
    region_to_tissue_map: Dict[int, int],
    tissue_usage_counter: Counter,
    region_usage_counter: Counter,
    excluded_tissue_labels: Optional[set[int]] = None,
    excluded_region_labels: Optional[set[int]] = None,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    选择 tissue_to_region 的 R1 tissue 和 R2 region。

    要求：
    - tissue 在当前 slice 可见
    - 当前 slice 有 fine region 的 dominant tissue 属于它
    - tissue 使用次数少优先
    - region 使用次数少优先
    """
    excluded_tissue_labels = excluded_tissue_labels or set()
    excluded_region_labels = excluded_region_labels or set()
    candidates = []

    for _, tr in tissue_rows.iterrows():
        tissue_label = int(tr["tissue_label"])
        if tissue_label in excluded_tissue_labels:
            continue
        if not tissue_row_has_enough_area(tr):
            continue

        matched_regions = []
        for _, rr in fine_region_rows.iterrows():
            region_label = int(rr["region_label"])
            if region_label in excluded_region_labels:
                continue
            if not region_row_has_enough_area(rr):
                continue
            if region_to_tissue_map.get(region_label) == tissue_label:
                matched_regions.append(rr.to_dict())

        if not matched_regions:
            continue

        chosen_region = choose_balanced_region(
            matched_regions,
            region_usage_counter,
            excluded_region_labels=excluded_region_labels,
        )
        if chosen_region is None:
            continue

        candidates.append({
            "tissue_row": tr.to_dict(),
            "region_row": chosen_region,
            "tissue_usage": tissue_usage_counter[tissue_label],
            "region_usage": region_usage_counter[int(chosen_region["region_label"])],
            "area_px": float(tr.get("area_px", 0)),
        })

    if not candidates:
        return None, None

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["tissue_usage"],
            x["region_usage"],
            -x["area_px"],
        )
    )

    chosen = candidates[0]
    return chosen["tissue_row"], chosen["region_row"]


def choose_contralateral_pair_for_slice(
    fine_region_rows: pd.DataFrame,
    region_usage_counter: Counter,
    pair_usage_counter: Counter,
    excluded_pair_keys: Optional[set[tuple]] = None,
    excluded_target_labels: Optional[set[int]] = None,
) -> Optional[Tuple[dict, dict]]:
    """
    选择对侧同名结构 pair。
    """
    excluded_pair_keys = excluded_pair_keys or set()
    excluded_target_labels = excluded_target_labels or set()
    rows = fine_region_rows.to_dict("records")
    by_name = {str(r["region_name"]): r for r in rows}

    candidates = []
    for ref in rows:
        if not region_row_has_enough_area(ref):
            continue

        ref_name = str(ref["region_name"])
        target_name = contralateral_region_name(ref_name)
        if target_name is None or target_name not in by_name:
            continue

        target = by_name[target_name]
        if not region_row_has_enough_area(target):
            continue

        ref_lab = int(ref["region_label"])
        tar_lab = int(target["region_label"])
        pair_key = (ref_lab, tar_lab, "contra")
        if pair_key in excluded_pair_keys or tar_lab in excluded_target_labels:
            continue

        candidates.append({
            "ref": ref,
            "target": target,
            "usage": pair_usage_counter[pair_key],
            "region_usage": region_usage_counter[tar_lab],
            "target_area": float(target.get("area_px", 0)),
            "pair_key": pair_key,
        })

    if not candidates:
        return None

    chosen = sorted(
        candidates,
        key=lambda x: (
            x["usage"],
            x["region_usage"],
            -x["target_area"],
        )
    )[0]
    return chosen["ref"], chosen["target"]


def choose_same_side_same_lobe_pair_for_slice(
    fine_region_rows: pd.DataFrame,
    region_usage_counter: Counter,
    pair_usage_counter: Counter,
    excluded_pair_keys: Optional[set[tuple]] = None,
    excluded_target_labels: Optional[set[int]] = None,
) -> Optional[Tuple[dict, dict]]:
    excluded_pair_keys = excluded_pair_keys or set()
    excluded_target_labels = excluded_target_labels or set()
    rows = fine_region_rows.to_dict("records")
    candidates = []

    for ref in rows:
        if not region_row_has_enough_area(ref):
            continue

        for target in rows:
            if int(ref["region_label"]) == int(target["region_label"]):
                continue
            if not region_row_has_enough_area(target):
                continue

            if str(ref.get("side")) != str(target.get("side")):
                continue
            if str(ref.get("lobe")) == "unknown" or str(ref.get("lobe")) != str(target.get("lobe")):
                continue

            ref_lab = int(ref["region_label"])
            tar_lab = int(target["region_label"])
            pair_key = (ref_lab, tar_lab, "same_side_same_lobe")
            if pair_key in excluded_pair_keys or tar_lab in excluded_target_labels:
                continue

            candidates.append({
                "ref": ref,
                "target": target,
                "usage": pair_usage_counter[pair_key],
                "region_usage": region_usage_counter[tar_lab],
                "target_area": float(target.get("area_px", 0)),
                "pair_key": pair_key,
            })

    if not candidates:
        return None

    chosen = sorted(
        candidates,
        key=lambda x: (
            x["usage"],
            x["region_usage"],
            -x["target_area"],
        )
    )[0]
    return chosen["ref"], chosen["target"]



def relation_scope_phrase_by_lobe(lobe: str, side_zh: str) -> str:
    """
    根据 lobe / structure group 生成更自然的关系描述。
    解决 Putamen/Caudate/Thalamus 这类皮层下结构被写成“同一脑叶范围”的问题。
    """
    lobe = str(lobe)

    cortical_lobes = {
        "frontal_lobe",
        "temporal_lobe",
        "parietal_lobe",
        "occipital_lobe",
        "insula",
    }

    if lobe in cortical_lobes:
        return f"{side_zh}同一脑叶范围"

    if lobe in {"subcortical_or_white_matter", "subcortical_nuclei"}:
        return f"{side_zh}皮层下深部结构范围"

    if lobe in {"ventricle_or_csf", "ventricle"}:
        return f"{side_zh}脑室系统相关结构范围"

    if lobe in {"brainstem_or_cerebellum"}:
        return "脑干或小脑相关结构范围"

    if lobe in {"hippocampal_region"}:
        return f"{side_zh}海马旁相关结构范围"

    return f"{side_zh}相关解剖结构范围"


def infer_image_direction(ref_row: dict, target_row: dict, min_delta: float = MIN_SPATIAL_DELTA) -> Optional[str]:
    c1 = parse_list2(ref_row.get("centroid_norm_xy"))
    c2 = parse_list2(target_row.get("centroid_norm_xy"))
    if c1 is None or c2 is None:
        return None

    ax, ay = c1
    bx, by = c2
    dx = bx - ax
    dy = by - ay

    if abs(dx) < min_delta and abs(dy) < min_delta:
        return None

    if abs(dx) >= abs(dy):
        if dx > min_delta:
            return "right"
        if dx < -min_delta:
            return "left"
    else:
        if dy > min_delta:
            return "down"
        if dy < -min_delta:
            return "up"

    return None


def choose_spatial_pair_for_slice(
    fine_region_rows: pd.DataFrame,
    region_usage_counter: Counter,
    pair_usage_counter: Counter,
    excluded_pair_keys: Optional[set[tuple]] = None,
    excluded_target_labels: Optional[set[int]] = None,
) -> Optional[Tuple[dict, dict, str]]:
    """
    从当前 slice 的 fine anatomical regions 中选择一对存在明显图像空间关系的 region。

    v4 改进：
    1. 优先选择同侧脑区之间的空间关系；
    2. 同侧候选不足时，再允许跨侧脑区；
    3. 仍然按 pair 使用次数少、target 使用次数少、target 面积大排序。
    """
    excluded_pair_keys = excluded_pair_keys or set()
    excluded_target_labels = excluded_target_labels or set()
    rows = fine_region_rows.to_dict("records")
    candidates = []

    for ref in rows:
        if not region_row_has_enough_area(ref):
            continue

        for target in rows:
            if int(ref["region_label"]) == int(target["region_label"]):
                continue
            if not region_row_has_enough_area(target):
                continue

            if float(ref.get("area_px", 0)) < MIN_SPATIAL_REGION_AREA_PX:
                continue
            if float(target.get("area_px", 0)) < MIN_SPATIAL_REGION_AREA_PX:
                continue

            direction = infer_image_direction(ref, target)
            if direction is None:
                continue

            ref_lab = int(ref["region_label"])
            tar_lab = int(target["region_label"])
            pair_key = (ref_lab, tar_lab, direction)
            if pair_key in excluded_pair_keys or tar_lab in excluded_target_labels:
                continue

            same_side = (
                str(ref.get("side")) in ["left", "right"]
                and str(ref.get("side")) == str(target.get("side"))
            )

            candidates.append({
                "ref": ref,
                "target": target,
                "direction": direction,
                "same_side": same_side,
                "usage": pair_usage_counter[pair_key],
                "region_usage": region_usage_counter[tar_lab],
                "target_area": float(target.get("area_px", 0)),
                "pair_key": pair_key,
            })

    if not candidates:
        return None

    same_side_candidates = [c for c in candidates if c["same_side"]]
    pool = same_side_candidates if same_side_candidates else candidates

    chosen = sorted(
        pool,
        key=lambda x: (
            x["usage"],
            x["region_usage"],
            -x["target_area"],
        )
    )[0]

    return chosen["ref"], chosen["target"], chosen["direction"]


def save_region_round_mask(slice_manifest: dict, region_label: int, out_path: Path) -> str:
    region_rel = slice_manifest["brainparc"]["region"]
    region_arr = read_label_png(region_rel)
    mask = region_arr.astype(np.int64) == int(region_label)
    return save_binary_mask(mask, out_path)


def save_tissue_round_mask(slice_manifest: dict, tissue_label: int, out_path: Path) -> str:
    tissue_rel = slice_manifest["brainparc"]["tissue"]
    tissue_arr = read_label_png(tissue_rel)
    mask = tissue_arr.astype(np.int64) == int(tissue_label)
    return save_binary_mask(mask, out_path)


def save_tumor_union_mask(
    slice_manifest: dict,
    tumor_labels: Tuple[int, ...],
    out_path: Path,
) -> str:
    """
    保存 BraTS 肿瘤区域 union mask。
    默认 tumor_labels=(1, 2, 3)，不包含 RC。
    """
    brats_seg = slice_manifest.get("brats_seg")
    if brats_seg is None:
        raise ValueError("No brats_seg found in slice manifest.")

    seg_rel = brats_seg.get("seg_label")
    if not seg_rel:
        raise ValueError("No seg_label path found in brats_seg.")

    seg_arr = read_label_png(seg_rel).astype(np.int64)
    mask = np.isin(seg_arr, list(tumor_labels))
    return save_binary_mask(mask, out_path)


def tumor_union_area_for_slice(
    slice_manifest: dict,
    tumor_labels: Tuple[int, ...],
) -> int:
    brats_seg = slice_manifest.get("brats_seg")
    if brats_seg is None:
        return 0

    seg_rel = brats_seg.get("seg_label")
    if not seg_rel:
        return 0

    seg_arr = read_label_png(seg_rel).astype(np.int64)
    return int(np.isin(seg_arr, list(tumor_labels)).sum())


def choose_tumor_overlapping_region_for_slice(
    slice_manifest: dict,
    fine_region_rows: pd.DataFrame,
    region_usage_counter: Counter,
) -> Optional[Tuple[dict, dict]]:
    """
    选择当前 slice 上与 BraTS 肿瘤区域重叠面积最大的 BrainParc fine anatomical region。

    返回：
        target_region: 目标 BrainParc 脑区 row
        overlap_info: overlap 统计信息

    注意：
    - 只在 fine_region_rows 里选，所以不会把 CSF / Cerebral_WM_L/R 当成目标脑区；
    - 默认 tumor union = BraTS label 1/2/3，不包含 RC；
    - 必须满足最小肿瘤面积、最小重叠面积、最小肿瘤覆盖比例。
    """
    brats_seg = slice_manifest.get("brats_seg")
    if brats_seg is None:
        return None

    seg_rel = brats_seg.get("seg_label")
    if not seg_rel:
        return None

    seg_arr = read_label_png(seg_rel).astype(np.int64)
    tumor_mask = np.isin(seg_arr, list(TUMOR_UNION_LABELS))
    tumor_area = int(tumor_mask.sum())

    if tumor_area < MIN_TUMOR_AREA_PX:
        return None

    region_rel = slice_manifest["brainparc"]["region"]
    region_arr = read_label_png(region_rel).astype(np.int64)

    candidates = []

    for _, rr in fine_region_rows.iterrows():
        region_label = int(rr["region_label"])
        region_mask = region_arr == region_label
        region_area = int(region_mask.sum())

        if region_area < MIN_REGION_AREA_PX:
            continue

        overlap_px = int((tumor_mask & region_mask).sum())
        if overlap_px < MIN_TUMOR_REGION_OVERLAP_PX:
            continue

        overlap_frac_of_tumor = float(overlap_px / max(tumor_area, 1))
        overlap_frac_of_region = float(overlap_px / max(region_area, 1))

        if overlap_frac_of_tumor < MIN_OVERLAP_FRAC_OF_TUMOR:
            continue

        candidates.append({
            "region_row": rr.to_dict(),
            "overlap_px": overlap_px,
            "tumor_area_px": tumor_area,
            "region_area_px": region_area,
            "overlap_frac_of_tumor": overlap_frac_of_tumor,
            "overlap_frac_of_region": overlap_frac_of_region,
            "region_usage": region_usage_counter[region_label],
        })

    if not candidates:
        return None

    chosen = sorted(
        candidates,
        key=lambda x: (
            -x["overlap_px"],
            -x["overlap_frac_of_tumor"],
            x["region_usage"],
        )
    )[0]

    target_region = chosen["region_row"]
    overlap_info = {
        "tumor_labels": list(TUMOR_UNION_LABELS),
        "tumor_name_zh": TUMOR_UNION_NAME_ZH,
        "tumor_area_px": int(chosen["tumor_area_px"]),
        "region_area_px": int(chosen["region_area_px"]),
        "overlap_px": int(chosen["overlap_px"]),
        "overlap_frac_of_tumor": float(chosen["overlap_frac_of_tumor"]),
        "overlap_frac_of_region": float(chosen["overlap_frac_of_region"]),
    }

    return target_region, overlap_info


def make_round(
    round_id: int,
    question: str,
    target_label: int,
    target_name: str,
    target_mask: str,
    target_type: str,
    extra: Optional[dict] = None,
) -> dict:
    d = {
        "round_id": int(round_id),
        "question": question,
        "answer": f"好的，{SEG_TOKEN}，记为实例{round_id}。",
        "target_label": int(target_label),
        "target_name": str(target_name),
        "target_type": target_type,
        "target_mask": target_mask,
    }
    if extra:
        d.update(extra)
    return d


def dialogue_dedup_key(
    case_id: Any,
    conversation_type: str,
    round_targets: List[Tuple[str, str, Optional[str]]],
) -> bytes:
    """
    计算对话去重签名（16 字节 blake2b 摘要）。

    去重定义：同一 case、同一对话类型、各轮目标结构序列完全相同
    （含 spatial 方向）的对话视为“同一任务”，跨切片折叠为一条。

    round_targets: [(target_type, target_name, spatial_relation_or_None), ...]

    注意：本函数与 4_remove_duplicate.py 中的同名逻辑必须保持一致，
    两边对“重复”的定义才会完全相同。
    """
    parts = []
    for (t_type, t_name, t_rel) in round_targets:
        parts.append(
            "|".join([str(t_type), str(t_name), "" if t_rel is None else str(t_rel)])
        )
    raw = "\x1e".join([str(case_id), str(conversation_type), "\x1f".join(parts)])
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).digest()


def make_region_round1_question(
    region_name: str,
    *keys: Any,
    template_usage_counter: Optional[Counter] = None,
) -> str:
    return render_template(
        "region_round1",
        {
            "region": pretty_region_name_zh(region_name),
        },
        region_name,
        *keys,
        usage_counter=template_usage_counter,
    )


def make_tissue_round1_question(
    tissue_label: int,
    *keys: Any,
    template_usage_counter: Optional[Counter] = None,
) -> str:
    return render_template(
        "tissue_round1",
        {
            "tissue": pretty_tissue_name_zh(tissue_label),
        },
        tissue_label,
        *keys,
        usage_counter=template_usage_counter,
    )


def choose_balanced_tissue(
    tissue_rows: pd.DataFrame,
    tissue_usage_counter: Counter,
    excluded_tissue_labels: Optional[set[int]] = None,
) -> Optional[dict]:
    excluded_tissue_labels = excluded_tissue_labels or set()
    candidates = []
    for _, row in tissue_rows.iterrows():
        tissue_label = int(row["tissue_label"])
        if tissue_label in excluded_tissue_labels:
            continue
        if not tissue_row_has_enough_area(row):
            continue
        candidates.append(row.to_dict())

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda r: (
            tissue_usage_counter[int(r["tissue_label"])],
            -float(r.get("area_px", 0)),
            int(r["tissue_label"]),
        ),
    )[0]


# ============================================================
# 5. 多轮数据集生成
# ============================================================

def build_multiround_dataset(manifest_rows: List[dict]) -> List[dict]:
    case_df, slice_df, region_df, tissue_df, corr_df = load_tables()
    region_to_tissue = get_region_to_tissue_map(corr_df)
    slice_manifest_map = get_slice_manifest_map(manifest_rows)

    # 分组
    region_by_slice = {sid: g.copy() for sid, g in region_df.groupby("slice_id")}
    tissue_by_slice = {sid: g.copy() for sid, g in tissue_df.groupby("slice_id")}

    tissue_usage_counter = Counter()
    region_usage_counter = Counter()
    pair_usage_counter = Counter()
    template_usage_counter = Counter()

    # 跨切片去重：记录已生成对话的签名，避免相邻切片产生完全相同的任务。
    seen_dedup_keys: set[bytes] = set()

    dialogues = []

    # 保持和 slice table 一致的顺序
    for _, srow in tqdm(slice_df.iterrows(), total=len(slice_df), desc="Building multiround dialogues"):
        slice_id = srow["slice_id"]
        case_id = srow["case_id"]
        plane = srow["plane"]
        if slice_id not in slice_manifest_map:
            continue

        slice_manifest = slice_manifest_map[slice_id]
        images_dict = get_manifest_images(slice_manifest)
        image_path = get_primary_image(slice_manifest) or srow.get("t1n_path", "")
        modalities = get_manifest_modalities(slice_manifest)
        main_modality = str(slice_manifest.get("main_modality", "t1n"))
        region_rows = region_by_slice.get(slice_id)
        tissue_rows = tissue_by_slice.get(slice_id)

        if region_rows is None or tissue_rows is None:
            continue

        fine_rows = region_rows[
            (region_rows["region_category"] == "fine_anatomical_region")
            & (region_rows["side"].isin(["left", "right"]))
        ].copy()

        # 为了可重复，按面积和 label 排序
        fine_rows = fine_rows.sort_values(["area_px", "region_label"], ascending=[False, True])
        tissue_rows = tissue_rows.sort_values(["area_px", "tissue_label"], ascending=[False, True])

        conv_idx = 1
        slice_dialogue_count = 0
        slice_used_target_labels: set[int] = set()
        slice_used_pair_keys: set[tuple] = set()
        slice_used_tissue_labels: set[int] = set()

        # ----------------------------------------------------
        # A. basic_segmentation
        #    单轮直接分割，专门用于提升基础 image-to-mask 能力
        # ----------------------------------------------------
        if ENABLE_BASIC_SEGMENTATION:
            basic_plan = ["region", "tissue", "region", "tumor", "tissue", "region"]
            basic_dialogue_count = 0
            for target_kind in basic_plan:
                if basic_dialogue_count >= MAX_BASIC_SEGMENTATION_PER_SLICE:
                    break
                if slice_dialogue_count >= MAX_DIALOGUES_PER_SLICE_TOTAL:
                    break

                conv_id = f"{slice_id}_C{conv_idx:03d}_basic_segmentation_{target_kind}"
                case_mask_dir = MULTIROUND_MASK_DIR / case_id

                if target_kind == "region":
                    region_row = choose_balanced_region(
                        fine_rows.to_dict("records"),
                        region_usage_counter,
                        excluded_region_labels=slice_used_target_labels,
                    )
                    if region_row is None:
                        continue

                    region_label = int(region_row["region_label"])
                    dkey = dialogue_dedup_key(
                        case_id,
                        "basic_segmentation",
                        [("brainparc_region", region_row["region_name"], None)],
                    )
                    if dkey in seen_dedup_keys:
                        slice_used_target_labels.add(region_label)
                        continue
                    mask = save_region_round_mask(
                        slice_manifest,
                        region_label,
                        case_mask_dir / f"{conv_id}_round01_region_{region_label}.png",
                    )
                    question = render_template(
                        "basic_region_segmentation",
                        {
                            "region": pretty_region_name_zh(region_row["region_name"]),
                        },
                        conv_id,
                        region_label,
                        usage_counter=template_usage_counter,
                    )
                    round_data = make_round(
                        1,
                        question,
                        region_label,
                        str(region_row["region_name"]),
                        mask,
                        "brainparc_region",
                        {
                            "display_name_zh": pretty_region_name_zh(region_row["region_name"]),
                            "segmentation_mode": "direct_single_round",
                        },
                    )
                    region_usage_counter[region_label] += 1
                    slice_used_target_labels.add(region_label)

                elif target_kind == "tissue":
                    tissue_row = choose_balanced_tissue(
                        tissue_rows,
                        tissue_usage_counter,
                        excluded_tissue_labels=slice_used_tissue_labels,
                    )
                    if tissue_row is None:
                        continue

                    tissue_label = int(tissue_row["tissue_label"])
                    dkey = dialogue_dedup_key(
                        case_id,
                        "basic_segmentation",
                        [("brainparc_tissue", pretty_tissue_name_zh(tissue_label), None)],
                    )
                    if dkey in seen_dedup_keys:
                        slice_used_tissue_labels.add(tissue_label)
                        continue
                    mask = save_tissue_round_mask(
                        slice_manifest,
                        tissue_label,
                        case_mask_dir / f"{conv_id}_round01_tissue_{tissue_label}.png",
                    )
                    question = render_template(
                        "basic_tissue_segmentation",
                        {
                            "tissue": pretty_tissue_name_zh(tissue_label),
                        },
                        conv_id,
                        tissue_label,
                        usage_counter=template_usage_counter,
                    )
                    round_data = make_round(
                        1,
                        question,
                        tissue_label,
                        pretty_tissue_name_zh(tissue_label),
                        mask,
                        "brainparc_tissue",
                        {
                            "tissue_name_en": pretty_tissue_name_en(tissue_label),
                            "segmentation_mode": "direct_single_round",
                        },
                    )
                    tissue_usage_counter[tissue_label] += 1
                    slice_used_tissue_labels.add(tissue_label)

                else:
                    tumor_area = tumor_union_area_for_slice(slice_manifest, TUMOR_UNION_LABELS)
                    if tumor_area < MIN_TUMOR_AREA_PX:
                        continue

                    dkey = dialogue_dedup_key(
                        case_id,
                        "basic_segmentation",
                        [("brats_tumor_union", TUMOR_UNION_NAME_ZH, None)],
                    )
                    if dkey in seen_dedup_keys:
                        continue

                    mask = save_tumor_union_mask(
                        slice_manifest,
                        TUMOR_UNION_LABELS,
                        case_mask_dir / f"{conv_id}_round01_tumor_union.png",
                    )
                    question = render_template(
                        "basic_tumor_segmentation",
                        {
                            "tumor": TUMOR_UNION_NAME_ZH,
                        },
                        conv_id,
                        usage_counter=template_usage_counter,
                    )
                    round_data = make_round(
                        1,
                        question,
                        -1,
                        TUMOR_UNION_NAME_ZH,
                        mask,
                        "brats_tumor_union",
                        {
                            "tumor_labels": list(TUMOR_UNION_LABELS),
                            "tumor_area_px": tumor_area,
                            "segmentation_mode": "direct_single_round",
                        },
                    )

                dialogues.append({
                    "conversation_id": conv_id,
                    "conversation_type": "basic_segmentation",
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "plane": plane,
                    "image": image_path,
                    "images": images_dict,
                    "main_modality": main_modality,
                    "modalities": modalities,
                    "rounds": [round_data],
                })

                seen_dedup_keys.add(dkey)
                slice_dialogue_count += 1
                basic_dialogue_count += 1
                conv_idx += 1

        if len(fine_rows) < 2:
            continue

        # ----------------------------------------------------
        # B. tissue_to_region
        # ----------------------------------------------------
        for _ in range(MAX_DIALOGUES_PER_SLICE_PER_TYPE):
            if slice_dialogue_count >= MAX_DIALOGUES_PER_SLICE_TOTAL:
                break

            tissue_row, region_row = choose_balanced_tissue_for_slice(
                tissue_rows=tissue_rows,
                fine_region_rows=fine_rows,
                region_to_tissue_map=region_to_tissue,
                tissue_usage_counter=tissue_usage_counter,
                region_usage_counter=region_usage_counter,
                excluded_tissue_labels=slice_used_tissue_labels,
                excluded_region_labels=slice_used_target_labels,
            )

            if tissue_row is None or region_row is None:
                break

            tissue_label = int(tissue_row["tissue_label"])
            region_label = int(region_row["region_label"])

            dkey = dialogue_dedup_key(
                case_id,
                "tissue_to_region",
                [
                    ("brainparc_tissue", pretty_tissue_name_zh(tissue_label), None),
                    ("brainparc_region", region_row["region_name"], None),
                ],
            )
            if dkey in seen_dedup_keys:
                slice_used_target_labels.add(region_label)
                continue

            conv_id = f"{slice_id}_C{conv_idx:03d}_tissue_to_region"
            case_mask_dir = MULTIROUND_MASK_DIR / case_id

            mask1 = save_tissue_round_mask(
                slice_manifest,
                tissue_label,
                case_mask_dir / f"{conv_id}_round01_tissue_{tissue_label}.png",
            )
            mask2 = save_region_round_mask(
                slice_manifest,
                region_label,
                case_mask_dir / f"{conv_id}_round02_region_{region_label}.png",
            )

            q1 = make_tissue_round1_question(
                tissue_label,
                conv_id,
                template_usage_counter=template_usage_counter,
            )
            q2 = render_template(
                "tissue_to_region",
                {
                    "region": pretty_region_name_zh(region_row["region_name"]),
                },
                conv_id,
                region_label,
                usage_counter=template_usage_counter,
            )

            dialogue = {
                "conversation_id": conv_id,
                "conversation_type": "tissue_to_region",
                "case_id": case_id,
                "slice_id": slice_id,
                "plane": plane,
                "image": image_path,
                "images": images_dict,
                "main_modality": main_modality,
                "modalities": modalities,
                "rounds": [
                    make_round(
                        1,
                        q1,
                        tissue_label,
                        pretty_tissue_name_zh(tissue_label),
                        mask1,
                        "brainparc_tissue",
                        {
                            "tissue_name_en": pretty_tissue_name_en(tissue_label),
                        },
                    ),
                    make_round(
                        2,
                        q2,
                        region_label,
                        str(region_row["region_name"]),
                        mask2,
                        "brainparc_region",
                        {
                            "ref_round_id": 1,
                            "relation_type": "tissue_contains_or_associated_region",
                            "display_name_zh": pretty_region_name_zh(region_row["region_name"]),
                        },
                    ),
                ],
            }
            dialogues.append(dialogue)

            seen_dedup_keys.add(dkey)
            tissue_usage_counter[tissue_label] += 1
            region_usage_counter[region_label] += 1
            slice_used_tissue_labels.add(tissue_label)
            slice_used_target_labels.add(region_label)
            slice_dialogue_count += 1
            conv_idx += 1

        # ----------------------------------------------------
        # C. contralateral_same_region
        # ----------------------------------------------------
        for _ in range(MAX_DIALOGUES_PER_SLICE_PER_TYPE):
            if slice_dialogue_count >= MAX_DIALOGUES_PER_SLICE_TOTAL:
                break

            pair = choose_contralateral_pair_for_slice(
                fine_rows,
                region_usage_counter,
                pair_usage_counter,
                excluded_pair_keys=slice_used_pair_keys,
                excluded_target_labels=slice_used_target_labels,
            )
            if pair is None:
                break

            ref, target = pair
            ref_label = int(ref["region_label"])
            target_label = int(target["region_label"])

            pair_key = (ref_label, target_label, "contra")
            dkey = dialogue_dedup_key(
                case_id,
                "contralateral_same_region",
                [
                    ("brainparc_region", ref["region_name"], None),
                    ("brainparc_region", target["region_name"], None),
                ],
            )
            if dkey in seen_dedup_keys:
                slice_used_pair_keys.add(pair_key)
                slice_used_target_labels.add(target_label)
                continue

            conv_id = f"{slice_id}_C{conv_idx:03d}_contralateral_same_region"
            case_mask_dir = MULTIROUND_MASK_DIR / case_id

            mask1 = save_region_round_mask(
                slice_manifest,
                ref_label,
                case_mask_dir / f"{conv_id}_round01_region_{ref_label}.png",
            )
            mask2 = save_region_round_mask(
                slice_manifest,
                target_label,
                case_mask_dir / f"{conv_id}_round02_region_{target_label}.png",
            )

            q1 = make_region_round1_question(
                ref["region_name"],
                conv_id,
                template_usage_counter=template_usage_counter,
            )
            q2 = render_template(
                "contralateral_same_region",
                {
                    "region": pretty_region_name_zh(target["region_name"]),
                },
                conv_id,
                target_label,
                usage_counter=template_usage_counter,
            )

            dialogue = {
                "conversation_id": conv_id,
                "conversation_type": "contralateral_same_region",
                "case_id": case_id,
                "slice_id": slice_id,
                "plane": plane,
                "image": image_path,
                "images": images_dict,
                "main_modality": main_modality,
                "modalities": modalities,
                "rounds": [
                    make_round(
                        1,
                        q1,
                        ref_label,
                        str(ref["region_name"]),
                        mask1,
                        "brainparc_region",
                        {
                            "display_name_zh": pretty_region_name_zh(ref["region_name"]),
                        },
                    ),
                    make_round(
                        2,
                        q2,
                        target_label,
                        str(target["region_name"]),
                        mask2,
                        "brainparc_region",
                        {
                            "ref_round_id": 1,
                            "relation_type": "contralateral_same_anatomical_name",
                            "display_name_zh": pretty_region_name_zh(target["region_name"]),
                        },
                    ),
                ],
            }
            dialogues.append(dialogue)

            seen_dedup_keys.add(dkey)
            pair_usage_counter[pair_key] += 1
            region_usage_counter[ref_label] += 1
            region_usage_counter[target_label] += 1
            slice_used_pair_keys.add(pair_key)
            slice_used_target_labels.add(target_label)
            slice_dialogue_count += 1
            conv_idx += 1

        # ----------------------------------------------------
        # D. same_side_same_lobe
        # ----------------------------------------------------
        for _ in range(MAX_DIALOGUES_PER_SLICE_PER_TYPE):
            if slice_dialogue_count >= MAX_DIALOGUES_PER_SLICE_TOTAL:
                break

            pair = choose_same_side_same_lobe_pair_for_slice(
                fine_rows,
                region_usage_counter,
                pair_usage_counter,
                excluded_pair_keys=slice_used_pair_keys,
                excluded_target_labels=slice_used_target_labels,
            )
            if pair is None:
                break

            ref, target = pair
            ref_label = int(ref["region_label"])
            target_label = int(target["region_label"])
            side_zh = "左侧" if ref.get("side") == "left" else "右侧"
            scope_phrase = relation_scope_phrase_by_lobe(ref.get("lobe"), side_zh)

            pair_key = (ref_label, target_label, "same_side_same_lobe")
            dkey = dialogue_dedup_key(
                case_id,
                "same_side_same_lobe",
                [
                    ("brainparc_region", ref["region_name"], None),
                    ("brainparc_region", target["region_name"], None),
                ],
            )
            if dkey in seen_dedup_keys:
                slice_used_pair_keys.add(pair_key)
                slice_used_target_labels.add(target_label)
                continue

            conv_id = f"{slice_id}_C{conv_idx:03d}_same_side_same_lobe"
            case_mask_dir = MULTIROUND_MASK_DIR / case_id

            mask1 = save_region_round_mask(
                slice_manifest,
                ref_label,
                case_mask_dir / f"{conv_id}_round01_region_{ref_label}.png",
            )
            mask2 = save_region_round_mask(
                slice_manifest,
                target_label,
                case_mask_dir / f"{conv_id}_round02_region_{target_label}.png",
            )

            q1 = make_region_round1_question(
                ref["region_name"],
                conv_id,
                template_usage_counter=template_usage_counter,
            )
            q2 = render_template(
                "same_side_same_lobe",
                {
                    "scope_phrase": scope_phrase,
                    "region": pretty_region_name_zh(target["region_name"]),
                },
                conv_id,
                target_label,
                usage_counter=template_usage_counter,
            )

            dialogue = {
                "conversation_id": conv_id,
                "conversation_type": "same_side_same_lobe",
                "case_id": case_id,
                "slice_id": slice_id,
                "plane": plane,
                "image": image_path,
                "images": images_dict,
                "main_modality": main_modality,
                "modalities": modalities,
                "rounds": [
                    make_round(
                        1,
                        q1,
                        ref_label,
                        str(ref["region_name"]),
                        mask1,
                        "brainparc_region",
                        {
                            "display_name_zh": pretty_region_name_zh(ref["region_name"]),
                            "side": str(ref.get("side")),
                            "lobe": str(ref.get("lobe")),
                        },
                    ),
                    make_round(
                        2,
                        q2,
                        target_label,
                        str(target["region_name"]),
                        mask2,
                        "brainparc_region",
                        {
                            "ref_round_id": 1,
                            "relation_type": "same_side_same_lobe",
                            "display_name_zh": pretty_region_name_zh(target["region_name"]),
                            "side": str(target.get("side")),
                            "lobe": str(target.get("lobe")),
                        },
                    ),
                ],
            }
            dialogues.append(dialogue)

            seen_dedup_keys.add(dkey)
            pair_usage_counter[pair_key] += 1
            region_usage_counter[ref_label] += 1
            region_usage_counter[target_label] += 1
            slice_used_pair_keys.add(pair_key)
            slice_used_target_labels.add(target_label)
            slice_dialogue_count += 1
            conv_idx += 1

        # ----------------------------------------------------
        # E. spatial_named_region
        # ----------------------------------------------------
        for _ in range(MAX_DIALOGUES_PER_SLICE_PER_TYPE):
            if slice_dialogue_count >= MAX_DIALOGUES_PER_SLICE_TOTAL:
                break

            spatial = choose_spatial_pair_for_slice(
                fine_rows,
                region_usage_counter,
                pair_usage_counter,
                excluded_pair_keys=slice_used_pair_keys,
                excluded_target_labels=slice_used_target_labels,
            )
            if spatial is None:
                break

            ref, target, direction = spatial
            ref_label = int(ref["region_label"])
            target_label = int(target["region_label"])
            direction_zh = DIRECTION_ZH[direction]

            pair_key = (ref_label, target_label, direction)
            dkey = dialogue_dedup_key(
                case_id,
                "spatial_named_region",
                [
                    ("brainparc_region", ref["region_name"], None),
                    ("brainparc_region", target["region_name"], direction),
                ],
            )
            if dkey in seen_dedup_keys:
                slice_used_pair_keys.add(pair_key)
                slice_used_target_labels.add(target_label)
                continue

            conv_id = f"{slice_id}_C{conv_idx:03d}_spatial_named_region"
            case_mask_dir = MULTIROUND_MASK_DIR / case_id

            mask1 = save_region_round_mask(
                slice_manifest,
                ref_label,
                case_mask_dir / f"{conv_id}_round01_region_{ref_label}.png",
            )
            mask2 = save_region_round_mask(
                slice_manifest,
                target_label,
                case_mask_dir / f"{conv_id}_round02_region_{target_label}.png",
            )

            q1 = make_region_round1_question(
                ref["region_name"],
                conv_id,
                template_usage_counter=template_usage_counter,
            )
            assert direction in DIRECTION_ZH, f"Invalid direction: {direction}"

            q2 = render_template(
                "spatial_named_region",
                {
                    "direction_zh": direction_zh,
                    "region": pretty_region_name_zh(target["region_name"]),
                },
                conv_id,
                target_label,
                direction,
                usage_counter=template_usage_counter,
            )

            dialogue = {
                "conversation_id": conv_id,
                "conversation_type": "spatial_named_region",
                "case_id": case_id,
                "slice_id": slice_id,
                "plane": plane,
                "image": image_path,
                "images": images_dict,
                "main_modality": main_modality,
                "modalities": modalities,
                "rounds": [
                    make_round(
                        1,
                        q1,
                        ref_label,
                        str(ref["region_name"]),
                        mask1,
                        "brainparc_region",
                        {
                            "display_name_zh": pretty_region_name_zh(ref["region_name"]),
                            "centroid_norm_xy": ref.get("centroid_norm_xy"),
                        },
                    ),
                    make_round(
                        2,
                        q2,
                        target_label,
                        str(target["region_name"]),
                        mask2,
                        "brainparc_region",
                        {
                            "ref_round_id": 1,
                            "relation_type": "image_spatial_relation",
                            "spatial_relation": direction,
                            "spatial_relation_zh": direction_zh,
                            "display_name_zh": pretty_region_name_zh(target["region_name"]),
                            "centroid_norm_xy": target.get("centroid_norm_xy"),
                        },
                    ),
                ],
            }
            dialogues.append(dialogue)

            seen_dedup_keys.add(dkey)
            pair_usage_counter[pair_key] += 1
            region_usage_counter[ref_label] += 1
            region_usage_counter[target_label] += 1
            slice_used_pair_keys.add(pair_key)
            slice_used_target_labels.add(target_label)
            slice_dialogue_count += 1
            conv_idx += 1


        # ----------------------------------------------------
        # F. tumor_to_overlapping_region
        #    BraTS 肿瘤区域 -> 与其空间重叠面积最大的 BrainParc 细脑区
        # ----------------------------------------------------
        if ENABLE_TUMOR_TO_OVERLAPPING_REGION and slice_dialogue_count < MAX_DIALOGUES_PER_SLICE_TOTAL:
            tumor_overlap = choose_tumor_overlapping_region_for_slice(
                slice_manifest=slice_manifest,
                fine_region_rows=fine_rows,
                region_usage_counter=region_usage_counter,
            )

            tumor_dkey = None
            if tumor_overlap is not None:
                target_region, overlap_info = tumor_overlap
                target_label = int(target_region["region_label"])
                tumor_dkey = dialogue_dedup_key(
                    case_id,
                    "tumor_to_overlapping_region",
                    [
                        ("brats_tumor_union", TUMOR_UNION_NAME_ZH, None),
                        ("brainparc_region", target_region["region_name"], None),
                    ],
                )

            if tumor_overlap is not None and tumor_dkey not in seen_dedup_keys:
                conv_id = f"{slice_id}_C{conv_idx:03d}_tumor_to_overlapping_region"
                case_mask_dir = MULTIROUND_MASK_DIR / case_id

                mask1 = save_tumor_union_mask(
                    slice_manifest,
                    TUMOR_UNION_LABELS,
                    case_mask_dir / f"{conv_id}_round01_tumor_union.png",
                )

                mask2 = save_region_round_mask(
                    slice_manifest,
                    target_label,
                    case_mask_dir / f"{conv_id}_round02_region_{target_label}.png",
                )

                q1 = render_template(
                    "tumor_round1",
                    {
                        "tumor": TUMOR_UNION_NAME_ZH,
                    },
                    conv_id,
                    usage_counter=template_usage_counter,
                )
                q2 = render_template(
                    "tumor_to_overlapping_region",
                    {
                        "region": pretty_region_name_zh(target_region["region_name"]),
                    },
                    conv_id,
                    target_label,
                    usage_counter=template_usage_counter,
                )

                dialogue = {
                    "conversation_id": conv_id,
                    "conversation_type": "tumor_to_overlapping_region",
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "plane": plane,
                    "image": image_path,
                    "images": images_dict,
                    "main_modality": main_modality,
                    "modalities": modalities,
                    "rounds": [
                        make_round(
                            1,
                            q1,
                            -1,
                            TUMOR_UNION_NAME_ZH,
                            mask1,
                            "brats_tumor_union",
                            {
                                "tumor_labels": list(TUMOR_UNION_LABELS),
                                "relation_type": "tumor_region",
                            },
                        ),
                        make_round(
                            2,
                            q2,
                            target_label,
                            str(target_region["region_name"]),
                            mask2,
                            "brainparc_region",
                            {
                                "ref_round_id": 1,
                                "relation_type": "largest_spatial_overlap_with_tumor",
                                "display_name_zh": pretty_region_name_zh(target_region["region_name"]),
                                "overlap_info": overlap_info,
                            },
                        ),
                    ],
                }

                dialogues.append(dialogue)
                seen_dedup_keys.add(tumor_dkey)
                region_usage_counter[target_label] += 1
                slice_dialogue_count += 1
                conv_idx += 1

    build_multiround_dataset.last_template_usage_summary = summarize_template_usage(template_usage_counter)
    return dialogues


# ============================================================
# 6. 主流程与检查
# ============================================================

def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not BRAINPARC_ROOT.exists():
        raise FileNotFoundError(f"BrainParc root not found: {BRAINPARC_ROOT}")

    if CLEAN_OLD_OUTPUTS:
        if TABLE_DIR.exists():
            shutil.rmtree(TABLE_DIR)
        if MULTIROUND_DIR.exists():
            shutil.rmtree(MULTIROUND_DIR)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MULTIROUND_MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"PROCESSED_ROOT: {PROCESSED_ROOT}")
    print(f"MANIFEST_PATH : {MANIFEST_PATH}")
    print(f"BRAINPARC_ROOT: {BRAINPARC_ROOT}")
    print(f"TABLE_DIR     : {TABLE_DIR}")
    print(f"MULTIROUND_DIR: {MULTIROUND_DIR}")
    print("=" * 100)

    manifest_rows = load_jsonl(MANIFEST_PATH)
    print(f"[INFO] manifest rows: {len(manifest_rows)}")

    case_ids = sorted({d["case"] for d in manifest_rows})
    label_map = load_global_label_map(case_ids)
    print(f"[INFO] case n: {len(case_ids)}")
    print(f"[INFO] global BrainParc label map n: {len(label_map)}")

    # 1. case/slice
    case_df, slice_df = build_case_and_slice_tables(manifest_rows)
    case_df.to_csv(TABLE_DIR / "case_level_table.csv", index=False, encoding="utf-8-sig")
    slice_df.to_csv(TABLE_DIR / "slice_level_table.csv", index=False, encoding="utf-8-sig")

    # 2. visible region/tissue
    region_df, tissue_df = build_visible_tables(manifest_rows, label_map)
    region_df.to_csv(TABLE_DIR / "region_visible_table.csv", index=False, encoding="utf-8-sig")
    tissue_df.to_csv(TABLE_DIR / "tissue_visible_table.csv", index=False, encoding="utf-8-sig")

    # 3. region-tissue correspondence
    corr_df = build_region_tissue_correspondence(case_df, label_map)
    corr_df.to_csv(TABLE_DIR / "region_tissue_correspondence_table.csv", index=False, encoding="utf-8-sig")

    # 4. multiround dataset
    dialogues = build_multiround_dataset(manifest_rows)
    write_jsonl(MULTIROUND_JSONL, dialogues)
    template_usage_summary = getattr(
        build_multiround_dataset,
        "last_template_usage_summary",
        summarize_template_usage(Counter()),
    )

    # 5. summary
    conv_counter = Counter([d["conversation_type"] for d in dialogues])
    target_counter = Counter()
    tissue_counter = Counter()

    for d in dialogues:
        if d["conversation_type"] == "tissue_to_region":
            tissue_counter[d["rounds"][0]["target_name"]] += 1
        for rd in d["rounds"]:
            target_counter[rd["target_name"]] += 1

    summary = {
        "processed_root": str(PROCESSED_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "n_cases": int(case_df.shape[0]),
        "n_slices": int(slice_df.shape[0]),
        "n_region_visible_rows": int(region_df.shape[0]),
        "n_tissue_visible_rows": int(tissue_df.shape[0]),
        "n_region_tissue_correspondence_rows": int(corr_df.shape[0]),
        "n_dialogues": int(len(dialogues)),
        "conversation_type_counts": dict(conv_counter),
        "dialogue_category_order": list(DIALOGUE_CATEGORY_ORDER),
        "dialogue_category_descriptions": DIALOGUE_CATEGORY_DESCRIPTIONS,
        "template_pool_sizes": TEMPLATE_POOL_SIZES,
        "template_usage_summary": template_usage_summary,
        "seg_token": SEG_TOKEN,
        "enable_basic_segmentation": ENABLE_BASIC_SEGMENTATION,
        "max_basic_segmentation_per_slice": MAX_BASIC_SEGMENTATION_PER_SLICE,
        "max_dialogues_per_slice_total": MAX_DIALOGUES_PER_SLICE_TOTAL,
        "min_round_region_area_px": MIN_REGION_AREA_PX,
        "min_round_tissue_area_px": MIN_TISSUE_AREA_PX,
        "min_round_spatial_region_area_px": MIN_SPATIAL_REGION_AREA_PX,
        "tissue_target_counts_in_tissue_to_region": dict(tissue_counter),
        "top_30_targets": target_counter.most_common(30),
        "tissue_label_name_zh": TISSUE_LABEL_NAME_ZH,
        "tissue_label_name_en": TISSUE_LABEL_NAME_EN,
        "enable_tumor_to_overlapping_region": ENABLE_TUMOR_TO_OVERLAPPING_REGION,
        "tumor_union_labels": list(TUMOR_UNION_LABELS),
        "tumor_union_name_zh": TUMOR_UNION_NAME_ZH,
        "min_tumor_area_px": MIN_TUMOR_AREA_PX,
        "min_tumor_region_overlap_px": MIN_TUMOR_REGION_OVERLAP_PX,
        "min_overlap_frac_of_tumor": MIN_OVERLAP_FRAC_OF_TUMOR,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TABLES ===")
    for name in [
        "case_level_table.csv",
        "slice_level_table.csv",
        "region_visible_table.csv",
        "tissue_visible_table.csv",
        "region_tissue_correspondence_table.csv",
    ]:
        p = TABLE_DIR / name
        df = pd.read_csv(p)
        print(f"{name}: {df.shape}")

    print("\n=== MULTIROUND DATASET ===")
    print(f"jsonl: {MULTIROUND_JSONL}")
    print(f"dialogues: {len(dialogues)}")
    print(f"conversation types: {dict(conv_counter)}")
    print(f"tissue targets in tissue_to_region: {dict(tissue_counter)}")
    print(f"summary: {SUMMARY_JSON}")

    print("\n=== QUICK CHECK: side/category ===")
    print(region_df["side"].value_counts(dropna=False))
    print(region_df["region_category"].value_counts(dropna=False))

    print("\n=== QUICK CHECK: first 3 dialogues ===")
    for d in dialogues[:3]:
        print("-" * 100)
        print("conversation_id:", d["conversation_id"])
        print("type:", d["conversation_type"])
        for rd in d["rounds"]:
            print("Q:", rd["question"])
            print("target:", rd["target_name"])
            print("mask:", rd["target_mask"])

    print("\nDONE.")


if __name__ == "__main__":
    main()
