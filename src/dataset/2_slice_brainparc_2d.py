
"""

- 输入 MRI: t1n, t1c, t2w, t2f
- BrainParc mask 仍来自 t1n 处理结果：tissue.nii.gz 与 dk-struct.nii.gz
- 每例必切 axial，再从 coronal/sagittal 固定随机选一个
- test3 只切每个方向中间 20 张，现在是全量
"""

from __future__ import annotations

import csv
import json
import hashlib
import logging
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

from src.dataset import data_pathes

# ============================================================
# 0. 参数配置区：只改这里即可
# ============================================================

def get_data_path(name: str, fallback_name: str | None = None) -> Path:
    value = getattr(data_pathes, name, None)
    if value is None and fallback_name is not None:
        value = getattr(data_pathes, fallback_name, None)
    if value is None:
        names = name if fallback_name is None else f"{name} or {fallback_name}"
        raise AttributeError(f"Please define {names} in src/dataset/data_pathes.py")
    return Path(value)


# 输入根目录。可以是单个 instance/case 目录，也可以是包含多个 instance/case 的上级目录。
INPUT_ROOT = get_data_path("original_instance_dir", "instance_path")

# BrainParc 输出根目录。支持 OUT_ROOT/case_id 或 OUT_ROOT/.../case_id 两种布局。
BRAINPARC_ROOT = get_data_path("instance_seg_root_dir", "instance_seg_path")

# 2D 输出根目录。多 instance 运行时所有 case 会按 case_id 写入 images/masks/manifests 子目录。
# 每例固定切 axial，并在 coronal/sagittal 中随机再切一个，建议使用新的 out_root，避免被旧的 DONE.marker 跳过。
OUT_ROOT = get_data_path("instance_out_dir", "instance_out_path")

# 平面选择模式：
#   "axial_plus_random_coronal_or_sagittal"：每个病例必须切 axial，
#       然后在 coronal / sagittal 中固定随机选 1 个额外平面。
#   "random_one_per_case"：每个病例在 PLANES 中固定随机选 1 个平面。
#   "all"：每个病例三个平面都切。
#   "fixed"：只切 PLANES 里指定的平面，比如 ["axial"]。
PLANE_SELECTION_MODE = "axial_plus_random_coronal_or_sagittal"

# 候选平面。当前模式会固定包含 axial，并从 coronal / sagittal 中随机选一个。
PLANES = ["axial", "coronal", "sagittal"]

# 固定随机种子：同一个 case_id 每次都会选到同一个额外平面，保证可复现。
RANDOM_SEED = 20260427

# 调试时可设 20，只切中间 20 张；全量设 0
ONLY_CENTER_K = 0

# 切片采样步长。建议第一版 axial 全保留，所以设 1。
SLICE_STRIDE = 1

# 要导出的 MRI 图像模态。t1n 会始终作为 BrainParc 对齐参考。
# BraTS 常见模态：t1n, t1c, t2w, t2f；如果你的文件名使用 t2，也可以写 t2。
IMAGE_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# 如果 True，IMAGE_MODALITIES 中任一模态缺失都会报错；如果 False，只跳过缺失的非 t1n 模态。
REQUIRE_ALL_IMAGE_MODALITIES = True

# 空切片过滤。T1N 中非零比例低于 EMPTY_THRESHOLD 的切片会被跳过。
SKIP_EMPTY = True
EMPTY_THRESHOLD = 0.01

# 如果有 BraTS seg，是否保存肿瘤 seg；没有也不影响 BrainParc 切片。
SAVE_BRATS_SEG_IF_AVAILABLE = True
REQUIRE_BRATS_SEG = False

# 是否要求 BrainParc tissue 和 dk-struct 必须存在。
# 你现在是做 BrainParc-guided 数据集，建议 True。
REQUIRE_BRAINPARC_MASKS = True

# affine 不一致时是否严格报错。通常第一版设 False，只 warning；shape 不一致一定会跳过。
STRICT_AFFINE = False

# PNG 保存参数。0 最快，适合全量。
PNG_COMPRESS_LEVEL = 0

# MRI 强度归一化
USE_NONZERO_FOR_NORM = True
MRI_LOW_PERCENTILE = 1
MRI_HIGH_PERCENTILE = 99

# 每张 2D slice 上，一个 BrainParc label 至少多少像素才算可见。
MIN_REGION_AREA_PX = 20
MIN_TISSUE_AREA_PX = 20

# 断点续跑 marker
DONE_MARKER = "DONE.marker"


# 只处理 BrainParc 已经跑完的病例，避免遍历全量 TrainingData 后大量报错
ONLY_PROCESS_BRAINPARC_DONE_CASES = True

# 调试时限制处理病例数；None 表示不限制
MAX_CASES = None

# 如果只想切指定病例，在这里填 case_id；为空则不指定
CASE_IDS: list[str] = []


# ============================================================
# 1. 日志与配置
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("slice_4modal_brainparc_2d")


@dataclass(frozen=True)
class Config:
    input_root: Path = INPUT_ROOT
    brainparc_root: Path = BRAINPARC_ROOT
    out_root: Path = OUT_ROOT
    plane_selection_mode: str = PLANE_SELECTION_MODE
    planes: Tuple[str, ...] = tuple(PLANES)
    random_seed: int = RANDOM_SEED
    only_center_k: int = ONLY_CENTER_K
    slice_stride: int = SLICE_STRIDE
    image_modalities: Tuple[str, ...] = tuple(IMAGE_MODALITIES)
    require_all_image_modalities: bool = REQUIRE_ALL_IMAGE_MODALITIES
    skip_empty: bool = SKIP_EMPTY
    empty_threshold: float = EMPTY_THRESHOLD
    save_brats_seg_if_available: bool = SAVE_BRATS_SEG_IF_AVAILABLE
    require_brats_seg: bool = REQUIRE_BRATS_SEG
    require_brainparc_masks: bool = REQUIRE_BRAINPARC_MASKS
    strict_affine: bool = STRICT_AFFINE
    png_compress_level: int = PNG_COMPRESS_LEVEL
    use_nonzero_for_norm: bool = USE_NONZERO_FOR_NORM
    mri_low_percentile: int = MRI_LOW_PERCENTILE
    mri_high_percentile: int = MRI_HIGH_PERCENTILE
    min_region_area_px: int = MIN_REGION_AREA_PX
    min_tissue_area_px: int = MIN_TISSUE_AREA_PX
    done_marker: str = DONE_MARKER

    only_process_brainparc_done_cases: bool = ONLY_PROCESS_BRAINPARC_DONE_CASES
    max_cases: Optional[int] = MAX_CASES
    case_ids: Tuple[str, ...] = tuple(CASE_IDS)


# ============================================================
# 2. 文件查找与 NIfTI 读取
# ============================================================

def case_id_from_t1n(path: Path) -> str:
    suffix = "-t1n.nii.gz"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected T1N filename: {path}")
    return name[: -len(suffix)]


def find_t1n_files(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.rglob("*-t1n.nii.gz") if p.is_file()])


MODALITY_ALIASES = {
    "t1": ("t1n", "t1"),
    "t1n": ("t1n", "t1"),
    "t1c": ("t1c", "t1ce"),
    "t1ce": ("t1ce", "t1c"),
    "t2": ("t2", "t2w"),
    "t2w": ("t2w", "t2"),
    "t2f": ("t2f", "flair"),
    "flair": ("flair", "t2f"),
}


def normalize_modality_name(mod: str) -> str:
    return str(mod).strip().lower()


def find_modality_file(case_dir: Path, mod: str) -> Optional[Path]:
    files = list(case_dir.glob("*.nii")) + list(case_dir.glob("*.nii.gz"))
    mod = normalize_modality_name(mod)
    aliases = MODALITY_ALIASES.get(mod, (mod,))
    for alias in aliases:
        pattern = re.compile(rf"(^|[_\-]){re.escape(alias)}([_\-.]|$)")
        matched = [p for p in files if pattern.search(p.name.lower())]
        if matched:
            return sorted(matched)[0]
    return None


def find_brainparc_case_dir(brainparc_root: Path, case_id: str) -> Optional[Path]:
    direct = brainparc_root / case_id
    if direct.exists() and direct.is_dir():
        return direct

    nested_exact = sorted([p for p in brainparc_root.rglob(case_id) if p.is_dir()])
    if nested_exact:
        return nested_exact[0]

    candidates = sorted([p for p in brainparc_root.glob(f"*{case_id}*") if p.is_dir()])
    if candidates:
        return candidates[0]

    nested_candidates = sorted([p for p in brainparc_root.rglob(f"*{case_id}*") if p.is_dir()])
    return nested_candidates[0] if nested_candidates else None


def find_file_recursive(root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        candidates = sorted(root.rglob(name))
        if candidates:
            return candidates[0]
    return None


def brainparc_case_is_done(brainparc_root: Path, case_id: str) -> bool:
    bp_case_dir = find_brainparc_case_dir(brainparc_root, case_id)

    if bp_case_dir is None:
        return False

    required = [
        find_file_recursive(bp_case_dir, ["tissue.nii.gz", "tissue.nii"]),
        find_file_recursive(bp_case_dir, ["dk-struct.nii.gz", "dk-struct.nii"]),
        find_file_recursive(bp_case_dir, ["present_labels.csv"]),
    ]

    for p in required:
        if p is None:
            return False
        if not p.exists():
            return False
        if p.stat().st_size <= 0:
            return False

    return True


def select_t1n_files_for_run(t1n_files: list[Path], cfg: Config) -> list[Path]:
    """
    从 TrainingData 里筛选本次真正要切片的病例。

    目的：
    1. 测试阶段只切已经跑完 BrainParc 的病例；
    2. 避免 TrainingData 里大量还没跑 BrainParc 的病例报错；
    3. 支持 CASE_IDS 和 MAX_CASES 调试。
    """
    if cfg.case_ids:
        wanted = set(cfg.case_ids)
        found = {case_id_from_t1n(p) for p in t1n_files}
        missing = wanted - found
        if missing:
            raise RuntimeError(f"These CASE_IDS were not found under INPUT_ROOT: {sorted(missing)}")
        t1n_files = [
            p for p in t1n_files
            if case_id_from_t1n(p) in wanted
        ]

    if cfg.only_process_brainparc_done_cases:
        t1n_files = [
            p for p in t1n_files
            if brainparc_case_is_done(
                cfg.brainparc_root,
                case_id_from_t1n(p)
            )
        ]

    if cfg.max_cases is not None:
        t1n_files = t1n_files[: cfg.max_cases]

    return t1n_files


def load_img_canonical(path: Path) -> nib.Nifti1Image:
    img = nib.as_closest_canonical(nib.load(str(path)))
    axcodes = nib.aff2axcodes(img.affine)
    if axcodes != ("R", "A", "S"):
        raise ValueError(f"Failed to reorient image to RAS: {path}, axcodes={axcodes}")
    return img


def affine_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    return bool(np.allclose(a, b, atol=atol))


def load_volume_float(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = load_img_canonical(path)
    arr = img.get_fdata(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3:
        raise ValueError(f"Not a 3D volume: {path}")
    return arr, img.affine


def load_aligned_image_volume(path: Path, ref_shape: Tuple[int, int, int], ref_affine: np.ndarray, cfg: Config, name: str) -> np.ndarray:
    arr, affine = load_volume_float(path)

    if tuple(arr.shape) != tuple(ref_shape):
        raise ValueError(f"{name} shape mismatch: {arr.shape}, expected={ref_shape}, path={path}")

    if not affine_close(affine, ref_affine):
        msg = f"{name} affine differs from t1n affine: {path}"
        if cfg.strict_affine:
            raise ValueError(msg)
        logger.warning(msg)

    return arr


def load_label_volume(path: Path, ref_shape: Tuple[int, int, int], ref_affine: np.ndarray, cfg: Config, name: str) -> np.ndarray:
    img = load_img_canonical(path)
    arr = img.get_fdata(dtype=np.float32)
    arr = np.rint(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))

    if arr.ndim != 3:
        raise ValueError(f"{name} is not 3D: {path}")

    if arr.size == 0:
        raise ValueError(f"{name} contains no voxels: {path}")
    if float(arr.min()) < 0:
        raise ValueError(f"{name} contains negative labels: {path}")
    if float(arr.max()) > np.iinfo(np.uint16).max:
        raise ValueError(f"{name} contains labels larger than uint16: {path}")

    if tuple(arr.shape) != tuple(ref_shape):
        raise ValueError(f"{name} shape mismatch: {arr.shape}, expected={ref_shape}, path={path}")

    if not affine_close(img.affine, ref_affine):
        msg = f"{name} affine differs from t1n affine: {path}"
        if cfg.strict_affine:
            raise ValueError(msg)
        logger.warning(msg)

    return arr.astype(np.uint16)


# ============================================================
# 3. 图像预处理与保存
# ============================================================

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    mn = float(np.min(image))
    mx = float(np.max(image))
    if mx - mn <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mn) / (mx - mn)


def mri_min_max_preprocess(image: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.use_nonzero_for_norm:
        mask = image > 0
    else:
        mask = np.ones_like(image, dtype=bool)

    if not mask.any():
        return np.zeros_like(image, dtype=np.uint8)

    low, high = np.percentile(image[mask], [cfg.mri_low_percentile, cfg.mri_high_percentile])
    image = np.clip(image, low, high)
    image = normalize_minmax(image)
    return (image * 255.0).round().astype(np.uint8)


def save_png_u8(arr: np.ndarray, out_path: Path, compress_level: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(out_path, compress_level=compress_level)


def save_label_png(arr: np.ndarray, out_path: Path, compress_level: int) -> None:
    if arr is None:
        raise ValueError(f"Cannot save missing label array to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_val = int(np.max(arr)) if arr.size else 0
    if max_val <= 255:
        Image.fromarray(arr.astype(np.uint8)).save(out_path, compress_level=compress_level)
    else:
        Image.fromarray(arr.astype(np.uint16)).save(out_path, compress_level=compress_level)


# ============================================================
# 4. 切片方向统一
# ============================================================

def orient_slice_for_vlm(slice2d: np.ndarray, plane: str) -> np.ndarray:
    """
    输入为 canonical RAS 体数据中的 raw slice。
    输出为固定的 2D 显示方向：neurological view。

    axial:    左=患者左，右=患者右，上=患者前，下=患者后
    coronal:  左=患者左，右=患者右，上=患者上，下=患者下
    sagittal: 左=患者前，右=患者后，上=患者上，下=患者下
    """
    if plane == "axial":
        return np.rot90(slice2d, k=1)
    if plane == "coronal":
        return np.rot90(slice2d, k=1)
    if plane == "sagittal":
        return np.fliplr(np.rot90(slice2d, k=1))
    raise ValueError(f"Unknown plane: {plane}")


def get_plane_axis(plane: str) -> int:
    if plane == "sagittal":
        return 0
    if plane == "coronal":
        return 1
    if plane == "axial":
        return 2
    raise ValueError(f"Unknown plane: {plane}")


def get_plane_metadata(plane: str) -> dict:
    if plane == "axial":
        return {
            "view_convention": "neurological",
            "plane": "axial",
            "image_left": "patient_left",
            "image_right": "patient_right",
            "image_top": "patient_anterior",
            "image_bottom": "patient_posterior",
        }
    if plane == "coronal":
        return {
            "view_convention": "neurological",
            "plane": "coronal",
            "image_left": "patient_left",
            "image_right": "patient_right",
            "image_top": "patient_superior",
            "image_bottom": "patient_inferior",
        }
    if plane == "sagittal":
        return {
            "view_convention": "neurological",
            "plane": "sagittal",
            "image_left": "patient_anterior",
            "image_right": "patient_posterior",
            "image_top": "patient_superior",
            "image_bottom": "patient_inferior",
        }
    raise ValueError(f"Unknown plane: {plane}")


def get_slice_indices(n: int, only_center_k: int) -> list[int]:
    if only_center_k is None or only_center_k <= 0:
        return list(range(n))
    k = min(only_center_k, n)
    start = max(0, n // 2 - k // 2)
    end = min(n, start + k)
    return list(range(start, end))


def choose_planes_for_case(case_id: str, cfg: Config) -> list[str]:
    """
    根据配置为当前病例选择要导出的切片平面。

    当前推荐模式 axial_plus_random_coronal_or_sagittal：
    - 每个病例一定导出 axial；
    - 再从 coronal / sagittal 中固定随机选择一个额外平面；
    - 使用 case_id + RANDOM_SEED 做确定性随机，同一病例重复运行结果一致；
    - 不受病例遍历顺序影响，便于复现实验和排查问题。
    """
    candidate_planes = list(cfg.planes)
    valid = {"axial", "coronal", "sagittal"}
    bad = [p for p in candidate_planes if p not in valid]
    if bad:
        raise ValueError(f"Invalid plane(s): {bad}. Valid planes are axial/coronal/sagittal.")

    if cfg.plane_selection_mode == "all":
        return candidate_planes

    if cfg.plane_selection_mode == "fixed":
        return candidate_planes

    if cfg.plane_selection_mode == "random_one_per_case":
        if not candidate_planes:
            raise ValueError("PLANES is empty.")
        key = f"{cfg.random_seed}:one:{case_id}".encode("utf-8")
        digest = hashlib.md5(key).hexdigest()
        idx = int(digest, 16) % len(candidate_planes)
        return [candidate_planes[idx]]

    if cfg.plane_selection_mode == "axial_plus_random_coronal_or_sagittal":
        if "axial" not in candidate_planes:
            raise ValueError("PLANES must contain axial for axial_plus_random_coronal_or_sagittal mode.")

        extra_candidates = [p for p in ["coronal", "sagittal"] if p in candidate_planes]
        if len(extra_candidates) != 2:
            raise ValueError("PLANES must contain both coronal and sagittal for axial_plus_random_coronal_or_sagittal mode.")

        key = f"{cfg.random_seed}:extra:{case_id}".encode("utf-8")
        digest = hashlib.md5(key).hexdigest()
        idx = int(digest, 16) % len(extra_candidates)
        extra_plane = extra_candidates[idx]
        return ["axial", extra_plane]

    raise ValueError(
        f"Unknown PLANE_SELECTION_MODE: {cfg.plane_selection_mode}. "
        "Use axial_plus_random_coronal_or_sagittal, random_one_per_case, all, or fixed."
    )


# ============================================================
# 5. 标签名、脑区统计、组织统计
# ============================================================

def load_brainparc_label_map_from_csv(path: Path) -> Dict[int, str]:
    label_map: Dict[int, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return label_map

    if not rows:
        return label_map

    cols = list(rows[0].keys())

    id_candidates = [c for c in cols if c.lower() in ["brainparc_index", "index", "label", "id"]]
    id_col = id_candidates[0] if id_candidates else None

    if id_col is None:
        for c in cols:
            try:
                int(float(str(rows[0].get(c, "")).strip()))
                id_col = c
                break
            except Exception:
                pass

    if id_col is None:
        return label_map

    # 优先选择名字相关列；若没有，就选第一个非数值文本列。
    name_keywords = ["name", "label", "structure", "region", "roi", "abbr"]
    name_candidates = [
        c for c in cols
        if c != id_col and any(k in c.lower() for k in name_keywords)
    ]

    def is_text_col(c: str) -> bool:
        for row in rows[:20]:
            val = str(row.get(c, "")).strip()
            if not val:
                continue
            try:
                float(val)
                continue
            except Exception:
                return True
        return False

    name_col = None
    for c in name_candidates:
        if is_text_col(c):
            name_col = c
            break

    if name_col is None:
        for c in cols:
            if c != id_col and is_text_col(c):
                name_col = c
                break

    if name_col is None:
        return label_map

    for row in rows:
        try:
            lab = int(float(str(row.get(id_col, "")).strip()))
            name = str(row.get(name_col, "")).strip()
            if lab > 0 and name:
                label_map[lab] = name
        except Exception:
            continue

    return label_map


def load_brainparc_label_map(case_bp_dir: Path) -> Dict[int, str]:
    csv_path = find_file_recursive(case_bp_dir, ["present_labels.csv"])
    if csv_path is not None:
        mp = load_brainparc_label_map_from_csv(csv_path)
        if mp:
            return mp
    return {}


def infer_left_right_from_region_name(name: str) -> str:
    if name is None:
        return "unknown"

    s = str(name).strip().lower()
    left_patterns = [
        r"(^|[_\-\s])left([_\-\s]|$)",
        r"(^|[_\-\s])lh([_\-\s]|$)",
        r"ctx-lh",
        r"ctx_lh",
        r"[_\-]l$",
        r"^l[_\-]",
    ]
    right_patterns = [
        r"(^|[_\-\s])right([_\-\s]|$)",
        r"(^|[_\-\s])rh([_\-\s]|$)",
        r"ctx-rh",
        r"ctx_rh",
        r"[_\-]r$",
        r"^r[_\-]",
    ]
    if any(re.search(p, s) for p in left_patterns):
        return "left"
    if any(re.search(p, s) for p in right_patterns):
        return "right"
    return "unknown"


def infer_lobe_from_region_name(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ["frontal", "precentral", "paracentral", "pars", "orbitofrontal", "superiorfrontal", "middlefrontal", "inferiorfrontal"]):
        return "frontal_lobe"
    if any(k in s for k in ["temporal", "fusiform", "entorhinal", "parahippocampal", "bankssts"]):
        return "temporal_lobe"
    if any(k in s for k in ["parietal", "postcentral", "precuneus", "supramarginal"]):
        return "parietal_lobe"
    if any(k in s for k in ["occipital", "lingual", "cuneus", "pericalcarine", "lateraloccipital"]):
        return "occipital_lobe"
    if any(k in s for k in ["insula", "insular"]):
        return "insula"
    if any(k in s for k in ["ventricle", "ventricular"]):
        return "ventricle"
    if any(k in s for k in ["hippocampus", "hippocampal"]):
        return "hippocampal_region"
    if any(k in s for k in ["caudate", "putamen", "pallidum", "thalamus", "accumbens"]):
        return "subcortical_nuclei"
    return "unknown"


def bbox_centroid_from_mask(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return {"bbox_xyxy": None, "bbox_norm_xyxy": None, "centroid_xy": None, "centroid_norm_xy": None}
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return {
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_norm_xyxy": [float(xmin / w), float(ymin / h), float(xmax / w), float(ymax / h)],
        "centroid_xy": [cx, cy],
        "centroid_norm_xy": [float(cx / w), float(cy / h)],
    }


def label_slice_stats(label_slice: np.ndarray, label_map: Dict[int, str], min_area_px: int, kind: str) -> dict:
    arr = label_slice.astype(np.int64)
    labels = sorted([int(x) for x in np.unique(arr) if int(x) != 0])
    h, w = arr.shape

    items = []
    for lab in labels:
        mask = arr == lab
        area = int(mask.sum())
        if area < min_area_px:
            continue
        name = label_map.get(lab, f"label_{lab}")
        geom = bbox_centroid_from_mask(mask)
        item = {
            "label": int(lab),
            "name": name,
            "area_px": area,
            "area_frac_of_image": float(area / (h * w)),
            **geom,
        }
        if kind == "region":
            item["side"] = infer_left_right_from_region_name(name)
            item["lobe"] = infer_lobe_from_region_name(name)
        items.append(item)

    return {
        "available": True,
        "kind": kind,
        "n_visible_labels": len(items),
        "visible_labels": items,
    }


# ============================================================
# 6. BraTS seg 可选处理
# ============================================================

BRATS2024_LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
VALID_BRATS_LABELS = (1, 2, 3, 4)


def clean_brats_seg(seg: np.ndarray) -> np.ndarray:
    out = seg.copy().astype(np.uint8)
    bad = (out != 0) & (~np.isin(out, list(VALID_BRATS_LABELS)))
    out[bad] = 0
    return out


def make_seg_vis(seg_label: np.ndarray) -> np.ndarray:
    h, w = seg_label.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[seg_label == 1] = [255, 0, 0]
    vis[seg_label == 2] = [0, 255, 0]
    vis[seg_label == 3] = [0, 0, 255]
    vis[seg_label == 4] = [255, 255, 0]
    return vis


def seg_to_stats(seg_label: np.ndarray) -> dict:
    tumor = seg_label != 0
    stats = {
        "available": True,
        "tumor_present": bool(tumor.any()),
        "labels_present": [],
        "label_names_present": [],
        "areas_px": {},
        "tumor_frac_of_image": 0.0,
        "bbox_xyxy": None,
        "bbox_norm_xyxy": None,
        "centroid_xy": None,
        "centroid_norm_xy": None,
    }
    if not tumor.any():
        return stats

    h, w = seg_label.shape
    stats["tumor_frac_of_image"] = float(tumor.mean())
    for lab in VALID_BRATS_LABELS:
        area = int((seg_label == lab).sum())
        stats["areas_px"][str(lab)] = area
        if area > 0:
            stats["labels_present"].append(int(lab))
            stats["label_names_present"].append(BRATS2024_LABELS[lab])

    stats.update(bbox_centroid_from_mask(tumor))
    return stats


# ============================================================
# 7. 单病例转换
# ============================================================

def case_done(case_done_dir: Path, marker_name: str) -> bool:
    return (case_done_dir / marker_name).exists()


def mark_done(case_done_dir: Path, marker_name: str, meta: dict) -> None:
    case_done_dir.mkdir(parents=True, exist_ok=True)
    (case_done_dir / marker_name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_requested_image_modalities(cfg: Config) -> list[str]:
    modalities = []
    seen = set()
    for mod in ("t1n", *cfg.image_modalities):
        mod = normalize_modality_name(mod)
        if not mod or mod in seen:
            continue
        modalities.append(mod)
        seen.add(mod)
    return modalities


def load_case_data(t1n_path: Path, cfg: Config) -> dict:
    case_id = case_id_from_t1n(t1n_path)
    case_dir = t1n_path.parent

    t1n, t1n_affine = load_volume_float(t1n_path)
    ref_shape = tuple(t1n.shape)

    image_volumes: Dict[str, np.ndarray] = {"t1n": t1n}
    image_source_paths: Dict[str, str] = {"t1n": str(t1n_path)}
    for mod in get_requested_image_modalities(cfg):
        if mod == "t1n":
            continue

        mod_path = find_modality_file(case_dir, mod)
        if mod_path is None:
            msg = f"Image modality '{mod}' not found for case={case_id}"
            if cfg.require_all_image_modalities:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue

        image_volumes[mod] = load_aligned_image_volume(mod_path, ref_shape, t1n_affine, cfg, f"MRI modality {mod}")
        image_source_paths[mod] = str(mod_path)

    # 可选 BraTS seg
    seg = None
    seg_path = find_modality_file(case_dir, "seg")
    if seg_path is not None and cfg.save_brats_seg_if_available:
        seg_arr = load_label_volume(seg_path, ref_shape, t1n_affine, cfg, "BraTS seg").astype(np.uint8)
        seg = clean_brats_seg(seg_arr)
    elif cfg.require_brats_seg:
        raise FileNotFoundError(f"BraTS seg required but not found for case={case_id}")

    # BrainParc outputs
    bp_case_dir = find_brainparc_case_dir(cfg.brainparc_root, case_id)
    if bp_case_dir is None:
        if cfg.require_brainparc_masks:
            raise FileNotFoundError(f"BrainParc output folder not found for case={case_id}")
        tissue = None
        region = None
        label_map = {}
    else:
        tissue_path = find_file_recursive(bp_case_dir, ["tissue.nii.gz", "tissue.nii"])
        region_path = find_file_recursive(bp_case_dir, ["dk-struct.nii.gz", "dk-struct.nii"])
        if tissue_path is None or region_path is None:
            if cfg.require_brainparc_masks:
                raise FileNotFoundError(f"Missing tissue/dk-struct for case={case_id}, dir={bp_case_dir}")
        tissue = load_label_volume(tissue_path, ref_shape, t1n_affine, cfg, "BrainParc tissue") if tissue_path else None
        region = load_label_volume(region_path, ref_shape, t1n_affine, cfg, "BrainParc region") if region_path else None
        label_map = load_brainparc_label_map(bp_case_dir)

    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "t1n": t1n,
        "image_volumes": image_volumes,
        "image_source_paths": image_source_paths,
        "seg": seg,
        "tissue": tissue,
        "region": region,
        "label_map": label_map,
        "volume_shape": ref_shape,
    }


def convert_one_case(t1n_path: Path, cfg: Config, images_dir: Path, masks_dir: Path, manifests_dir: Path) -> bool:
    case_id = case_id_from_t1n(t1n_path)
    case_done_dir = cfg.out_root / "cases_done" / case_id
    case_manifest = manifests_dir / f"{case_id}.jsonl"
    if case_done(case_done_dir, cfg.done_marker) and case_manifest.exists() and case_manifest.stat().st_size > 0:
        return True

    data = load_case_data(t1n_path, cfg)
    t1n = data["t1n"]
    image_volumes = data["image_volumes"]
    image_source_paths = data["image_source_paths"]
    seg = data["seg"]
    tissue = data["tissue"]
    region = data["region"]
    label_map = data["label_map"]
    volume_shape = data["volume_shape"]

    if tissue is None or region is None:
        raise ValueError(f"BrainParc tissue and region masks are required for case={case_id}")

    image_u8_by_mod = {
        mod: mri_min_max_preprocess(volume, cfg)
        for mod, volume in image_volumes.items()
    }
    t1n_u8 = image_u8_by_mod["t1n"]

    case_images_dir = images_dir / case_id
    case_mask_root = masks_dir / case_id
    case_tissue_dir = case_mask_root / "brainparc_tissue"
    case_region_dir = case_mask_root / "brainparc_region"
    case_seg_label_dir = case_mask_root / "seg_label"
    case_seg_vis_dir = case_mask_root / "seg_vis"

    for p in [case_images_dir, case_tissue_dir, case_region_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if seg is not None:
        case_seg_label_dir.mkdir(parents=True, exist_ok=True)
        case_seg_vis_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest = case_manifest.with_suffix(".jsonl.tmp")

    selected_planes = choose_planes_for_case(case_id, cfg)

    saved = 0
    total_possible = 0

    try:
        with tmp_manifest.open("w", encoding="utf-8") as f:
            for plane in selected_planes:
                axis = get_plane_axis(plane)
                n_slices = volume_shape[axis]
                indices = get_slice_indices(n_slices, cfg.only_center_k)
                total_possible += len(indices)

                for idx in indices:
                    if idx % cfg.slice_stride != 0:
                        continue

                    t1n_raw = np.take(t1n_u8, idx, axis=axis)
                    nz_ratio = float((np.abs(t1n_raw) > 1e-8).mean())
                    if cfg.skip_empty and nz_ratio < cfg.empty_threshold:
                        continue

                    image_rel_paths = {}
                    for mod, image_u8 in image_u8_by_mod.items():
                        image_raw = np.take(image_u8, idx, axis=axis)
                        image_oriented = orient_slice_for_vlm(image_raw, plane)
                        image_out = case_images_dir / f"{case_id}-{mod}_{plane}_slice_{idx:03d}.png"
                        save_png_u8(image_oriented, image_out, cfg.png_compress_level)
                        image_rel_paths[mod] = str(image_out.relative_to(cfg.out_root))

                    tissue_raw = np.take(tissue, idx, axis=axis) if tissue is not None else None
                    region_raw = np.take(region, idx, axis=axis) if region is not None else None
                    tissue_oriented = orient_slice_for_vlm(tissue_raw, plane) if tissue_raw is not None else None
                    region_oriented = orient_slice_for_vlm(region_raw, plane) if region_raw is not None else None

                    tissue_out = case_tissue_dir / f"{case_id}-brainparc_tissue_{plane}_slice_{idx:03d}.png"
                    region_out = case_region_dir / f"{case_id}-brainparc_region_{plane}_slice_{idx:03d}.png"

                    save_label_png(tissue_oriented, tissue_out, cfg.png_compress_level)
                    save_label_png(region_oriented, region_out, cfg.png_compress_level)

                    brats_seg_info = None
                    if seg is not None:
                        seg_raw = np.take(seg, idx, axis=axis).astype(np.uint8)
                        seg_oriented = orient_slice_for_vlm(seg_raw, plane).astype(np.uint8)
                        seg_label_out = case_seg_label_dir / f"{case_id}-seglabel_{plane}_slice_{idx:03d}.png"
                        seg_vis_out = case_seg_vis_dir / f"{case_id}-segvis_{plane}_slice_{idx:03d}.png"
                        save_label_png(seg_oriented, seg_label_out, cfg.png_compress_level)
                        Image.fromarray(make_seg_vis(seg_oriented)).save(seg_vis_out, compress_level=cfg.png_compress_level)
                        brats_seg_info = {
                            "seg_label": str(seg_label_out.relative_to(cfg.out_root)),
                            "seg_vis": str(seg_vis_out.relative_to(cfg.out_root)),
                            "seg_stats": seg_to_stats(seg_oriented),
                        }

                    region_stats = label_slice_stats(region_oriented, label_map, cfg.min_region_area_px, kind="region")
                    tissue_stats = label_slice_stats(tissue_oriented, {}, cfg.min_tissue_area_px, kind="tissue")

                    view_meta = get_plane_metadata(plane)
                    condition = {
                        "view": view_meta,
                        "main_modality": "t1n",
                        "available_image_modalities": list(image_rel_paths.keys()),
                        "brainparc_region_stats": region_stats,
                        "brainparc_tissue_stats": tissue_stats,
                        "instruction_note": (
                            "Use this structured BrainParc information as ground truth. "
                            "BrainParc tissue/region masks were generated from t1n and aligned to all saved modalities. "
                            "Do not infer patient left/right only from visual appearance."
                        ),
                    }
                    if brats_seg_info is not None:
                        condition["brats_seg_stats"] = brats_seg_info["seg_stats"]

                    row = {
                        "id": f"{case_id}_{plane}_{idx:03d}",
                        "case": case_id,
                        "plane": plane,
                        "slice": int(idx),
                        "main_modality": "t1n",
                        "modalities": list(image_rel_paths.keys()),
                        "primary_image": image_rel_paths.get("t1n"),
                        "volume_shape_canonical": list(map(int, volume_shape)),
                        "view": view_meta,
                        "images": image_rel_paths,
                        "image_source_paths": image_source_paths,
                        "brainparc": {
                            "tissue": str(tissue_out.relative_to(cfg.out_root)),
                            "region": str(region_out.relative_to(cfg.out_root)),
                            "tissue_stats": tissue_stats,
                            "region_stats": region_stats,
                        },
                        "brats_seg": brats_seg_info,
                        "vlm_cond": "<ANATOMY_JSON>" + json.dumps(condition, ensure_ascii=False) + "</ANATOMY_JSON>",
                    }

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    saved += 1

        tmp_manifest.replace(case_manifest)
        mark_done(
            case_done_dir,
            cfg.done_marker,
            {
                "case": case_id,
                "n_slices_possible": int(total_possible),
                "n_slices_saved": int(saved),
                "plane_selection_mode": cfg.plane_selection_mode,
                "candidate_planes": list(cfg.planes),
                "selected_planes": list(selected_planes),
                "image_modalities": list(image_u8_by_mod.keys()),
                "image_source_paths": image_source_paths,
                "volume_shape_canonical": list(map(int, volume_shape)),
                "brainparc_label_map_n": len(label_map),
                "manifest": str(case_manifest.relative_to(cfg.out_root)),
            },
        )
        return True

    except Exception:
        if tmp_manifest.exists():
            try:
                tmp_manifest.unlink()
            except Exception:
                pass
        raise


# ============================================================
# 8. 全局 manifest
# ============================================================

def build_global_manifest(manifests_dir: Path, global_manifest: Path) -> None:
    files = sorted([p for p in manifests_dir.glob("*.jsonl") if p.is_file()])
    global_manifest.parent.mkdir(parents=True, exist_ok=True)
    with global_manifest.open("w", encoding="utf-8") as out:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)


# ============================================================
# 9. 主流程
# ============================================================

def main() -> None:
    cfg = Config()

    if cfg.slice_stride <= 0:
        raise ValueError(f"SLICE_STRIDE must be >= 1, got {cfg.slice_stride}")
    if not get_requested_image_modalities(cfg):
        raise ValueError("IMAGE_MODALITIES must contain at least one modality.")
    if not 0 <= cfg.png_compress_level <= 9:
        raise ValueError(f"PNG_COMPRESS_LEVEL must be between 0 and 9, got {cfg.png_compress_level}")
    if not 0.0 <= cfg.empty_threshold <= 1.0:
        raise ValueError(f"EMPTY_THRESHOLD must be between 0 and 1, got {cfg.empty_threshold}")
    if not 0 <= cfg.mri_low_percentile <= 100 or not 0 <= cfg.mri_high_percentile <= 100:
        raise ValueError(
            "MRI percentiles must be between 0 and 100, "
            f"got low={cfg.mri_low_percentile}, high={cfg.mri_high_percentile}"
        )
    if cfg.mri_low_percentile > cfg.mri_high_percentile:
        raise ValueError(
            "MRI_LOW_PERCENTILE must be <= MRI_HIGH_PERCENTILE, "
            f"got {cfg.mri_low_percentile} > {cfg.mri_high_percentile}"
        )

    if not cfg.input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {cfg.input_root}")
    if not cfg.brainparc_root.exists():
        raise FileNotFoundError(f"BRAINPARC_ROOT not found: {cfg.brainparc_root}")

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    images_dir = cfg.out_root / "images"
    masks_dir = cfg.out_root / "masks"
    manifests_dir = cfg.out_root / "manifests"
    global_manifest = cfg.out_root / "manifest.jsonl"

    all_t1n_files = find_t1n_files(cfg.input_root)

    if not all_t1n_files:
        raise RuntimeError(f"No '*-t1n.nii.gz' found under: {cfg.input_root}")

    t1n_files = select_t1n_files_for_run(all_t1n_files, cfg)

    if not t1n_files:
        raise RuntimeError(
            "No cases selected for slicing. Please check BRAINPARC_ROOT, "
            "ONLY_PROCESS_BRAINPARC_DONE_CASES, MAX_CASES, and CASE_IDS."
    )

    logger.info(f"[config] input_root={cfg.input_root}")
    logger.info(f"[config] brainparc_root={cfg.brainparc_root}")
    logger.info(f"[config] out_root={cfg.out_root}")
    logger.info(f"[config] plane_selection_mode={cfg.plane_selection_mode}")
    logger.info(f"[config] candidate_planes={cfg.planes}")
    logger.info(f"[config] image_modalities={get_requested_image_modalities(cfg)}")
    logger.info(f"[config] require_all_image_modalities={cfg.require_all_image_modalities}")
    logger.info(f"[config] random_seed={cfg.random_seed}")
    logger.info(f"[config] only_center_k={cfg.only_center_k}")
    logger.info(f"[config] slice_stride={cfg.slice_stride}")
    logger.info(f"[config] empty_threshold={cfg.empty_threshold}")
    logger.info(f"[config] only_process_brainparc_done_cases={cfg.only_process_brainparc_done_cases}")
    logger.info(f"[config] max_cases={cfg.max_cases}")
    logger.info(f"[config] case_ids={list(cfg.case_ids)}")
    logger.info(f"[cases] total_t1n_found={len(all_t1n_files)}")
    logger.info(f"[cases] selected_for_slicing={len(t1n_files)}")

    done = 0
    failed = 0
    skipped = 0
    summary_rows = []

    for t1n_path in tqdm(t1n_files, desc="Converting cases"):
        case_id = case_id_from_t1n(t1n_path)
        try:
            ok = convert_one_case(t1n_path, cfg, images_dir, masks_dir, manifests_dir)
            if ok:
                done += 1
                status = "ok"
            else:
                skipped += 1
                status = "skipped"
            selected_planes = choose_planes_for_case(case_id, cfg)
            summary_rows.append({
                "case_id": case_id,
                "status": status,
                "selected_planes": ";".join(selected_planes),
                "t1n_path": str(t1n_path),
                "error": "",
            })
        except Exception as e:
            failed += 1
            logger.error(f"[ERROR] case failed: {case_id}")
            traceback.print_exc()
            try:
                selected_planes = choose_planes_for_case(case_id, cfg)
                selected_planes_str = ";".join(selected_planes)
            except Exception:
                selected_planes_str = ""
            summary_rows.append({
                "case_id": case_id,
                "status": "failed",
                "selected_planes": selected_planes_str,
                "t1n_path": str(t1n_path),
                "error": repr(e),
            })

    build_global_manifest(manifests_dir, global_manifest)

    # 保存 summary.csv
    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(cfg.out_root / "summary.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

    logger.info("=== DONE ===")
    logger.info(f"done={done} skipped={skipped} failed={failed}")
    logger.info(f"global manifest: {global_manifest}")


if __name__ == "__main__":
    main()
