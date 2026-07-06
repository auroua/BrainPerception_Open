#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_perception_dataset.py

PyTorch Dataset + DataLoader for the multi-round BrainParc reasoning-segmentation
data produced by:
    1_run_brainparc.py -> 2_slice_brainparc_2d.py -> 3_build_multiround_dataset.py

It is meant to train a LISA / SegLLM-style "brain perception" model:
a multimodal LLM (vision tower + LLM) that, for each conversation round, emits a
single ``<SEG>`` token whose last-layer hidden state is fed to a mask decoder
(e.g. a SAM-style head) to produce a binary mask.

What one dataset sample (one conversation) provides
---------------------------------------------------
- ``images_clip`` : 3xHcxWc tensor for the LLM's vision tower (CLIP-normalized,
  built from the main modality, replicated to 3 channels).
- ``images_seg``  : CxHsxWs tensor for the mask decoder branch. By default it
  stacks all available MRI modalities (t1n, t1c, t2w, t2f) as C channels so the
  segmentation head can exploit the full multi-modal signal.
- ``input_ids`` / ``labels`` : the full multi-round conversation tokenized with
  piece-wise label masking (only assistant tokens, including ``<SEG>`` and the
  turn-ending eos, are supervised).
- ``masks`` : (num_seg, Hm, Wm) float{0,1} tensor, one ground-truth mask per
  ``<SEG>`` token, in round order. ``num_seg == (input_ids == seg_token_id).sum()``.

Dataset layout this loader expects (DATASET_ROOT)::

    DATASET_ROOT/
      multiround_dataset/multiround_dialogues.jsonl   <- source of truth
      multiround_dataset/masks/<case>/*.png           <- binary 0/255 masks
      images/<case>/<case>-<mod>_<plane>_slice_<idx>.png

All ``image`` / ``images`` / ``target_mask`` paths inside the jsonl are relative
to DATASET_ROOT.

Note: the top-level ``manifest.jsonl`` may be empty in subset exports; this loader
does NOT depend on it. The dialogues jsonl is the single source of truth.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover - allows dry-run validation without torch
    _HAS_TORCH = False

    class Dataset:  # type: ignore
        ...


# ============================================================
# 0. Constants
# ============================================================

IGNORE_INDEX = -100               # label id ignored by CrossEntropyLoss
IMAGE_TOKEN_INDEX = -200          # LLaVA convention; expanded inside model.forward
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_TOKEN = "<SEG>"       # must match SEG_TOKEN used by 3_build_multiround_dataset.py

# Modalities, in the channel order used to build images_seg.
ALL_MODALITIES: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")

# CLIP (OpenAI) image normalization for the LLM vision tower.
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SAM's image encoder is pretrained on ImageNet-normalized pixels (on the [0,1]
# scale). images_seg is fed straight into that (frozen) encoder, so by default we
# normalize with these stats rather than leaving raw [0,1] pixels, which would be
# out of distribution for the encoder. Extra (4th+) MRI channels reuse the mean
# RGB statistic, matching how the inflated patch-embed initializes its extra filters.
SEG_IMAGENET_MEAN = (0.485, 0.456, 0.406)
SEG_IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_seg_norm(n_channels: int) -> Tuple[List[float], List[float]]:
    extra_mean = sum(SEG_IMAGENET_MEAN) / 3.0
    extra_std = sum(SEG_IMAGENET_STD) / 3.0
    mean = [SEG_IMAGENET_MEAN[i] if i < 3 else extra_mean for i in range(n_channels)]
    std = [SEG_IMAGENET_STD[i] if i < 3 else extra_std for i in range(n_channels)]
    return mean, std

DEFAULT_SYSTEM_PROMPT = (
    "You are a brain MRI perception assistant. The image is a 2D brain MRI slice "
    "with aligned multi-modal channels. For every request, locate the asked target "
    "and answer with a segmentation token."
)


# ============================================================
# 1. Config
# ============================================================

@dataclass
class BrainPerceptionDataConfig:
    # --- paths ---
    dataset_root: str = "/Users/albert_we/Datasets/segllm_10samples"
    dialogues_rel: str = "multiround_dataset/multiround_dialogues.jsonl"

    # --- modality handling ---
    # Modality fed to the LLM vision tower (replicated to 3 channels, CLIP-normalized).
    clip_modality: str = "t1n"
    # Build the generic 3-ch CLIP branch (images_clip). Set False when the VLM has
    # its own image processor (e.g. Qwen3-VL); then only vlm_image_path is returned.
    build_clip_branch: bool = True
    # Modalities stacked (in this order) to form images_seg for the mask decoder.
    # Missing modalities are zero-filled so channel count stays constant.
    seg_modalities: Tuple[str, ...] = ALL_MODALITIES
    # Randomly select this many of seg_modalities per sample to form images_seg. Set to
    # 3 so SAM runs on its native 3-channel patch embed (no encoder inflation needed) and
    # to act as modality-dropout augmentation. None or >= len(seg_modalities) -> use all
    # (in order). MUST equal the model's seg_in_channels; the trainer keeps them in sync.
    seg_select_k: Optional[int] = 3
    # True -> pick a random subset each __getitem__ (augmentation, for training).
    # False -> deterministically take the first seg_select_k modalities (use for val/test).
    seg_random_modalities: bool = True

    # --- spatial sizes ---
    clip_image_size: int = 224        # vision-tower input
    seg_image_size: int = 1024        # mask-decoder branch input (SAM-style)
    mask_size: int = 1024             # GT mask resolution (usually == seg_image_size)

    # --- seg-branch pixel normalization ---
    # Per-channel (x/255 - mean)/std. If both None, SAM-style ImageNet stats are used
    # (see _default_seg_norm), because the frozen SAM encoder expects normalized input.
    seg_pixel_mean: Optional[Tuple[float, ...]] = None
    seg_pixel_std: Optional[Tuple[float, ...]] = None

    # --- conversation / tokenization ---
    seg_token: str = DEFAULT_SEG_TOKEN
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_seq_len: int = 2048           # conversations longer than this are dropped at init
    # If True, the user question of round 1 carries the <image> placeholder.
    place_image_in_first_round: bool = True

    # --- filtering ---
    keep_conversation_types: Optional[Tuple[str, ...]] = None  # None == keep all
    keep_planes: Optional[Tuple[str, ...]] = None              # e.g. ("axial",)
    keep_case_ids: Optional[Tuple[str, ...]] = None            # for train/val split
    max_rounds: Optional[int] = None

    # --- misc ---
    seed: int = 20260427


# ============================================================
# 2. Image / mask transforms (numpy core, torch tensor out)
# ============================================================

def _load_gray_u8(path: Path) -> np.ndarray:
    """Load a PNG as a single-channel uint8 HxW array."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _resize_np(arr: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    """Resize a 2D array to (size, size). nearest for masks, bilinear for images."""
    resample = Image.NEAREST if nearest else Image.BILINEAR
    img = Image.fromarray(arr)
    img = img.resize((size, size), resample=resample)
    return np.array(img)


def build_clip_image(case_dir_rel_paths: Dict[str, str], root: Path, cfg: BrainPerceptionDataConfig) -> np.ndarray:
    """Return a (3, Hc, Wc) float32 CLIP-normalized array from the clip modality."""
    mod = cfg.clip_modality
    rel = case_dir_rel_paths.get(mod) or next(iter(case_dir_rel_paths.values()))
    g = _load_gray_u8(root / rel)
    g = _resize_np(g, cfg.clip_image_size, nearest=False).astype(np.float32) / 255.0
    img = np.stack([g, g, g], axis=0)  # (3, H, W)
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    return (img - mean) / std


def _select_seg_modalities(images_rel: Dict[str, str], cfg: BrainPerceptionDataConfig) -> Tuple[str, ...]:
    """Pick which modalities go into images_seg. With seg_select_k < len(seg_modalities),
    keep k of them (random subset for training, first-k for deterministic val), preferring
    modalities that are actually present so we don't emit blank channels. Selected
    modalities are returned in the original seg_modalities order for stable channel
    semantics, and the count is always exactly k (missing ones get zero-filled later)."""
    mods = list(cfg.seg_modalities)
    k = cfg.seg_select_k
    if k is None or k >= len(mods) or k <= 0:
        return tuple(mods)
    avail = [m for m in mods if images_rel.get(m) is not None]
    pool = avail if len(avail) >= k else mods
    if cfg.seg_random_modalities:
        keep = set(random.sample(pool, k))
    else:
        keep = set(pool[:k])
    return tuple(m for m in mods if m in keep)


def build_seg_image(images_rel: Dict[str, str], root: Path, cfg: BrainPerceptionDataConfig) -> np.ndarray:
    """Return a (C, Hs, Ws) float32 array stacking the selected seg modalities."""
    chans = []
    for mod in _select_seg_modalities(images_rel, cfg):
        rel = images_rel.get(mod)
        if rel is None:
            chans.append(np.zeros((cfg.seg_image_size, cfg.seg_image_size), dtype=np.float32))
            continue
        g = _load_gray_u8(root / rel)
        g = _resize_np(g, cfg.seg_image_size, nearest=False).astype(np.float32)
        chans.append(g)
    img = np.stack(chans, axis=0)  # (C, H, W) in 0..255

    if cfg.seg_pixel_mean is not None and cfg.seg_pixel_std is not None:
        mean = np.array(cfg.seg_pixel_mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(cfg.seg_pixel_std, dtype=np.float32).reshape(-1, 1, 1)
    else:
        mean_l, std_l = _default_seg_norm(img.shape[0])
        mean = np.array(mean_l, dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(std_l, dtype=np.float32).reshape(-1, 1, 1)
    img = (img / 255.0 - mean) / std
    return img


def build_mask(mask_rel: str, root: Path, cfg: BrainPerceptionDataConfig) -> np.ndarray:
    """Return an (Hm, Wm) float32 {0,1} array from a binary 0/255 PNG."""
    m = _load_gray_u8(root / mask_rel)
    m = _resize_np(m, cfg.mask_size, nearest=True)
    return (m > 127).astype(np.float32)


# ============================================================
# 3. Conversation assembly + tokenization
# ============================================================

def tokenizer_image_token(text: str, tokenizer, image_token: str = DEFAULT_IMAGE_TOKEN) -> List[int]:
    """
    Tokenize ``text`` replacing each ``image_token`` with a single IMAGE_TOKEN_INDEX
    sentinel (LLaVA convention). No BOS/EOS added here.
    """
    if image_token not in text:
        return tokenizer(text, add_special_tokens=False).input_ids
    chunks = text.split(image_token)
    ids: List[int] = []
    for i, chunk in enumerate(chunks):
        if chunk:
            ids.extend(tokenizer(chunk, add_special_tokens=False).input_ids)
        if i != len(chunks) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    return ids


def build_conversation_ids(
    rounds: List[dict],
    tokenizer,
    cfg: BrainPerceptionDataConfig,
) -> Tuple[List[int], List[int]]:
    """
    Build (input_ids, labels) for a multi-round conversation with piece-wise label
    masking. Only assistant tokens (answer text + eos) are supervised.

    Layout::

        [BOS] {system}
        USER: <image>\\n{q1} ASSISTANT: {a1}{eos}
        USER: {q2} ASSISTANT: {a2}{eos}
        ...
    """
    eos_id = tokenizer.eos_token_id
    input_ids: List[int] = []
    labels: List[int] = []

    def add(ids: Sequence[int], supervise: bool) -> None:
        input_ids.extend(ids)
        labels.extend(list(ids) if supervise else [IGNORE_INDEX] * len(ids))

    if tokenizer.bos_token_id is not None:
        add([tokenizer.bos_token_id], supervise=False)

    if cfg.system_prompt:
        add(tokenizer(cfg.system_prompt + "\n\n", add_special_tokens=False).input_ids, supervise=False)

    for i, rd in enumerate(rounds):
        question = str(rd["question"])
        answer = str(rd["answer"])

        if i == 0 and cfg.place_image_in_first_round:
            human = f"USER: {DEFAULT_IMAGE_TOKEN}\n{question} ASSISTANT: "
        else:
            human = f"USER: {question} ASSISTANT: "

        add(tokenizer_image_token(human, tokenizer), supervise=False)

        ans_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if eos_id is not None:
            ans_ids = ans_ids + [eos_id]
        add(ans_ids, supervise=True)

    return input_ids, labels


def add_brain_perception_tokens(tokenizer) -> int:
    """
    Ensure the tokenizer knows the ``<SEG>`` token. Returns the number of tokens
    added (call ``model.resize_token_embeddings(len(tokenizer))`` if > 0).
    """
    added = tokenizer.add_tokens([DEFAULT_SEG_TOKEN], special_tokens=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return added


# ============================================================
# 4. Dataset
# ============================================================

class BrainPerceptionDataset(Dataset):
    """
    One item == one multi-round conversation grounded on a single 2D MRI slice
    (with its aligned modalities and per-round binary masks).

    Pass a HuggingFace tokenizer to get tokenized ``input_ids`` / ``labels``.
    If ``tokenizer is None`` the item still returns images, masks and the raw
    rounds (useful for debugging / building your own tokenization).
    """

    def __init__(self, cfg: BrainPerceptionDataConfig, tokenizer=None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to instantiate BrainPerceptionDataset.")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.root = Path(cfg.dataset_root)
        self.jsonl_path = self.root / cfg.dialogues_rel
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"dialogues jsonl not found: {self.jsonl_path}")

        self.seg_token_id = None
        if tokenizer is not None:
            sid = tokenizer.convert_tokens_to_ids(cfg.seg_token)
            unk = getattr(tokenizer, "unk_token_id", None)
            if sid is None or sid == unk:
                raise ValueError(
                    f"Tokenizer does not contain seg token {cfg.seg_token!r}. "
                    "Call add_brain_perception_tokens(tokenizer) and resize embeddings first."
                )
            self.seg_token_id = sid

        self.offsets: List[int] = self._build_index()
        if not self.offsets:
            raise RuntimeError("No conversations left after filtering. Check the config filters.")

    # ---- indexing (lazy, memory-light) ----
    def _passes_filters(self, d: dict) -> bool:
        c = self.cfg
        if c.keep_conversation_types is not None and d.get("conversation_type") not in c.keep_conversation_types:
            return False
        if c.keep_planes is not None and d.get("plane") not in c.keep_planes:
            return False
        if c.keep_case_ids is not None and d.get("case_id") not in c.keep_case_ids:
            return False
        rounds = d.get("rounds", [])
        if not rounds:
            return False
        # NOTE: max_rounds is applied by *truncating* in __getitem__, not by dropping
        # the dialogue here — otherwise longer dialogues would be silently discarded.
        return True

    def _build_index(self) -> List[int]:
        offsets: List[int] = []
        with open(self.jsonl_path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if self._passes_filters(d):
                    offsets.append(off)
        return offsets

    def _read(self, offset: int) -> dict:
        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline().decode("utf-8"))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self._read(self.offsets[idx])
        cfg = self.cfg
        images_rel: Dict[str, str] = d.get("images") or {"t1n": d["image"]}
        rounds: List[dict] = d["rounds"]
        if cfg.max_rounds is not None:
            # Truncate to the first max_rounds rounds instead of dropping the dialogue.
            # Refs point backward (to lower round_id), so a prefix stays self-consistent;
            # masks / <SEG> / target_labels below are all rebuilt from this list.
            rounds = rounds[: cfg.max_rounds]

        # images
        images_seg = torch.from_numpy(build_seg_image(images_rel, self.root, cfg)).float()

        # masks (one per round, in order)
        masks_np = [build_mask(rd["target_mask"], self.root, cfg) for rd in rounds]
        masks = torch.from_numpy(np.stack(masks_np, axis=0)).float()  # (R, Hm, Wm)

        # absolute path to the modality fed to the VLM vision tower (its own processor
        # normalizes/tiles it). Used by the Qwen3-VL collator instead of images_clip.
        vlm_rel = images_rel.get(cfg.clip_modality) or next(iter(images_rel.values()))

        sample: Dict[str, Any] = {
            "conversation_id": d["conversation_id"],
            "conversation_type": d["conversation_type"],
            "case_id": d["case_id"],
            "plane": d["plane"],
            "images_seg": images_seg,
            "masks": masks,
            "vlm_image_path": str(self.root / vlm_rel),
            "target_labels": torch.tensor([int(rd["target_label"]) for rd in rounds], dtype=torch.long),
            "rounds": rounds,
        }
        if cfg.build_clip_branch:
            sample["images_clip"] = torch.from_numpy(build_clip_image(images_rel, self.root, cfg)).float()

        if self.tokenizer is not None:
            input_ids, labels = build_conversation_ids(rounds, self.tokenizer, cfg)
            n_seg = sum(1 for t in input_ids if t == self.seg_token_id)
            if n_seg != masks.shape[0]:
                raise RuntimeError(
                    f"{d['conversation_id']}: #<SEG> tokens ({n_seg}) != #masks ({masks.shape[0]})."
                )
            sample["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            sample["labels"] = torch.tensor(labels, dtype=torch.long)
        return sample


# ============================================================
# 5. Collate + DataLoader
# ============================================================

@dataclass
class BrainPerceptionCollator:
    """Pads variable-length conversations and stacks the fixed-size image tensors.

    Masks vary in count per sample, so they are kept as a list plus a flat tensor
    with per-sample offsets — convenient for a LISA-style decoder that gathers
    ``<SEG>`` hidden states across the whole batch.
    """
    pad_token_id: int
    has_text: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "conversation_ids": [b["conversation_id"] for b in batch],
            "conversation_types": [b["conversation_type"] for b in batch],
            "case_ids": [b["case_id"] for b in batch],
            "planes": [b["plane"] for b in batch],
            "images_seg": torch.stack([b["images_seg"] for b in batch], dim=0),
            "masks_list": [b["masks"] for b in batch],
            "masks": torch.cat([b["masks"] for b in batch], dim=0),  # (sum_R, H, W)
            "num_masks_per_sample": torch.tensor([b["masks"].shape[0] for b in batch], dtype=torch.long),
            "target_labels": [b["target_labels"] for b in batch],
        }
        if "images_clip" in batch[0]:
            out["images_clip"] = torch.stack([b["images_clip"] for b in batch], dim=0)

        if self.has_text and "input_ids" in batch[0]:
            max_len = max(b["input_ids"].shape[0] for b in batch)
            input_ids, labels, attn = [], [], []
            for b in batch:
                ids = b["input_ids"]
                lab = b["labels"]
                pad = max_len - ids.shape[0]
                if pad > 0:
                    ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
                    lab = torch.cat([lab, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
                a = torch.zeros(max_len, dtype=torch.long)
                a[: b["input_ids"].shape[0]] = 1
                input_ids.append(ids)
                labels.append(lab)
                attn.append(a)
            out["input_ids"] = torch.stack(input_ids, dim=0)
            out["labels"] = torch.stack(labels, dim=0)
            out["attention_mask"] = torch.stack(attn, dim=0)
        return out


def split_cases_by_ratio(
    all_case_ids: Sequence[str], val_ratio: float = 0.2, seed: int = 20260427
) -> Tuple[List[str], List[str]]:
    """Deterministic case-level train/val split (no slice leakage across splits)."""
    cases = sorted(set(all_case_ids))
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_ratio))) if len(cases) > 1 else 0
    val = sorted(cases[:n_val])
    train = sorted(cases[n_val:])
    return train, val


def list_all_case_ids(cfg: BrainPerceptionDataConfig) -> List[str]:
    cases = set()
    path = Path(cfg.dataset_root) / cfg.dialogues_rel
    with open(path, "rb") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                cases.add(json.loads(s)["case_id"])
            except Exception:
                continue
    return sorted(cases)


def build_brain_perception_dataloader(
    cfg: BrainPerceptionDataConfig,
    tokenizer=None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> "DataLoader":
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required to build the dataloader.")
    dataset = BrainPerceptionDataset(cfg, tokenizer=tokenizer)
    pad_id = 0
    has_text = tokenizer is not None
    if has_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    collator = BrainPerceptionCollator(pad_token_id=pad_id, has_text=has_text)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )


# ============================================================
# 6. Dry-run validation (no torch / tokenizer required)
# ============================================================

def _dry_run(root: str, n: int = 4) -> None:
    cfg = BrainPerceptionDataConfig(dataset_root=root)
    rootp = Path(root)
    path = rootp / cfg.dialogues_rel
    print(f"[dry-run] dialogues: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()][:n]
    for ln in lines:
        d = json.loads(ln)
        images_rel = d.get("images") or {"t1n": d["image"]}
        clip = build_clip_image(images_rel, rootp, cfg)
        seg = build_seg_image(images_rel, rootp, cfg)
        masks = [build_mask(rd["target_mask"], rootp, cfg) for rd in d["rounds"]]
        masks = np.stack(masks, axis=0)
        print("-" * 90)
        print(f"id={d['conversation_id']}  type={d['conversation_type']}  plane={d['plane']}  rounds={len(d['rounds'])}")
        print(f"  images_clip {clip.shape} {clip.dtype}  range[{clip.min():.3f},{clip.max():.3f}]")
        print(f"  images_seg  {seg.shape} {seg.dtype}  range[{seg.min():.3f},{seg.max():.3f}]")
        print(f"  masks       {masks.shape}  pos_frac={(masks > 0).mean():.4f}")
        for rd in d["rounds"]:
            print(f"    R{rd['round_id']} target={rd['target_name']!r} label={rd['target_label']} type={rd['target_type']}")
    print("-" * 90)
    print("[dry-run] OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/albert_we/Datasets/segllm_10samples")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    _dry_run(args.root, args.n)
