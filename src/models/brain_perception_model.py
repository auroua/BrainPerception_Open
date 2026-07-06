#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_perception_model.py

LISA / MedSeg-R style "embedding-as-mask" brain-perception model:

    Qwen3-VL-4B  (vision + language encoder)
        │  hidden state at each <SEG> token
        ▼
    seg_projection (MLP)  ->  SAM prompt embedding
        │
        ▼
    SAM mask decoder  (image features from images_seg)  ->  binary mask logits

The mask-decoder head can be either SAM 3 (default) or SAM 2, selected with
`BrainPerceptionModelConfig.sam_version` ("sam3" / "sam2"). SAM 3 builds on the
SAM 2 module layout, so the encode/decode wrapper is shared; only model loading
differs (`SAMSegHead._load_sam2` / `_load_sam3`).

Training signal:
    L = lm_weight * CE(text)  +  bce_weight * BCE(mask)  +  dice_weight * Dice(mask)

The novel "embedding-as-mask" wiring (gathering <SEG> hidden states and decoding
them into masks) is fully concrete here. The two *version-dependent* touch points
are isolated so you only adapt small wrappers if your installed package APIs differ:

  * Qwen3-VL loading / forward  -> `load_qwen3_vl(...)`
  * SAM 2 / SAM 3 encode/decode -> `SAMSegHead` (alias `SAM3SegHead`)

Pairs with src/dataset/brain_perception_dataset.py (use build_clip_branch=False).

Tested shapes (single A100/H100, LoRA on the LLM, SAM image encoder frozen):
  images_seg : (B, 4, 1024, 1024)
  masks      : (sum_R, 1024, 1024)   one GT per <SEG>, batch-major / round order
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.brain_perception_dataset import (
    DEFAULT_SEG_TOKEN,
    IGNORE_INDEX,
    BrainPerceptionDataConfig,
    BrainPerceptionDataset,
)


# ============================================================
# 0. Config
# ============================================================

@dataclass
class BrainPerceptionModelConfig:
    # backbones
    vlm_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"  # "base" lacks a chat template; see note in load_qwen3_vl
    # Where to resolve vlm_name_or_path from. "huggingface" keeps the normal
    # transformers behavior; "modelscope" first downloads/resolves a local
    # ModelScope snapshot, then loads that local directory with transformers.
    vlm_source: str = "huggingface"
    vlm_revision: Optional[str] = None
    vlm_cache_dir: Optional[str] = None
    # Which SAM family to use as the mask-decoder head: "sam3" (default) or "sam2".
    # SAM 3 builds on the SAM 2 module layout, so the encode/decode wrapper is shared;
    # only the model-loading path differs (see SAMSegHead._load_sam).
    sam_version: str = "sam3"
    sam_name_or_path: str = "facebook/sam3"              # or a local MedSAM-3 checkpoint dir
    # Default checkpoint used when sam_version=="sam2" and sam_name_or_path is left at
    # the sam3 default (so switching versions with a single flag "just works").
    sam2_name_or_path: str = "facebook/sam2.1-hiera-large"
    seg_token: str = DEFAULT_SEG_TOKEN

    # image / mask geometry (must match the data config)
    # 3 selected MRI modalities -> SAM runs on its native 3-channel patch embed, so the
    # encoder is used unmodified (no channel inflation). Set to 4 to stack all modalities
    # and inflate the patch embed instead. MUST match the data config's seg_select_k.
    seg_in_channels: int = 3
    seg_image_size: int = 1024
    mask_size: int = 1024
    sam_prompt_dim: int = 256     # SAM sparse-prompt embedding dim
    sam_low_res: int = 256        # decoder mask logits resolution before upsampling

    # projection MLP (text hidden -> sam prompt)
    proj_hidden_dim: int = 1024
    proj_dropout: float = 0.0

    # what to train
    freeze_vlm: bool = True            # train via LoRA instead of full fine-tune
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    freeze_sam_image_encoder: bool = True
    train_sam_mask_decoder: bool = True

    # loss weights
    lm_weight: float = 0.1
    bce_weight: float = 1.0
    dice_weight: float = 2.0

    # mask pixel-classification loss — handles the small-lesion vs huge-background
    # imbalance so the loss isn't dominated by easy background pixels:
    #   "weighted" -> BCE with a foreground pos_weight (default; auto-balanced per batch)
    #   "focal"    -> focal loss (down-weights easy, well-classified pixels)
    #   "plain"    -> original unweighted BCE
    # (Dice is left as-is — it's already imbalance-robust.)
    mask_bce_mode: str = "weighted"
    bce_pos_weight: Optional[float] = None   # "weighted": fixed value, or None = auto (neg/pos)
    bce_pos_weight_cap: float = 100.0        # cap for the auto pos_weight (avoids blow-up if pos~0)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # dtype
    torch_dtype: str = "bfloat16"


# ============================================================
# 1. Mask losses (LISA-style, per-mask then averaged)
# ============================================================

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pred_logits, target: (N, H, W). Returns scalar mean Dice loss."""
    pred = pred_logits.sigmoid().flatten(1)
    tgt = target.flatten(1)
    num = 2 * (pred * tgt).sum(-1)
    den = pred.sum(-1) + tgt.sum(-1) + eps
    return (1 - (num + eps) / den).mean()


def sigmoid_bce_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                     pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """pred_logits, target: (N, H, W). Optional scalar pos_weight upweights the foreground
    (positive) class to counter background dominance. Returns scalar mean BCE."""
    return F.binary_cross_entropy_with_logits(
        pred_logits, target, pos_weight=pos_weight, reduction="none"
    ).flatten(1).mean()


def sigmoid_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """pred_logits, target: (N, H, W). Focal loss (Lin et al. 2017): scales each pixel's
    BCE by (1 - p_t)**gamma, so easy well-classified pixels — the vast background —
    contribute little while hard/foreground pixels dominate. alpha balances pos/neg.
    Background is still supervised (not masked out), just down-weighted. Scalar mean."""
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p = pred_logits.sigmoid()
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss.flatten(1).mean()


# ============================================================
# 2. SAM segmentation head  (VERSION-DEPENDENT WRAPPER: SAM 2 / SAM 3)
# ============================================================

def _adapt_first_conv_to_n_channels(module: nn.Module, n_in: int) -> Optional[nn.Module]:
    """
    Replace the first Conv2d (patch embed) of `module` with an n_in-channel conv,
    copying pretrained weights for the first min(3, n_in) channels and initializing
    extra channels with the mean of the originals. Returns the new conv if a
    3-channel conv was found, else None.
    """
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            old = child
            new = nn.Conv2d(
                n_in, old.out_channels, old.kernel_size, old.stride, old.padding,
                bias=old.bias is not None,
            ).to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                w = old.weight  # (out, 3, kh, kw)
                if n_in >= 3:
                    new.weight[:, :3] = w
                    if n_in > 3:
                        new.weight[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, n_in - 3, 1, 1)
                else:
                    new.weight[:] = w[:, :n_in]
                if old.bias is not None:
                    new.bias[:] = old.bias
            # set the new conv back on its parent
            parent = module
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, new)
            return new
    return None


class SAMSegHead(nn.Module):
    """
    Thin wrapper around a SAM 2 / SAM 3 model exposing exactly two operations
    needed for embedding-as-mask:

        encode_image(images_seg)              -> image features (+ positional enc)
        decode(features, sparse_prompt_embed) -> low-res mask logits

    SAM 3 builds on the SAM 2 module layout, so `encode_image` / `decode` are
    shared across both versions. Only model construction is version-specific:
    `_load_sam` dispatches on `cfg.sam_version` ("sam2" or "sam3"). If your
    installed package's attribute names differ, edit the accessors / loaders below.
    """

    def __init__(self, cfg: BrainPerceptionModelConfig):
        super().__init__()
        self.cfg = cfg
        self.sam = self._load_sam(cfg)

        # adapt the image encoder's patch embed to 4-channel MRI input
        self._inflated_conv: Optional[nn.Module] = None
        if cfg.seg_in_channels != 3:
            self._inflated_conv = _adapt_first_conv_to_n_channels(
                self._image_encoder(), cfg.seg_in_channels
            )
            if self._inflated_conv is None:
                raise RuntimeError(
                    "Could not locate a 3-channel patch-embed Conv2d to inflate to "
                    f"{cfg.seg_in_channels} channels. Adapt SAM3SegHead._image_encoder()."
                )

        if cfg.freeze_sam_image_encoder:
            for p in self._image_encoder().parameters():
                p.requires_grad_(False)
            # Keep the inflated patch-embed trainable even with a frozen encoder, so the
            # extra MRI-modality channels (initialized as the mean of the RGB filters)
            # can actually be learned instead of staying fixed at their init forever.
            if self._inflated_conv is not None:
                for p in self._inflated_conv.parameters():
                    p.requires_grad_(True)
        if not cfg.train_sam_mask_decoder:
            for p in self._mask_decoder().parameters():
                p.requires_grad_(False)

    # ---- construct the SAM model (dispatch on cfg.sam_version) ----
    @staticmethod
    def _resolve_sam_name(cfg: BrainPerceptionModelConfig) -> str:
        """Pick the checkpoint id. When sam_version=='sam2' but sam_name_or_path is
        still the sam3 default, fall back to the configured SAM 2 default so a single
        --sam_version flag does not accidentally load SAM 3 weights."""
        version = (cfg.sam_version or "sam3").lower()
        sam3_default = BrainPerceptionModelConfig.sam_name_or_path  # "facebook/sam3"
        if version == "sam2" and cfg.sam_name_or_path == sam3_default:
            return cfg.sam2_name_or_path
        return cfg.sam_name_or_path

    def _load_sam(self, cfg: BrainPerceptionModelConfig) -> nn.Module:
        version = (cfg.sam_version or "sam3").lower()
        if version == "sam3":
            return self._load_sam3(cfg)
        if version == "sam2":
            return self._load_sam2(cfg)
        raise ValueError(f"Unknown sam_version {cfg.sam_version!r}; expected 'sam2' or 'sam3'.")

    def _load_sam3(self, cfg: BrainPerceptionModelConfig) -> nn.Module:
        name = self._resolve_sam_name(cfg)
        try:
            from sam3.build_sam import build_sam3  # type: ignore
            return build_sam3(name)
        except Exception as e:  # fall back to HF transformers if packaged there
            try:
                from transformers import Sam3Model  # type: ignore
                return Sam3Model.from_pretrained(name)
            except Exception:
                raise ImportError(
                    "Could not load SAM 3. Install Meta's `sam3` package "
                    "(github.com/facebookresearch/sam3) or a transformers build that "
                    f"exposes Sam3Model, then set sam_name_or_path. Original error: {e}"
                )

    def _load_sam2(self, cfg: BrainPerceptionModelConfig) -> nn.Module:
        name = self._resolve_sam_name(cfg)
        try:
            # Meta's sam2 package: builds a native model directly from an HF hub id,
            # giving the SAM2 module layout this wrapper targets.
            from sam2.build_sam import build_sam2_hf  # type: ignore
            return build_sam2_hf(name)
        except Exception as e:  # fall back to HF transformers if packaged there
            try:
                from transformers import Sam2Model  # type: ignore
                return Sam2Model.from_pretrained(name)
            except Exception:
                raise ImportError(
                    "Could not load SAM 2. Install Meta's `sam2` package "
                    "(github.com/facebookresearch/sam2) or a transformers build that "
                    f"exposes Sam2Model, then set sam_name_or_path (e.g. "
                    f"'facebook/sam2.1-hiera-large'). Original error: {e}"
                )

    # submodule accessors (adapt attribute names to your sam3 build)
    def _image_encoder(self) -> nn.Module:
        return getattr(self.sam, "image_encoder", getattr(self.sam, "vision_encoder", self.sam))

    def _prompt_encoder(self) -> nn.Module:
        return getattr(self.sam, "sam_prompt_encoder", getattr(self.sam, "prompt_encoder"))

    def _mask_decoder(self) -> nn.Module:
        return getattr(self.sam, "sam_mask_decoder", getattr(self.sam, "mask_decoder"))

    # ---- image features ----
    # Field names the image embedding / FPN features go by across SAM 1/2/3 builds
    # (native packages return a dict; HF builds return a ModelOutput). The embed is the
    # final low-res feature map (B, c, h, w); high-res feats are the FPN list (optional).
    _EMBED_KEYS = ("vision_features", "image_embed", "image_embeds",
                   "image_embeddings", "last_hidden_state")
    _HIRES_KEYS = ("backbone_fpn", "high_res_feats", "high_res_features")

    @staticmethod
    def _normalize_encoder_output(out):
        """Pull (image_embed, high_res_feats) out of whatever the encoder returned."""
        if isinstance(out, torch.Tensor):
            return out, None
        # dict OR transformers ModelOutput (both expose .get)
        getter = out.get if hasattr(out, "get") else None
        if getter is not None:
            embed = next((getter(k) for k in SAMSegHead._EMBED_KEYS if getter(k) is not None), None)
            hires = next((getter(k) for k in SAMSegHead._HIRES_KEYS if getter(k) is not None), None)
            return embed, hires
        # plain object with attributes
        embed = next((getattr(out, k, None) for k in SAMSegHead._EMBED_KEYS
                      if getattr(out, k, None) is not None), None)
        hires = next((getattr(out, k, None) for k in SAMSegHead._HIRES_KEYS
                      if getattr(out, k, None) is not None), None)
        return embed, hires

    def _is_hf_sam(self) -> bool:
        """True when self.sam is a 🤗 transformers Sam2/Sam-style model (different API
        from Meta's native sam2/sam3 packages)."""
        return hasattr(self.sam, "get_image_features") and hasattr(self.sam, "backbone_feature_sizes")

    def _encode_image_hf(self, images_seg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """transformers Sam2 path. Uses the model's own get_image_embeddings() so the
        internal feature-map layout (which varies across transformers versions) stays the
        model's responsibility, not ours. That method is decorated with @torch.no_grad(),
        which would freeze the trainable inflated patch-embed; we strip the decorator via
        __wrapped__ (when present) so gradients still flow. Returns image_embed=(B,c,h,w)
        (lowest-res level) and high_res_feats=the higher FPN levels the decoder upsamples
        through."""
        model = self.sam
        # Strip get_image_embeddings' @torch.no_grad() decorator (when present) so gradients
        # flow to whatever IS trainable here — the mask decoder's conv_s0/conv_s1 FPN
        # projections (they run inside get_image_embeddings) and any trainable patch embed.
        # The frozen hiera encoder still never backprops: its params require no grad and
        # images_seg has none, so autograd builds no graph through it regardless. Keeping
        # grad here therefore costs ~nothing but preserves the decoder's configured training.
        fn = getattr(type(model).get_image_embeddings, "__wrapped__", type(model).get_image_embeddings)
        image_embeddings = list(fn(model, images_seg))
        return {"image_embed": image_embeddings[-1], "high_res_feats": list(image_embeddings[:-1])}

    def encode_image(self, images_seg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """images_seg: (B, C, H, W) -> dict with 'image_embed' (B, c, h, w) and
        optional 'high_res_feats' list for the decoder's upsampling path."""
        if self._is_hf_sam():
            return self._encode_image_hf(images_seg)
        # ---- native SAM (Meta sam2/sam3 packages) ----
        enc = self._image_encoder()
        out = enc(images_seg)
        image_embed, high_res = self._normalize_encoder_output(out)
        if image_embed is None:
            # Fail loudly with the actual structure so the right field can be wired in,
            # instead of returning None and crashing later on image_embed[sample_index].
            if hasattr(out, "keys"):
                detail = f"a {type(out).__name__} with keys={list(out.keys())}"
            elif isinstance(out, (tuple, list)):
                detail = (f"a {type(out).__name__} of len {len(out)} with element types "
                          f"{[type(o).__name__ for o in out]}")
            else:
                attrs = [a for a in dir(out) if not a.startswith('_')][:40]
                detail = f"a {type(out).__name__} with attrs={attrs}"
            raise RuntimeError(
                "SAM image encoder produced no recognizable image-feature tensor. "
                f"_image_encoder() resolved to {type(enc).__name__}; its forward returned {detail}. "
                "Add the right field name to SAMSegHead._EMBED_KEYS (and _HIRES_KEYS for the "
                "FPN path), or fix _image_encoder() if it resolved to the wrong submodule."
            )
        return {"image_embed": image_embed, "high_res_feats": high_res}

    def _decode_hf(self, feats: Dict[str, torch.Tensor], sparse_prompt: torch.Tensor) -> torch.Tensor:
        """transformers Sam2 mask-decoder path. Treats each <SEG> as one image in the
        decoder's batch dim (N) with point_batch_size=1, injecting the projected <SEG>
        embedding as the single sparse prompt token (LISA-style 'embedding as mask')."""
        model = self.sam
        image_embed = feats["image_embed"]                       # (N, c, h, w)
        n, _, h, w = image_embed.shape

        # dense positional embedding (1,c,h,w) -> broadcast to the N "images"
        image_pe = model.get_image_wide_positional_embeddings().to(image_embed.dtype)
        image_pe = image_pe.expand(n, -1, -1, -1)
        # "no mask" dense prompt, broadcast over the feature grid
        no_mask = model.prompt_encoder.no_mask_embed.weight          # (1, hidden)
        dense_embed = no_mask.reshape(1, -1, 1, 1).expand(n, -1, h, w).to(image_embed.dtype)

        # sparse prompt -> (N, point_batch=1, num_points=1, hidden)
        sparse = sparse_prompt
        if sparse.dim() == 3:                                        # (N, 1, hidden)
            sparse = sparse.unsqueeze(1)                             # (N, 1, 1, hidden)

        candidate = dict(
            image_embeddings=image_embed,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense_embed,
            multimask_output=False,
            high_resolution_features=feats.get("high_res_feats"),
        )
        # Pass only what this transformers version's decoder accepts (kwarg names have
        # drifted across releases), so an extra/renamed optional arg can't TypeError.
        accepted = inspect.signature(model.mask_decoder.forward).parameters
        call = {k: v for k, v in candidate.items() if k in accepted}
        out = model.mask_decoder(**call)
        masks = out[0] if isinstance(out, (tuple, list)) else out
        # masks: (N, point_batch=1, num_masks=1, low, low) -> (N, 1, low, low)
        return masks[:, 0]

    def decode(self, feats: Dict[str, torch.Tensor], sparse_prompt: torch.Tensor) -> torch.Tensor:
        """
        feats: output of encode_image, with image_embed possibly expanded to (N,...)
        sparse_prompt: (N, 1, sam_prompt_dim)  one projected <SEG> embedding per mask.
        Returns low-res mask logits (N, 1, low, low).
        """
        if self._is_hf_sam():
            return self._decode_hf(feats, sparse_prompt)

        # ---- native SAM (Meta sam2/sam3 packages) ----
        prompt_enc = self._prompt_encoder()
        decoder = self._mask_decoder()

        image_embed = feats["image_embed"]                       # (N, c, h, w)
        image_pe = prompt_enc.get_dense_pe()                     # (1, c, h, w)
        n = image_embed.shape[0]
        dense_embed = prompt_enc.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            n, -1, image_embed.shape[-2], image_embed.shape[-1]
        )

        kwargs = dict(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,   # (N, 1, C)
            dense_prompt_embeddings=dense_embed,
            multimask_output=False,
            repeat_image=False,                       # SAM2/3: one prompt per image already
            high_res_features=feats.get("high_res_feats"),
        )
        # pass only the args this decoder version accepts (SAM1 vs SAM2/3 differ)
        accepted = inspect.signature(decoder.forward).parameters
        call = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
        out = decoder(**call)
        low_res_masks = out[0] if isinstance(out, (tuple, list)) else out
        return low_res_masks  # (N, 1, low, low)


# Backward-compatible alias: the head used to be SAM3-only.
SAM3SegHead = SAMSegHead


def build_sam_seg_head(cfg: BrainPerceptionModelConfig) -> SAMSegHead:
    """Construct the SAM 2 / SAM 3 segmentation head selected by cfg.sam_version."""
    return SAMSegHead(cfg)


# ============================================================
# 3. Qwen3-VL loader  (VERSION-DEPENDENT)
# ============================================================

def resolve_vlm_name_or_path(cfg: BrainPerceptionModelConfig) -> str:
    """Resolve the VLM id/path for transformers.from_pretrained()."""
    name = cfg.vlm_name_or_path
    local_path = Path(name).expanduser()
    if local_path.exists():
        return str(local_path)

    source = (cfg.vlm_source or "huggingface").lower()
    if source in {"huggingface", "hf"}:
        return name
    if source not in {"modelscope", "ms"}:
        raise ValueError(
            f"Unknown vlm_source={cfg.vlm_source!r}; expected 'huggingface' or 'modelscope'."
        )

    try:
        from modelscope import snapshot_download  # type: ignore
    except ImportError as e:
        raise ImportError(
            "vlm_source='modelscope' requires the ModelScope Python package. "
            "Install it with `pip install modelscope`, or use --vlm_source huggingface."
        ) from e

    kwargs = {}
    if cfg.vlm_revision:
        kwargs["revision"] = cfg.vlm_revision
    if cfg.vlm_cache_dir:
        kwargs["cache_dir"] = cfg.vlm_cache_dir
    return snapshot_download(name, **kwargs)


def load_qwen3_vl(cfg: BrainPerceptionModelConfig):
    """Returns (model, processor). Adds <SEG> and resizes embeddings."""
    from transformers import AutoProcessor

    dtype = getattr(torch, cfg.torch_dtype)
    try:
        from transformers import Qwen3VLForConditionalGeneration as _VLM  # type: ignore
    except Exception:
        from transformers import AutoModelForImageTextToText as _VLM  # type: ignore

    vlm_path = resolve_vlm_name_or_path(cfg)
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    model = _VLM.from_pretrained(vlm_path, torch_dtype=dtype, trust_remote_code=True)

    tok = processor.tokenizer
    if tok.convert_tokens_to_ids(cfg.seg_token) == tok.unk_token_id:
        tok.add_tokens([cfg.seg_token], special_tokens=True)
        model.resize_token_embeddings(len(tok))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # NOTE: a *-Base checkpoint may not define a chat template. The collator below
    # relies on processor.apply_chat_template. If you truly want the base model,
    # set processor.tokenizer.chat_template to a Qwen-im_start/im_end template, or
    # switch to the *-Instruct checkpoint (recommended).
    return model, processor


def maybe_wrap_lora(model, cfg: BrainPerceptionModelConfig):
    if not cfg.use_lora:
        return model
    import dataclasses
    from peft import LoraConfig, get_peft_model

    lconf_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        # Train the input/output embedding rows for the newly added <SEG> token.
        # resize_token_embeddings() runs before this wrap, so without modules_to_save
        # PEFT would freeze the <SEG> rows at init and the model could not learn to
        # emit <SEG> autoregressively (LISA-style fix).
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Qwen3-VL has tie_word_embeddings=True, so embed_tokens and lm_head share weights.
    # Both are in modules_to_save, so keep PEFT tying them during training — otherwise
    # PEFT trains two independent copies (silently untying them) and warns about it.
    # Guard for older PEFT that lacks this field.
    if any(f.name == "ensure_weight_tying" for f in dataclasses.fields(LoraConfig)):
        lconf_kwargs["ensure_weight_tying"] = True
    lconf = LoraConfig(**lconf_kwargs)
    peft_model = get_peft_model(model, lconf)
    # Show which modules actually received LoRA adapters so you can confirm the
    # target_modules matched the intended (text) tower and not the vision tower.
    adapted = sorted({
        n.rsplit(".lora_", 1)[0].split(".")[-1]
        for n, _ in peft_model.named_parameters() if ".lora_" in n
    })
    print(f"[lora] adapted module types: {adapted}")
    print(f"[lora] modules_to_save: {list(lconf.modules_to_save)}")
    return peft_model


# ============================================================
# 4. The model
# ============================================================

class BrainPerceptionModel(nn.Module):
    def __init__(self, cfg: BrainPerceptionModelConfig, vlm=None, processor=None, sam_head=None):
        super().__init__()
        self.cfg = cfg
        if vlm is None or processor is None:
            vlm, processor = load_qwen3_vl(cfg)
        self.processor = processor
        self.seg_token_id = processor.tokenizer.convert_tokens_to_ids(cfg.seg_token)

        if cfg.freeze_vlm and not cfg.use_lora:
            for p in vlm.parameters():
                p.requires_grad_(False)
        self.vlm = maybe_wrap_lora(vlm, cfg) if cfg.use_lora else vlm

        text_dim = self._vlm_hidden_size()
        self.seg_projection = nn.Sequential(
            nn.Linear(text_dim, cfg.proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.proj_dropout),
            nn.Linear(cfg.proj_hidden_dim, cfg.sam_prompt_dim),
        )

        # sam_head can be injected (e.g. a mock for CPU shape tests) to avoid loading SAM.
        self.sam_head = sam_head if sam_head is not None else build_sam_seg_head(cfg)

    def _vlm_hidden_size(self) -> int:
        base = getattr(self.vlm, "base_model", self.vlm)
        cfgobj = base.config
        # Qwen3-VL nests the text config
        return getattr(cfgobj, "hidden_size", None) or cfgobj.text_config.hidden_size

    # ---- core: gather <SEG> hidden states across the batch ----
    def _gather_seg_embeddings(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, D); input_ids: (B, L). Returns (N_total, D) in batch-major,
        position order — the same order the collator concatenates GT masks."""
        seg_mask = input_ids == self.seg_token_id           # (B, L)
        return hidden[seg_mask]                              # (N_total, D)

    def _mask_ce_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Per-pixel classification loss for the masks, with small-lesion vs big-background
        imbalance handling selected by cfg.mask_bce_mode. pred, gt: (N, H, W) floats."""
        cfg = self.cfg
        mode = getattr(cfg, "mask_bce_mode", "plain")
        if mode == "focal":
            return sigmoid_focal_loss(pred, gt, cfg.focal_alpha, cfg.focal_gamma)
        if mode == "weighted":
            if cfg.bce_pos_weight is not None:
                pos_weight = torch.as_tensor(cfg.bce_pos_weight, device=gt.device, dtype=gt.dtype)
            else:
                # auto-balance: pos_weight = (#background / #foreground) so the (few)
                # foreground pixels contribute on par with the (many) background pixels.
                pos = gt.sum()
                neg = gt.numel() - pos
                pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=cfg.bce_pos_weight_cap)
            return sigmoid_bce_loss(pred, gt, pos_weight=pos_weight)
        return sigmoid_bce_loss(pred, gt)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = next(self.parameters()).device
        vlm_dtype = next(self.vlm.parameters()).dtype
        sam_dtype = next(self.sam_head.parameters()).dtype
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images_seg = batch["images_seg"].to(device)
        gt_masks = batch["masks"].to(device)                # (sum_R, Hm, Wm)
        # Force to CPU: it's compared against the CPU-side seg_counts below (torch.equal
        # needs both on the same device), and accelerate may have moved the batch to GPU.
        # It is explicitly .to(device)'d again where a CUDA copy is needed (sample_index).
        num_per_sample = batch["num_masks_per_sample"].cpu()

        # ---- 1. VLM forward (text loss + hidden states) ----
        vlm_inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        if "pixel_values" in batch:
            vlm_inputs["pixel_values"] = batch["pixel_values"].to(device, vlm_dtype)
        if "image_grid_thw" in batch:
            vlm_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        if "mm_token_type_ids" in batch:
            vlm_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

        out = self.vlm(**vlm_inputs)
        lm_loss = out.loss
        hidden = out.hidden_states[-1]                       # (B, L, D)

        # ---- 2. <SEG> -> prompt embeddings ----
        seg_embeds = self._gather_seg_embeddings(hidden, input_ids)   # (N, D)
        n_seg = seg_embeds.shape[0]
        n_masks = gt_masks.shape[0]
        # per-sample alignment guard: the image->mask mapping (sample_index) below
        # assumes each sample's <SEG> count equals its mask count, in batch order.
        seg_counts = (input_ids == self.seg_token_id).sum(dim=1).cpu()
        if n_seg != n_masks or not torch.equal(seg_counts, num_per_sample):
            raise RuntimeError(
                f"<SEG>/mask mismatch: total <SEG>={n_seg}, masks={n_masks}, "
                f"per-sample <SEG>={seg_counts.tolist()}, masks={num_per_sample.tolist()}."
            )
        if n_seg == 0:
            zero = images_seg.sum() * 0.0
            return {"loss": lm_loss, "lm_loss": lm_loss.detach(), "mask_loss": zero, "n_masks": 0}

        sparse_prompt = self.seg_projection(seg_embeds.to(self.seg_projection[0].weight.dtype))
        sparse_prompt = sparse_prompt.unsqueeze(1)           # (N, 1, sam_prompt_dim)

        # ---- 3. SAM image features, expanded per <SEG> ----
        feats = self.sam_head.encode_image(images_seg.to(sam_dtype))
        image_embed = feats["image_embed"]                   # (B, c, h, w)
        # map each of the N seg tokens to its source image in the batch
        sample_index = torch.repeat_interleave(
            torch.arange(images_seg.shape[0], device=device), num_per_sample.to(device)
        )                                                    # (N,)
        feats_exp = {
            "image_embed": image_embed[sample_index],
            "high_res_feats": (
                [f[sample_index] for f in feats["high_res_feats"]]
                if feats.get("high_res_feats") is not None else None
            ),
        }

        # ---- 4. decode + mask loss ----
        low_res = self.sam_head.decode(feats_exp, sparse_prompt.to(image_embed.dtype))  # (N,1,low,low)
        pred = F.interpolate(low_res, size=(cfg.mask_size, cfg.mask_size),
                             mode="bilinear", align_corners=False).squeeze(1)            # (N,Hm,Wm)
        pred = pred.float()
        gt = gt_masks.float()

        l_bce = self._mask_ce_loss(pred, gt)
        l_dice = dice_loss(pred, gt)
        mask_loss = cfg.bce_weight * l_bce + cfg.dice_weight * l_dice
        loss = cfg.lm_weight * lm_loss + mask_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss.detach(),
            "bce_loss": l_bce.detach(),
            "dice_loss": l_dice.detach(),
            "mask_loss": mask_loss.detach(),
            "n_masks": n_seg,
        }

    @torch.no_grad()
    def trainable_parameter_report(self) -> str:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"trainable {tr/1e6:.1f}M / total {tot/1e6:.1f}M ({100*tr/tot:.2f}%)"


# ============================================================
# 5. Qwen3-VL collator: builds processor inputs + masked labels + SAM tensors
# ============================================================

@dataclass
class Qwen3VLSegCollator:
    """
    Turns raw dataset samples (rounds + vlm_image_path + images_seg + masks) into a
    training batch for BrainPerceptionModel.

    Builds, via the Qwen3-VL processor:
        input_ids, attention_mask, pixel_values, image_grid_thw
    and a `labels` tensor that supervises ONLY assistant spans (so the model learns
    to emit the answer text and the <SEG> token). The image goes only in round 1.
    """
    processor: Any
    system_prompt: str = (
        "You are a brain MRI perception assistant. Locate the requested target and "
        "answer with a segmentation token."
    )

    def _messages(self, rounds: List[dict], image_path: str) -> List[dict]:
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        for i, rd in enumerate(rounds):
            user_content = []
            if i == 0:
                user_content.append({"type": "image", "image": image_path})
            user_content.append({"type": "text", "text": str(rd["question"])})
            msgs.append({"role": "user", "content": user_content})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(rd["answer"])}]})
        return msgs

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info  # ships with Qwen-VL

        proc = self.processor
        all_messages = [self._messages(b["rounds"], b["vlm_image_path"]) for b in batch]

        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in all_messages]
        image_inputs, _ = process_vision_info(all_messages)

        # return_mm_token_type_ids=True: newer Qwen3-VL (transformers >=4.57) needs
        # mm_token_type_ids to compute M-RoPE 3D position ids; the model raises without it.
        enc = proc(text=texts, images=image_inputs, padding=True, return_tensors="pt",
                   return_mm_token_type_ids=True)
        labels = self._mask_labels(enc["input_ids"])
        enc["labels"] = labels

        # SAM-branch tensors
        enc["images_seg"] = torch.stack([b["images_seg"] for b in batch], dim=0)
        enc["masks"] = torch.cat([b["masks"] for b in batch], dim=0)
        enc["num_masks_per_sample"] = torch.tensor([b["masks"].shape[0] for b in batch], dtype=torch.long)
        enc["conversation_ids"] = [b["conversation_id"] for b in batch]
        return dict(enc)

    def _mask_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Supervise only assistant content. Qwen wraps turns as
            <|im_start|>{role}\n {content}<|im_end|>
        For each turn we read the role line by *decoding* tokens until a newline
        (robust to the role spanning multiple tokens, which convert_tokens_to_ids
        would silently miss). If role == 'assistant' we unmask the content up to and
        including <|im_end|> so the model learns the answer text, the <SEG> token, and
        when to stop.
        """
        tok = self.processor.tokenizer
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if im_start is None or im_start == tok.unk_token_id:
            raise RuntimeError(
                "Tokenizer has no <|im_start|> token; the *-Base checkpoint likely lacks "
                "a chat template. Use the -Instruct model or set a Qwen chat template."
            )

        n_supervised = 0
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            n = len(ids)
            i = 0
            while i < n:
                if ids[i] != im_start:
                    i += 1
                    continue
                # decode the role line: tokens after im_start until one yields a newline
                j = i + 1
                role_tokens: List[int] = []
                while j < n:
                    role_tokens.append(ids[j])
                    piece = tok.decode([ids[j]])
                    j += 1
                    if "\n" in piece:
                        break
                role = tok.decode(role_tokens).strip()
                if role == "assistant":
                    k = j
                    while k < n and ids[k] != im_end:
                        k += 1
                    end = min(k, n - 1)
                    labels[b, j:end + 1] = input_ids[b, j:end + 1]   # include <|im_end|>
                    n_supervised += end + 1 - j
                    i = end + 1
                else:
                    i = j
        if n_supervised == 0:
            raise RuntimeError(
                "No assistant tokens were supervised — chat-template role markers did not "
                "match. Verify the processor's template before training."
            )
        return labels


# ============================================================
# 6. Build helpers
# ============================================================

def build_model_and_collator(
    model_cfg: BrainPerceptionModelConfig,
) -> Tuple[BrainPerceptionModel, Qwen3VLSegCollator]:
    model = BrainPerceptionModel(model_cfg)
    collator = Qwen3VLSegCollator(processor=model.processor)
    return model, collator


def build_dataset_for_vlm(data_cfg: BrainPerceptionDataConfig) -> BrainPerceptionDataset:
    # tokenizer=None -> raw mode; the Qwen collator does tokenization.
    data_cfg.build_clip_branch = False
    return BrainPerceptionDataset(data_cfg, tokenizer=None)


# ============================================================
# 7. Overfit smoke test (run in your training env)
# ============================================================

def _overfit_smoke(root: str, steps: int = 30, batch_size: int = 2) -> None:
    from torch.utils.data import DataLoader

    model_cfg = BrainPerceptionModelConfig()
    data_cfg = BrainPerceptionDataConfig(dataset_root=root, build_clip_branch=False, max_rounds=2)

    model, collator = build_model_and_collator(model_cfg)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    print(model.trainable_parameter_report())

    ds = build_dataset_for_vlm(data_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collator)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    it = iter(loader)
    fixed = next(it)  # overfit a single batch
    for step in range(steps):
        out = model(fixed)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step:03d} | loss {out['loss'].item():.4f} "
                  f"| lm {out['lm_loss'].item():.4f} | mask {out['mask_loss'].item():.4f} "
                  f"| n_masks {out['n_masks']}")
    print("[overfit smoke] done — loss should be trending down.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (e.g. .../segllm_10samples)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()
    _overfit_smoke(args.root, args.steps, args.batch_size)
