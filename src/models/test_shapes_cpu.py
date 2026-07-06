#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_shapes_cpu.py

CPU-only plumbing test for BrainPerceptionModel. It injects a tiny fake VLM and a
fake SAM head so NO 4B / SAM weights are downloaded — it exercises only the custom
"embedding-as-mask" logic in BrainPerceptionModel.forward:

  * gathering <SEG> hidden states in batch-major / round order,
  * the text-hidden -> SAM-prompt projection,
  * the per-<SEG> image-feature expansion (sample_index),
  * the decode -> upsample -> BCE/Dice path,
  * the per-sample <SEG>/mask alignment guard.

Requires only torch. Run:
    pip install torch            # CPU build is fine
    PYTHONPATH=$PWD python src/models/test_shapes_cpu.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/Users/albert_we/Workspaces/BrainPerception")
from src.dataset.brain_perception_dataset import IGNORE_INDEX
from src.models.brain_perception_model import BrainPerceptionModel, BrainPerceptionModelConfig

# small synthetic vocabulary
HIDDEN = 64
VOCAB = 256
SEG_ID = 200  # distinct from the random "content" ids we sample below (< 150)


# ============================================================
# Fakes
# ============================================================

class _FakeTokenizer:
    unk_token_id = 0
    pad_token_id = 1
    eos_token = "</s>"

    def convert_tokens_to_ids(self, token: str) -> int:
        return SEG_ID


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class FakeVLM(nn.Module):
    """Embedding + linear LM head. Returns .loss and .hidden_states like an HF VLM."""

    def __init__(self, hidden=HIDDEN, vocab=VOCAB):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, return_dict=True, **kw):
        h = self.embed(input_ids)                      # (B, L, D)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=(h,))


class FakeSAMHead(nn.Module):
    """Mimics SAM3SegHead.encode_image / decode with the right shapes."""

    def __init__(self, cfg: BrainPerceptionModelConfig):
        super().__init__()
        self.low = cfg.sam_low_res
        self.enc = nn.Conv2d(cfg.seg_in_channels, 8, kernel_size=3, padding=1)
        self.dec = nn.Linear(cfg.sam_prompt_dim, self.low * self.low)

    def encode_image(self, images_seg):
        f = self.enc(images_seg)                                   # (B, 8, H, W)
        f = F.adaptive_avg_pool2d(f, (self.low, self.low))        # (B, 8, low, low)
        return {"image_embed": f, "high_res_feats": None}

    def decode(self, feats, sparse_prompt):
        n = sparse_prompt.shape[0]
        m = self.dec(sparse_prompt.squeeze(1)).view(n, 1, self.low, self.low)
        # mix in image features so the (fake) encoder receives gradient too
        m = m + feats["image_embed"].mean(dim=1, keepdim=True)
        return m                                                   # (N, 1, low, low)


# ============================================================
# Synthetic batch
# ============================================================

def make_batch(cfg: BrainPerceptionModelConfig, seg_per_sample=(1, 2), seq_len=16):
    b = len(seg_per_sample)
    input_ids = torch.randint(0, 150, (b, seq_len))   # content ids stay < SEG_ID
    attention_mask = torch.ones(b, seq_len, dtype=torch.long)
    labels = torch.full((b, seq_len), IGNORE_INDEX)

    # place the right number of <SEG> per sample at deterministic positions
    positions = [[3], [4, 9]]
    assert [len(p) for p in positions] == list(seg_per_sample)
    for bi, ps in enumerate(positions):
        for p in ps:
            input_ids[bi, p] = SEG_ID
            labels[bi, p] = SEG_ID                    # supervise the seg token
        labels[bi, ps[0] - 1] = int(input_ids[bi, ps[0] - 1])  # supervise one more token

    n_masks = sum(seg_per_sample)
    masks = (torch.rand(n_masks, cfg.mask_size, cfg.mask_size) > 0.5).float()
    images_seg = torch.rand(b, cfg.seg_in_channels, cfg.seg_image_size, cfg.seg_image_size)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images_seg": images_seg,
        "masks": masks,
        "num_masks_per_sample": torch.tensor(seg_per_sample, dtype=torch.long),
    }


# ============================================================
# Tests
# ============================================================

def build_model(cfg):
    return BrainPerceptionModel(
        cfg, vlm=FakeVLM(), processor=_FakeProcessor(), sam_head=FakeSAMHead(cfg)
    )


def test_forward_backward():
    torch.manual_seed(0)
    cfg = BrainPerceptionModelConfig(
        seg_in_channels=4, seg_image_size=64, mask_size=64, sam_low_res=16,
        sam_prompt_dim=256, proj_hidden_dim=64, use_lora=False, freeze_vlm=False,
        torch_dtype="float32",
    )
    model = build_model(cfg)
    batch = make_batch(cfg, seg_per_sample=(1, 2))

    out = model(batch)
    loss = out["loss"]
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert out["n_masks"] == 3, out["n_masks"]
    for k in ("lm_loss", "bce_loss", "dice_loss", "mask_loss"):
        assert k in out, f"missing {k}"

    loss.backward()
    # gradient must reach: VLM embedding, projection, and SAM decoder
    assert model.vlm.embed.weight.grad is not None, "no grad into VLM"
    assert model.seg_projection[0].weight.grad is not None, "no grad into projection"
    assert model.sam_head.dec.weight.grad is not None, "no grad into SAM decoder"
    print(f"[ok] forward/backward  loss={loss.item():.4f}  "
          f"lm={out['lm_loss'].item():.4f}  bce={out['bce_loss'].item():.4f}  "
          f"dice={out['dice_loss'].item():.4f}  n_masks={out['n_masks']}")


def test_alignment_guard():
    cfg = BrainPerceptionModelConfig(
        seg_in_channels=4, seg_image_size=64, mask_size=64, sam_low_res=16,
        sam_prompt_dim=256, proj_hidden_dim=64, use_lora=False, freeze_vlm=False,
        torch_dtype="float32",
    )
    model = build_model(cfg)
    batch = make_batch(cfg, seg_per_sample=(1, 2))
    # corrupt the count: claim 1 mask for sample 2 while it has 2 <SEG>
    batch["masks"] = batch["masks"][:2]
    batch["num_masks_per_sample"] = torch.tensor([1, 1], dtype=torch.long)
    raised = False
    try:
        model(batch)
    except RuntimeError as e:
        raised = "<SEG>/mask mismatch" in str(e)
    assert raised, "alignment guard did not fire on mismatched counts"
    print("[ok] alignment guard fires on <SEG>/mask mismatch")


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = BrainPerceptionModelConfig(
        seg_in_channels=4, seg_image_size=64, mask_size=64, sam_low_res=16,
        sam_prompt_dim=256, proj_hidden_dim=64, use_lora=False, freeze_vlm=False,
        torch_dtype="float32",
    )
    model = build_model(cfg)
    batch = make_batch(cfg, seg_per_sample=(1, 2))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    first = last = None
    for step in range(40):
        out = model(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step == 0:
            first = out["loss"].item()
        last = out["loss"].item()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    print(f"[ok] overfit single batch  loss {first:.4f} -> {last:.4f}")


if __name__ == "__main__":
    test_forward_backward()
    test_alignment_guard()
    test_overfit_single_batch()
    print("\nALL CPU SHAPE TESTS PASSED")
