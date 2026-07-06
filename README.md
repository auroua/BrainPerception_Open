# Brain Perception Multimodal Large Language Models

LISA / MedSeg-R style **"embedding-as-mask"** brain-MRI perception model.

A MLLMs (Qwen3-VL-4B) answers multi-round questions about a brain
MRI and emits a special `<SEG>` token. The hidden state at each `<SEG>` position is
projected into a SAM prompt embedding, which the SAM mask decoder turns into a binary
segmentation mask.

```
Qwen3-VL-4B  (vision + language)
    │  hidden state at each <SEG> token
    ▼
seg_projection (MLP)  ->  SAM prompt embedding
    │
    ▼
SAM mask decoder  (features from images_seg)  ->  binary mask logits
```

Training signal:

```
L = lm_weight * CE(text) + bce_weight * BCE(mask) + dice_weight * Dice(mask)
```

**What is trained vs. frozen**

- **LLM** — trained with **LoRA** on the text-tower attention projections
  (`q/k/v/o_proj`); the new `<SEG>` input/output embeddings (`embed_tokens`, `lm_head`)
  are also trained so the model can learn to emit `<SEG>`.
- **SAM image encoder** — **frozen and used unmodified**. `images_seg` stacks **3 of the
  4 MRI modalities** (`t1n, t1c, t2w, t2f`), randomly selected per sample, so SAM runs
  on its native **3-channel** patch embed (no channel inflation). Because the encoder is
  frozen and its input carries no gradient, no backward pass runs through it — fast and
  memory-light. *(Optional: set `seg_in_channels=4` to stack all modalities and inflate
  the patch embed instead.)*
- **SAM mask decoder** — fine-tuned.

Either **SAM 2** or **SAM 3** can be the mask head (`--sam_version`). SAM 3 builds on
the SAM 2 layout, so only model loading differs; the encode/decode wrapper is shared.

---

## 1. Environment setup

```bash
pip install -r requirements.txt          # torch, transformers, peft, accelerate, qwen-vl-utils, ...
```

- **VLM branch** needs `transformers` (new enough to expose
  `Qwen3VLForConditionalGeneration`), `qwen_vl_utils`, and `peft`. Tested with
  Python 3.12 / transformers 5.12.1 (see `requirements.txt` for exact pins).
- **Mask head**:
  - **SAM 2** (default that works out of the box): if Meta's native `sam2` package is
    absent, the loader falls back to `transformers.Sam2Model` — no extra install needed.
    Use `--sam_version sam2`.
  - **SAM 3**: needs Meta's `sam3` package (not on PyPI):
    `pip install "git+https://github.com/facebookresearch/sam3"`, then `--sam_version sam3`.
- **Multi-GPU** training uses 🤗 `accelerate` (in `requirements.txt`).
- **`--wandb`** logging needs `pip install wandb` and `wandb login`.
- **Step 1 (BrainParc)** additionally needs the BrainParc + AutoBET pretrained models
  on disk (see the `*_MODEL` constants in `1_run_brainparc.py`).

Set `PYTHONPATH` to the repo root for every command below:

```bash
cd BrainPerception
export PYTHONPATH=$PWD
```

---

## Downloads

The released checkpoints, datasets, and BrainParc-generated masks are available from
Baidu Netdisk:

| Resource | Baidu Netdisk link | Extraction code | Notes |
|----------|---------------------|-----------------|-------|
| Trained models | [Download](https://pan.baidu.com/s/139PIpbN6x35yV8i-zY4CQQ) | `ks5e` | Trained BrainPerception model checkpoints |
| Training dataset | [Download](https://pan.baidu.com/s/1_mNKlx52jCB2ima5tJbxuQ) | `6pw6` | Training split used by the model |
| Test dataset | [Download](https://pan.baidu.com/s/1IEN4sKLIDWCGcDQ3vFWNIA) | `inhj` | Held-out test split |
| BrainParc segmented masks | [Download](https://pan.baidu.com/s/1B188WRPmZEYOmJusY8D6Aw) | `yq6h` | BrainParc model segmented masks |

---

## 2. Data generation pipeline (steps 1 → 5)

Turns raw BraTS-style MRI volumes into the 2D dialogue dataset the trainer reads.
**Run in order.** Steps 2–5 read paths from `src/dataset/data_pathes.py`; **step 1 has
its paths hardcoded at the top of its own file.**

### Configure paths — `src/dataset/data_pathes.py`

```python
original_instance_dir = ".../original_data"          # raw MRI volumes (*-t1n.nii.gz, ...)
instance_seg_root_dir = ".../brainparc_seg_folder"   # BrainParc output (step 1 OUT_ROOT)
instance_out_dir      = ".../BrainPerception_2D"     # final dataset (== training --root)
```

| Step                        | Command | Produces |
|-----------------------------|---------|----------|
| 1. BrainParc seg            | `python src/dataset/1_run_brainparc.py` | `tissue.nii.gz`, `dk-struct.nii.gz`, `present_labels.csv` per case |
| 2. Slice 3D→2D              | `python src/dataset/2_slice_brainparc_2d.py` | `images/<case>/*.png`, `masks/<case>/…`, `manifests/<case>.jsonl` |
| 3. Build dialogues          | `python src/dataset/3_build_multiround_dataset.py` | `multiround_dataset/multiround_dialogues.jsonl` + binary GT `masks/` |
| 4. De-duplicate             | `python src/dataset/4_remove_duplicate.py` | `multiround_dataset/multiround_dialogues.dedup.jsonl` |
| 5. Split train / test       | `python src/dataset/5_extract_training_dialogues.py` | `multiround_dialogues[.dedup].train.jsonl` + `….test.jsonl` (case-level split by `data/ids/*_case_ids.txt`) + `split_summary.json` |

The trainer **automatically prefers `multiround_dialogues.dedup.jsonl`** when it exists,
else the raw `multiround_dialogues.jsonl`. All image/mask paths inside the jsonl are
**relative to `instance_out_dir`**.

Step 5 only **splits** that deduped file into `.train.jsonl` / `.test.jsonl` by the
subject-id lists in `data/ids/` — it does not rename the auto-picked file. So the
trainer's default auto-pick still selects `multiround_dialogues.dedup.jsonl` (**all**
subjects). To train on the training subjects only, point the trainer at the split
explicitly: `--dialogues_rel multiround_dataset/multiround_dialogues.dedup.train.jsonl`.

### Resulting `dataset_root` (= `instance_out_dir`)

```
<dataset_root>/
  multiround_dataset/multiround_dialogues.dedup.jsonl   # trainer input (preferred)
  multiround_dataset/masks/<case>/*.png                 # binary 0/255 GT masks
  images/<case>/<case>-<mod>_<plane>_slice_<idx>.png
```

---

## 3. Train the model

Entry point: `tools/train_brain_perception.py`. Point `--root` at your `dataset_root`.
The trainer uses 🤗 accelerate, so the **same script** runs on 1 GPU or many.

### Single GPU

```bash
export PYTHONPATH=$PWD
python tools/train_brain_perception.py \
    --root /path/to/BrainPerception_2D \
    --out  runs/exp1 \
    --sam_version sam2 \
    --epochs 3 --batch_size 2 --grad_accum 8 \
    --lr 1e-4 --precision bf16
```

### Multi-GPU (e.g. 8 GPUs, one node)

```bash
export PYTHONPATH=$PWD
accelerate launch --num_processes 8 tools/train_brain_perception.py \
    --root /path/to/BrainPerception_2D \
    --out  runs/exp1 \
    --sam_version sam2 \
    --batch_size 6 --grad_accum 1 --precision bf16 \
    --wandb --wandb_project brain
# torchrun --nproc_per_node 8 ... also works
```

- To load Qwen3-VL from ModelScope, add
  `--vlm_source modelscope --vlm Qwen/Qwen3-VL-4B-Instruct`.

On startup it prints the LoRA wiring — confirm the adapted modules are the **text-tower**
attention projections only:

```
[lora] adapted module types: ['k_proj', 'o_proj', 'q_proj', 'v_proj']
[lora] modules_to_save: ['embed_tokens', 'lm_head']
```

Resume: `--resume runs/exp1/last.pt` (restores step/epoch/optimizer/scheduler).

### Key training flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--root` | `data_pathes.instance_out_dir` | `dataset_root` |
| `--out` | `runs/exp1` | checkpoint / log dir |
| `--vlm` | `Qwen/Qwen3-VL-4B-Instruct` | VLM checkpoint (use **-Instruct**, it has a chat template) |
| `--vlm_source` | `huggingface` | `huggingface` or `modelscope`; ModelScope resolves `--vlm` with `snapshot_download` |
| `--vlm_revision` / `--vlm_cache_dir` | `None` / `None` | Optional ModelScope snapshot revision/cache directory |
| `--sam_version` | `sam3` | `sam2` (works without extra install) or `sam3` |
| `--sam` | `facebook/sam3` | SAM checkpoint; with `--sam_version sam2` the SAM 2 default is used automatically |
| `--epochs` / `--max_steps` | `3` / `0` | `--max_steps > 0` overrides epochs |
| `--batch_size` / `--grad_accum` | `1` / `8` | per-process; effective batch = product × #GPUs |
| `--lr` / `--weight_decay` | `1e-4` / `0.0` | AdamW; cosine schedule with warmup |
| `--warmup_ratio` / `--grad_clip` | `0.03` / `1.0` | LR warmup fraction; grad-norm clip |
| `--precision` | `bf16` | `bf16` (recommended) / `fp16` (CUDA) / `fp32` |
| `--seg_size` / `--mask_size` | `1024` / `1024` | **SAM input must be 1024**; `mask_size` (GT/loss res) may be smaller |
| `--num_workers` | `4` | dataloader workers per process |
| `--val_ratio` | `0.2` | case-level val split (no slice leakage) |
| `--mask_bce_mode` | `weighted` | mask class-imbalance handling: `weighted` (BCE + auto foreground `pos_weight`), `focal`, or `plain` |
| `--bce_pos_weight` | `None` (auto) | fixed foreground weight for `weighted` mode |
| `--eval_every` / `--save_every` | `200` / `500` | optimizer-step cadence |
| `--log_every` / `--seed` | `10` / `20260427` | log cadence; RNG seed |
| `--wandb` / `--wandb_project` / `--wandb_run_name` / `--wandb_entity` | off / `brain-perception` / auto / auto | Weights & Biases logging |

LoRA rank/alpha/dropout, loss weights, focal α/γ, and freeze flags live in
`BrainPerceptionModelConfig` (`src/models/brain_perception_model.py`).

---

## 4. Evaluate the model

Evaluation is **two stages**:

1. **Inference** — `tools/tools_evaluate.py` loads a trained checkpoint and writes one
   predicted mask PNG per sample.
2. **Scoring** — `tools/calc_dice_hd95.py` (segmentation: Dice / HD95) and/or
   `tools/calc_detection_metrics.py` (box grounding: IoU / Acc@τ) read those masks and
   compare them to the GT masks named in the manifest.

All three scripts read the same **evaluation manifest** — a jsonl with one sample per
line. Each line needs at least:

```json
{"sample_id": "...", "image_path": ".../*-t1n_axial_slice_075.png",
 "gt_mask_path": ".../masks/<case>/....png", "prompt_for_model": "…question…",
 "task_type": "basic_segmentation", "target_type": "...", "target_name": "..."}
```

`image_path` and `prompt_for_model` drive inference; `gt_mask_path` is only used by the
scorers (and to pick the output resolution). The SAM-branch input is built from the **3
modalities `t1n, t1c, t2w`** (no `t2f`), derived from `image_path` by swapping the modality
tag — the deterministic first-3 selection also used for validation.

### 4.1 Inference → predicted masks

```bash
export PYTHONPATH=$PWD
python tools/tools_evaluate.py \
    --ckpt runs/exp1/last.pt \
    --manifest /path/to/eval_manifest.jsonl \
    --out_root runs/exp1/eval \
    --vlm Qwen/Qwen3-VL-4B-Instruct \
    --sam_version sam2 --sam facebook/sam2.1-hiera-large \
    --mode teacher_force
```

Use the **same `--vlm` / `--sam_version` / `--sam`** you trained with. Writes under
`--out_root`:

```
runs/exp1/eval/
  masks/<sample_id>.png        # predicted binary mask (0/255), sized to the GT mask
  predictions.jsonl            # one record per sample (status, reply, paths, task_type…)
  logs/run_summary.json        # counts: ok / no_seg / failed, elapsed, speed
```

**Two modes** (`--mode`):

| Mode | What it measures |
|------|------------------|
| `teacher_force` (default) | Feeds the canonical answer containing exactly one `<SEG>` (`--teacher_answer`) and reads the hidden state at that `<SEG>` — **mask quality in isolation**. |
| `generate` | The model produces its own answer and must emit `<SEG>` itself; if it emits none, an empty mask is saved (scores as a miss) — **end-to-end**. |

Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--ckpt` | *(required)* | trained checkpoint (e.g. `runs/exp1/last.pt`) |
| `--manifest` | *(required)* | evaluation jsonl |
| `--out_root` | *(required)* | output dir for masks / predictions / logs |
| `--vlm` / `--sam` | *(required)* | VLM and SAM checkpoints (match training) |
| `--sam_version` | `sam2` | `sam2` (works without extra install) or `sam3` |
| `--mode` | `teacher_force` | `teacher_force` or `generate` |
| `--mask_threshold` | `0.5` | sigmoid threshold for the binary mask |
| `--teacher_answer` | `好的，<SEG>，记为实例1。` | assistant answer injected in `teacher_force` (must contain `<SEG>`) |
| `--max_new_tokens` | `64` | generation budget in `--mode generate` |
| `--n_per_task` / `--max_samples` | `None` / `None` | cap samples per task / overall (for quick runs) |
| `--resume` | off | skip `sample_id`s already in `predictions.jsonl` and append |
| `--precision` | `bf16` | `bf16` / `fp16` / `fp32` |

### 4.2 Score segmentation — Dice / HD95

```bash
python tools/calc_dice_hd95.py \
    --manifest /path/to/eval_manifest.jsonl \
    --pred_mask_dir runs/exp1/eval/masks \
    --out_dir runs/exp1/eval/score_seg
```

Writes `detail_dice_hd95.csv` (per sample) plus `summary_overall.csv` and
`summary_by_{task,difficulty,target_type}.csv`, reporting **Dice** mean/std, **HD95** (in
pixels) mean/median, and the empty-prediction rate per group.

### 4.3 Score box grounding — IoU / Acc@τ

```bash
python tools/calc_detection_metrics.py \
    --manifest /path/to/eval_manifest.jsonl \
    --pred_mask_dir runs/exp1/eval/masks \
    --out_dir runs/exp1/eval/score_det \
    --method BrainPerception
```

Converts each predicted / GT mask to a tight `xyxy` box, counts a sample as **recognized**
when the boxes overlap (IoU > 0), and reports **IoU mean** and **Acc@0.1 / 0.3 / 0.5**
over recognized samples. Writes the per-sample and per-group CSVs plus a ready-to-paste
`table_detection_latex_row.txt`.

### Task difficulty groups

Both scorers bucket `task_type` into `regular` / `hard`:

| Difficulty | Tasks |
|------------|-------|
| `regular` | `basic_segmentation`, `tissue_to_region` |
| `hard` | `contralateral_same_region`, `same_side_same_lobe`, `spatial_named_region`, `tumor_to_overlapping_region` |

### Path remapping

The manifest stores **training-machine** paths (`/root/autodl-tmp/…`). To score on a
different server:

- **Scorers** (`calc_dice_hd95.py`, `calc_detection_metrics.py`) remap paths via the
  `PATH_REMAPS` list near the top of each file — edit it for your dataset root, or pass
  `--no_remap` to use the manifest paths as-is.
- **`tools_evaluate.py`** currently applies a **hardcoded** remap inside its loop (the two
  `item[...].replace(...)` lines) — edit those to your paths, or make the manifest already
  hold local paths.
