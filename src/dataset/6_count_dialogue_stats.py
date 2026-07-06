#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_dialogue_stats.py

Compute distribution statistics over a multi-round dialogue dataset produced by
3_build_multiround_dataset.py (optionally de-duplicated by 4_remove_duplicate.py)
and render two compact publication-quality pie charts with labels placed inside the wedges:

  1. Distribution of dialogues across conversation categories (conversation_type).
  2. Distribution of tissue occupancy (how often each tissue is a <SEG> target).

Each dialogue line has the schema:
    {
      "conversation_type": "basic_segmentation" | "tissue_to_region" | ...,
      "rounds": [
        {"target_type": "brainparc_region" | "brainparc_tissue" | "brats_tumor_union",
         "target_name": <str>, "target_label": <int>, ...}, ...
      ], ...
    }

Figures (vector PDF + 300-dpi PNG) are written to --outdir:
    fig_dialogue_categories_pie.{pdf,png}
    fig_tissue_occupancy_pie.{pdf,png}

Usage:
    python src/dataset/count_dialogue_stats.py                       # infer paths from data_pathes
    python src/dataset/count_dialogue_stats.py --input dialogues.jsonl --outdir figures
    python src/dataset/count_dialogue_stats.py --input ... --out-json stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Schema constants + English display labels (Chinese target_names are
# mapped to English via the numeric target_label so figures stay font-safe).
# ------------------------------------------------------------------
TISSUE_TYPES = {"brainparc_tissue"}

# BrainParc tissue labels (see 3_build_multiround_dataset.py: 1=CSF, 2=GM, 3=WM).
TISSUE_LABEL_EN = {1: "CSF", 2: "Gray matter", 3: "White matter"}
TISSUE_ORDER = [1, 2, 3]

# Fixed categorical order + readable names for the six dialogue categories.
CATEGORY_DISPLAY = {
    "basic_segmentation": "Basic segmentation",
    "tissue_to_region": "Tissue-to-region",
    "contralateral_same_region": "Contralateral same region",
    "same_side_same_lobe": "Same-side same-lobe",
    "spatial_named_region": "Spatial named region",
    "tumor_to_overlapping_region": "Tumor-to-overlapping-region",
}

# Short labels printed inside each pie wedge to save figure space.
CATEGORY_ABBR = {
    "basic_segmentation": "BS",
    "tissue_to_region": "T2R",
    "contralateral_same_region": "CSR",
    "same_side_same_lobe": "SSL",
    "spatial_named_region": "SNR",
    "tumor_to_overlapping_region": "TOR",
}
TISSUE_ABBR = {1: "CSF", 2: "GM", 3: "WM"}

# One-line descriptions used to build the figure caption ("figure description").
CATEGORY_DESC = {
    "basic_segmentation":
        "a single-round request that directly segments one named structure",
    "tissue_to_region":
        "a tissue mask cues the first round, then a fine anatomical region sharing "
        "that tissue attribute is segmented",
    "contralateral_same_region":
        "a reference region is given, then its counterpart in the opposite hemisphere "
        "is segmented",
    "same_side_same_lobe":
        "one structure sets a hemisphere and lobe, then another region within the same "
        "scope is segmented",
    "spatial_named_region":
        "one region acts as a spatial anchor, then a named region in a relative image "
        "direction is segmented",
    "tumor_to_overlapping_region":
        "a tumor mask is given, then the anatomical region with the largest spatial "
        "overlap is segmented",
}
TISSUE_DESC = {
    "CSF": "cerebrospinal fluid",
    "Gray matter": "cortical and subcortical gray matter",
    "White matter": "cerebral white matter",
}

# Okabe-Ito colorblind-safe categorical palette (fixed order, never cycled).
CATEGORY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
# Semantically suggestive, CVD-safe tissue colors (CSF / gray matter / white matter).
TISSUE_COLORS = {1: "#0072B2", 2: "#999999", 3: "#E69F00"}


def default_input_path() -> Optional[Path]:
    """Prefer the de-duplicated dialogues file, else the raw one, using the
    dataset root from src/dataset/data_pathes.py when available."""
    try:
        from src.dataset.data_pathes import instance_out_dir
    except Exception:
        try:
            from data_pathes import instance_out_dir  # type: ignore
        except Exception:
            return None
    base = Path(instance_out_dir) / "multiround_dataset"
    dedup = base / "multiround_dialogues.train.jsonl"
    # raw = base / "multiround_dialogues.jsonl"
    if dedup.exists():
        return dedup
    # if raw.exists():
    #     return raw
    return dedup  # report the expected path even if missing


def iter_dialogues(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_stats(path: Path) -> dict:
    n_dialogues = 0
    n_rounds = 0
    case_ids: set = set()
    slice_ids: set = set()
    conversation_type_counts: Counter = Counter()
    tissue_label_counts: Counter = Counter()  # keyed by numeric tissue label

    for d in iter_dialogues(path):
        n_dialogues += 1
        cid = d.get("case_id")
        if cid is not None:
            case_ids.add(str(cid))
        sid = d.get("slice_id")
        if sid is not None:
            slice_ids.add(str(sid))
        conversation_type_counts[str(d.get("conversation_type", "unknown"))] += 1
        for rd in (d.get("rounds") or []):
            n_rounds += 1
            if str(rd.get("target_type")) in TISSUE_TYPES:
                try:
                    lab = int(rd.get("target_label"))
                except (TypeError, ValueError):
                    continue
                tissue_label_counts[lab] += 1

    return {
        "input": str(path),
        "n_cases": len(case_ids),
        "n_slices": len(slice_ids),
        "n_dialogues": n_dialogues,
        "n_rounds": n_rounds,
        "conversation_type_counts": dict(conversation_type_counts.most_common()),
        "tissue_label_counts": {int(k): int(v) for k, v in tissue_label_counts.items()},
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def _apply_academic_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#222222",
        "savefig.bbox": "tight",
    })


def _hex_to_rgb01(hex_color: str):
    """Convert '#RRGGBB' to RGB values in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 0.5, 0.5, 0.5
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _text_color_for_wedge(hex_color: str) -> str:
    """Choose black/white text according to wedge brightness."""
    r, g, b = _hex_to_rgb01(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 0.58 else "white"


def _pie(labels, values, colors, out_base: Path, formats) -> list:
    """Compact academic pie chart.

    Labels are placed inside wedges as abbreviation + percentage. For very small
    wedges, only the abbreviation is printed, and the exact count/percentage are
    reported in the caption text.
    """
    total = sum(values) or 1

    # Compact square canvas: no outside labels, no side legend.
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    wedges, _texts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=1.0,
        wedgeprops=dict(edgecolor="white", linewidth=1.4),
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    for wedge, name, val, color in zip(wedges, labels, values, colors):
        pct = val / total * 100.0
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        x = math.cos(math.radians(ang))
        y = math.sin(math.radians(ang))

        # Small wedges cannot contain both abbreviation and percentage clearly.
        # Keep the wedge label inside and put exact numbers in the caption.
        if pct < 3.0:
            label = name
            r = 0.76
            fontsize = 7.0
            rotation = ang
        elif pct < 7.0:
            label = f"{name}\n{pct:.1f}%"
            r = 0.70
            fontsize = 7.8
            rotation = 0
        else:
            label = f"{name}\n{pct:.1f}%"
            r = 0.62
            fontsize = 9.0
            rotation = 0

        # Keep radial small-wedge labels readable instead of upside down.
        if rotation != 0 and 90 < rotation < 270:
            rotation += 180

        ax.text(
            r * x,
            r * y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=_text_color_for_wedge(color),
            rotation=rotation,
            rotation_mode="anchor",
        )

    # Remove almost all outer whitespace; useful for manuscript layouts.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    written = []
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        written.append(p)
    plt.close(fig)
    return written

def _write_caption(out_base: Path, text: str) -> Path:
    p = out_base.with_name(out_base.name + "_caption.txt")
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p


def _fmt_count_pct(name: str, count: int, total: int) -> str:
    pct = count / (total or 1) * 100.0
    return f"{name}, n={count:,}, {pct:.1f}%"


def category_caption(counts: dict) -> str:
    """One-paragraph figure description explaining each dialogue category and abbreviation."""
    present = [c for c in CATEGORY_DISPLAY if counts.get(c, 0) > 0]
    present += [c for c in counts if c not in CATEGORY_DISPLAY and counts.get(c, 0) > 0]
    total = sum(counts.values())

    parts = []
    for c in present:
        full = CATEGORY_DISPLAY.get(c, c.replace("_", " "))
        abbr = CATEGORY_ABBR.get(c, full[:3].upper())
        desc = CATEGORY_DESC.get(c, "see text")
        parts.append(f"{abbr}={_fmt_count_pct(full, counts[c], total)}; {desc}")

    body = "; ".join(parts)
    return (
        f"Distribution of dialogue categories over {total:,} generated dialogues. "
        f"Wedge labels use compact abbreviations to save space: {body}."
    )


def tissue_caption(counts: dict) -> str:
    """One-paragraph figure description explaining the tissue-occupancy chart."""
    labs = [l for l in TISSUE_ORDER if counts.get(l, 0) > 0]
    labs += [l for l in counts if l not in TISSUE_ORDER and counts.get(l, 0) > 0]
    total = sum(counts.values())

    parts = []
    for lab in labs:
        full = TISSUE_LABEL_EN.get(lab, f"tissue {lab}")
        abbr = TISSUE_ABBR.get(lab, f"T{lab}")
        desc = TISSUE_DESC.get(full, full)
        parts.append(f"{abbr}={_fmt_count_pct(full, counts[lab], total)}; {desc}")

    body = "; ".join(parts)
    return (
        f"Distribution of BrainParc tissue occupancy across {total:,} tissue-target "
        f"segmentation rounds. Wedge labels use compact abbreviations: {body}."
    )


def _category_pairs(counts: dict):
    """Dialogue categories in fixed pipeline order, labelled by compact abbreviations."""
    ordered = [c for c in CATEGORY_DISPLAY if c in counts]
    ordered += [c for c in counts if c not in CATEGORY_DISPLAY]
    labels = [CATEGORY_ABBR.get(c, CATEGORY_DISPLAY.get(c, c)[:3].upper()) for c in ordered]
    values = [counts[c] for c in ordered]
    colors = [CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(ordered))]
    return labels, values, colors


def _tissue_pairs(counts: dict):
    """Tissues in canonical order CSF -> GM -> WM, labelled by compact abbreviations."""
    labs = [l for l in TISSUE_ORDER if counts.get(l, 0) > 0]
    labs += [l for l in counts if l not in TISSUE_ORDER and counts.get(l, 0) > 0]
    labels = [TISSUE_ABBR.get(l, f"T{l}") for l in labs]
    values = [counts[l] for l in labs]
    colors = [TISSUE_COLORS.get(l, "#BBBBBB") for l in labs]
    return labels, values, colors

def render_figures(stats: dict, outdir: Path, formats) -> list:
    _apply_academic_style()
    outdir.mkdir(parents=True, exist_ok=True)
    written: list = []

    cat_base = outdir / "fig_dialogue_categories_pie"
    cat_labels, cat_values, cat_colors = _category_pairs(stats["conversation_type_counts"])
    written += _pie(cat_labels, cat_values, cat_colors, cat_base, formats)
    written.append(_write_caption(cat_base, category_caption(stats["conversation_type_counts"])))

    tis_base = outdir / "fig_tissue_occupancy_pie"
    tis_labels, tis_values, tis_colors = _tissue_pairs(stats["tissue_label_counts"])
    written += _pie(tis_labels, tis_values, tis_colors, tis_base, formats)
    written.append(_write_caption(tis_base, tissue_caption(stats["tissue_label_counts"])))

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None,
                    help="Path to multiround_dialogues(.dedup).jsonl "
                         "(default: inferred from data_pathes.py).")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Directory for the figures (default: <input dir>/figures).")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    help="Figure formats to write. Default: pdf png.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Optional path to also dump the full statistics as JSON.")
    args = ap.parse_args()

    input_path = args.input or default_input_path()
    if input_path is None:
        ap.error("Could not infer the input path; please pass --input explicitly.")
    if not input_path.exists():
        ap.error(f"Input file not found: {input_path}")

    outdir = args.outdir or (input_path.parent / "figures")
    stats = compute_stats(input_path)

    print(f"input     : {stats['input']}")
    print(f"cases     : {stats['n_cases']:,}")
    print(f"slices    : {stats['n_slices']:,}")
    print(f"dialogues : {stats['n_dialogues']:,} | rounds : {stats['n_rounds']:,}")
    print("\n[manuscript sentence]")
    print(f"  In total, the dataset comprises {stats['n_cases']:,} cases, "
          f"{stats['n_slices']:,} 2D slices, and {stats['n_dialogues']:,} "
          f"multi-round dialogues after de-duplication.")

    written = render_figures(stats, outdir, formats=args.formats)
    print(f"\n[figures] {len(written)} file(s) written to {outdir}:")
    for p in written:
        print(f"  {p.name}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[written] {args.out_json}")


if __name__ == "__main__":
    main()
