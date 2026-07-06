#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_extract_training_dialogues.py

Split the full multi-round dialogue file (produced by 3_build_multiround_dataset.py
and de-duplicated by 4_remove_duplicate.py) into a training file and a test file,
according to the case-id lists in:

    data/ids/train_case_ids.txt
    data/ids/test_case_ids.txt

Each dialogue is routed by its ``case_id``:
  * case_id in the training id list -> training output
  * case_id in the test id list     -> test output
  * case_id in neither list         -> dropped (counted and reported)

Rationale: the generation pipeline (steps 1-3) builds dialogues for every case,
so a single file mixes training and test subjects. This step enforces the
predefined subject-level split so that no test case leaks into training.

The pass is streaming (only the two id sets and the compact per-split counters
stay in memory) and writes the original jsonl bytes unchanged, so the split
dialogues are byte-for-byte identical to the input.

Usage:
    # infer input + id lists + output dir from the repo / data_pathes:
    python src/dataset/5_extract_training_dialogues.py

    # be explicit:
    python src/dataset/5_extract_training_dialogues.py \
        --input   .../multiround_dialogues.dedup.jsonl \
        --train-ids data/ids/train_case_ids.txt \
        --test-ids  data/ids/test_case_ids.txt \
        --outdir  .../multiround_dataset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_IDS = REPO_ROOT / "data" / "ids" / "train_case_ids.txt"
DEFAULT_TEST_IDS = REPO_ROOT / "data" / "ids" / "test_case_ids.txt"


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
    dedup = base / "multiround_dialogues.dedup.jsonl"
    raw = base / "multiround_dialogues.jsonl"
    if dedup.exists():
        return dedup
    if raw.exists():
        return raw
    return dedup  # report the expected path even if missing


def load_case_ids(path: Path) -> Set[str]:
    """Read a case-id list: one id per non-empty line; '#' starts a comment."""
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
    return ids


def split_dialogues(
    input_path: Path,
    train_ids: Set[str],
    test_ids: Set[str],
    train_out: Path,
    test_out: Path,
    report_path: Optional[Path],
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} case id(s) appear in BOTH train and test lists, "
            f"e.g. {sorted(overlap)[:5]}. The split must be disjoint."
        )

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_train = n_test = n_other = n_bad = 0
    train_cases_seen: Set[str] = set()
    test_cases_seen: Set[str] = set()
    other_cases_seen: Set[str] = set()
    train_by_type: Counter = Counter()
    test_by_type: Counter = Counter()

    with input_path.open("rb") as fin, \
            train_out.open("wb") as f_train, \
            test_out.open("wb") as f_test:
        for raw_line in fin:
            if not raw_line.strip():
                continue
            n_total += 1
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            case_id = str(d.get("case_id", ""))
            ctype = str(d.get("conversation_type", "unknown"))
            out_line = raw_line if raw_line.endswith(b"\n") else raw_line + b"\n"

            if case_id in train_ids:
                f_train.write(out_line)
                n_train += 1
                train_cases_seen.add(case_id)
                train_by_type[ctype] += 1
            elif case_id in test_ids:
                f_test.write(out_line)
                n_test += 1
                test_cases_seen.add(case_id)
                test_by_type[ctype] += 1
            else:
                n_other += 1
                if case_id:
                    other_cases_seen.add(case_id)

            if n_total % 1_000_000 == 0:
                print(f"...processed {n_total:,} | train {n_train:,} | test {n_test:,}",
                      flush=True)

    summary = {
        "input": str(input_path),
        "train_out": str(train_out),
        "test_out": str(test_out),
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "n_dropped_not_in_any_list": n_other,
        "n_unparseable_skipped": n_bad,
        "n_train_ids": len(train_ids),
        "n_test_ids": len(test_ids),
        "n_train_cases_with_dialogues": len(train_cases_seen),
        "n_test_cases_with_dialogues": len(test_cases_seen),
        "n_train_ids_without_dialogues": len(train_ids - train_cases_seen),
        "n_test_ids_without_dialogues": len(test_ids - test_cases_seen),
        "n_dropped_unique_cases": len(other_cases_seen),
        "dropped_case_ids_sample": sorted(other_cases_seen)[:20],
        "train_ids_without_dialogues_sample": sorted(train_ids - train_cases_seen)[:20],
        "test_ids_without_dialogues_sample": sorted(test_ids - test_cases_seen)[:20],
        "train_conversation_type_counts": dict(train_by_type.most_common()),
        "test_conversation_type_counts": dict(test_by_type.most_common()),
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    return summary


def _default_outputs(input_path: Path, outdir: Path) -> Tuple[Path, Path]:
    """Derive train/test filenames from the input, dropping a trailing '.dedup'."""
    stem = input_path.name
    for suffix in (".jsonl", ".dedup"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return outdir / f"{stem}.train.jsonl", outdir / f"{stem}.test.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=None,
                    help="Full multiround_dialogues(.dedup).jsonl "
                         "(default: inferred from data_pathes.py).")
    ap.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN_IDS,
                    help=f"Training case-id list (default: {DEFAULT_TRAIN_IDS}).")
    ap.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS,
                    help=f"Test case-id list (default: {DEFAULT_TEST_IDS}).")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Directory for the split files (default: the input's directory).")
    ap.add_argument("--train-out", type=Path, default=None,
                    help="Explicit training output path (overrides --outdir naming).")
    ap.add_argument("--test-out", type=Path, default=None,
                    help="Explicit test output path (overrides --outdir naming).")
    ap.add_argument("--report", type=Path, default=None,
                    help="Split summary json (default: <outdir>/split_summary.json).")
    args = ap.parse_args()

    input_path = args.input or default_input_path()
    if input_path is None:
        ap.error("Could not infer the input path; please pass --input explicitly.")
    if not input_path.exists():
        ap.error(f"Input file not found: {input_path}")
    for p in (args.train_ids, args.test_ids):
        if not p.exists():
            ap.error(f"Case-id list not found: {p}")

    outdir = args.outdir or input_path.parent
    train_out, test_out = _default_outputs(input_path, outdir)
    if args.train_out is not None:
        train_out = args.train_out
    if args.test_out is not None:
        test_out = args.test_out
    report_path = args.report or (outdir / "split_summary.json")

    train_ids = load_case_ids(args.train_ids)
    test_ids = load_case_ids(args.test_ids)

    print("=" * 80)
    print(f"INPUT     : {input_path}")
    print(f"TRAIN IDS : {args.train_ids}  ({len(train_ids)} cases)")
    print(f"TEST IDS  : {args.test_ids}  ({len(test_ids)} cases)")
    print(f"TRAIN OUT : {train_out}")
    print(f"TEST OUT  : {test_out}")
    print("=" * 80)

    summary = split_dialogues(input_path, train_ids, test_ids,
                              train_out, test_out, report_path)

    print("\n=== SPLIT SUMMARY ===")
    print(f"total dialogues            : {summary['n_total']:,}")
    print(f"  -> train                 : {summary['n_train']:,} "
          f"({summary['n_train_cases_with_dialogues']} cases)")
    print(f"  -> test                  : {summary['n_test']:,} "
          f"({summary['n_test_cases_with_dialogues']} cases)")
    print(f"  -> dropped (no list)     : {summary['n_dropped_not_in_any_list']:,} "
          f"({summary['n_dropped_unique_cases']} unique cases)")
    if summary["n_unparseable_skipped"]:
        print(f"  -> unparseable skipped   : {summary['n_unparseable_skipped']:,}")
    if summary["n_train_ids_without_dialogues"]:
        print(f"train ids with no dialogues: {summary['n_train_ids_without_dialogues']} "
              f"(e.g. {summary['train_ids_without_dialogues_sample'][:5]})")
    if summary["n_test_ids_without_dialogues"]:
        print(f"test ids with no dialogues : {summary['n_test_ids_without_dialogues']} "
              f"(e.g. {summary['test_ids_without_dialogues_sample'][:5]})")
    if summary["n_dropped_unique_cases"]:
        print(f"[warn] {summary['n_dropped_unique_cases']} case(s) had dialogues but were "
              f"in neither list, e.g. {summary['dropped_case_ids_sample'][:5]}")
    print(f"\nreport: {report_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
