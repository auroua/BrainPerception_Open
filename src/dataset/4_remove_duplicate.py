"""
4_remove_duplicate.py

去重脚本：流式读取 3_build_multiround_dataset.py 生成的
multiround_dialogues.jsonl，删除“同一 case 内、跨切片的重复任务”。

去重定义（与 3_build_multiround_dataset.py 中 dialogue_dedup_key 完全一致）：
    key = (
        case_id,
        conversation_type,
        每一轮按顺序的 (target_type, target_name, spatial_relation 或 ""),
    )

也就是说：
- 同一 case、同一对话类型、各轮目标结构序列完全相同（含 spatial 方向）
  的对话被视为“同一任务”；
- 相邻切片（例如 axial_104..108）上反复出现的同名结构任务属于同一 key。

为了避免训练数据过少，每个 key 不再只保留 1 条，而是最多保留
--max-per-key 条（默认 10），超出部分才丢弃。设为 1 即恢复“只保留首条”。

特点：
- 流式处理，常驻内存只保存“已见签名”的 16 字节摘要集合，可处理数十 GB 文件；
- 输出原始行字节，不重新序列化，保证保留的对话与输入完全一致；
- 同时写出去重统计 dedup_summary.json。

运行：
    python 4_remove_duplicate.py \
        --input  /path/to/multiround_dialogues.jsonl \
        --output /path/to/multiround_dialogues.dedup.jsonl

不带参数时，使用 data_pathes 中 instance_out_dir 推断出的默认路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append("/mnt/rna01/chenw/Workspaces/BrainPerception")


def _default_paths():
    """从 data_pathes 推断默认输入/输出路径；失败则返回 (None, None)。"""
    try:
        from data_pathes import instance_out_dir
    except Exception:
        try:
            from src.dataset.data_pathes import instance_out_dir
        except Exception:
            return None, None
    base = Path(instance_out_dir) / "multiround_dataset"
    return base / "multiround_dialogues.jsonl", base / "multiround_dialogues.dedup.jsonl"


def dialogue_dedup_key(dialogue: dict) -> bytes:
    """
    计算对话的去重签名（返回 16 字节 blake2b 摘要）。

    必须与 3_build_multiround_dataset.py 的同名逻辑保持一致：
    使用 case_id + conversation_type + 每轮 (target_type, target_name, spatial_relation)。
    """
    rounds = dialogue.get("rounds", []) or []
    parts = []
    for r in rounds:
        rel = r.get("spatial_relation")
        parts.append(
            "|".join(
                [
                    str(r.get("target_type")),
                    str(r.get("target_name")),
                    "" if rel is None else str(rel),
                ]
            )
        )
    raw = "\x1e".join(
        [
            str(dialogue.get("case_id")),
            str(dialogue.get("conversation_type")),
            "\x1f".join(parts),
        ]
    )
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).digest()


def dedup_file(
    input_path: Path,
    output_path: Path,
    report_path: Path | None,
    max_per_key: int = 10,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if max_per_key < 1:
        raise ValueError(f"max_per_key must be >= 1, got {max_per_key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # key -> 已保留条数；最多保留 max_per_key 条
    seen: Counter = Counter()
    n_total = 0
    n_kept = 0
    n_dropped = 0
    n_bad = 0
    kept_by_type: Counter = Counter()
    dropped_by_type: Counter = Counter()

    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        for raw_line in fin:
            if not raw_line.strip():
                continue
            n_total += 1
            try:
                d = json.loads(raw_line)
            except Exception:
                # 无法解析的行原样保留，避免悄悄丢数据
                n_bad += 1
                fout.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
                continue

            ctype = str(d.get("conversation_type"))
            key = dialogue_dedup_key(d)
            if seen[key] >= max_per_key:
                n_dropped += 1
                dropped_by_type[ctype] += 1
            else:
                seen[key] += 1
                n_kept += 1
                kept_by_type[ctype] += 1
                fout.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")

            if n_total % 1_000_000 == 0:
                print(
                    f"...processed {n_total:,} | kept {n_kept:,} | dropped {n_dropped:,}",
                    flush=True,
                )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "max_per_key": max_per_key,
        "n_unique_keys": len(seen),
        "n_total": n_total,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "n_unparseable_kept": n_bad,
        "drop_fraction": (n_dropped / n_total) if n_total else 0.0,
        "kept_by_type": dict(kept_by_type),
        "dropped_by_type": dict(dropped_by_type),
        "dedup_key": "case_id + conversation_type + per-round (target_type, target_name, spatial_relation)",
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    default_in, default_out = _default_paths()

    parser = argparse.ArgumentParser(description="Remove near-duplicate multiround dialogues.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_in,
        help="输入 multiround_dialogues.jsonl 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="去重后输出 jsonl 路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="去重统计 json 路径（默认写到 output 同目录 dedup_summary.json）",
    )
    parser.add_argument(
        "--max-per-key",
        type=int,
        default=10,
        help="每个去重 key 最多保留多少条（默认 10；设为 1 即只保留首条）",
    )
    args = parser.parse_args()

    if args.input is None or args.output is None:
        parser.error("无法推断默认路径，请显式提供 --input 和 --output。")

    report_path = args.report or (args.output.parent / "dedup_summary.json")

    print("=" * 80)
    print(f"INPUT : {args.input}")
    print(f"OUTPUT: {args.output}")
    print(f"REPORT: {report_path}")
    print(f"MAX/KEY: {args.max_per_key}")
    print("=" * 80)

    summary = dedup_file(args.input, args.output, report_path, max_per_key=args.max_per_key)

    print("\n=== DEDUP SUMMARY ===")
    print(f"max_per_key : {summary['max_per_key']}")
    print(f"unique_keys : {summary['n_unique_keys']:,}")
    print(f"total       : {summary['n_total']:,}")
    print(f"kept        : {summary['n_kept']:,}")
    print(f"dropped     : {summary['n_dropped']:,}")
    print(f"drop_frac   : {summary['drop_fraction']:.4f}")
    if summary["n_unparseable_kept"]:
        print(f"unparseable (kept as-is): {summary['n_unparseable_kept']:,}")
    print(f"kept_by_type   : {summary['kept_by_type']}")
    print(f"dropped_by_type: {summary['dropped_by_type']}")
    print(f"\nreport: {report_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
