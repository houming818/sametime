#!/usr/bin/env python3
"""Build deterministic, disjoint raw/purified/evaluation TSV files."""

import argparse
import hashlib
import json
from pathlib import Path


def stable_key(seed: int, row: dict) -> str:
    text = f"{seed}:{row['line']}:{row['zh']}\t{row['en']}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_tsv(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = f"{row['zh']}\t{row['en']}\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def partition_slot(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big") % 1000


def force_partition(row: dict, slot: int) -> str:
    base = f"{row['zh']}\t{row['en']}"
    for padding in range(10_000):
        line = base + (" " * padding) + "\n"
        if partition_slot(line) == slot:
            return line
    raise RuntimeError(f"could not assign partition slot={slot} line={row['line']}")


def write_eval_tsv(path: Path, valid_rows: list[dict], test_rows: list[dict], filler: dict) -> str:
    digest = hashlib.sha256()
    lines = [force_partition(filler, 2)]
    lines.extend(force_partition(row, 0) for row in valid_rows)
    lines.extend(force_partition(row, 1) for row in test_rows)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-rows", type=int, default=40_000)
    parser.add_argument("--eval-rows", type=int, default=1_024, help="Rows per valid/test split")
    parser.add_argument("--purified-threshold", type=float, default=0.98)
    parser.add_argument("--eval-threshold", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=14102)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    eval_candidates = [row for row in rows if row["score"] >= args.eval_threshold]
    eval_rows = sorted(eval_candidates, key=lambda row: stable_key(args.seed + 1, row))[: 2 * args.eval_rows]
    valid_rows, test_rows = eval_rows[: args.eval_rows], eval_rows[args.eval_rows :]
    eval_lines = {row["line"] for row in eval_rows}
    remaining = [row for row in rows if row["line"] not in eval_lines]
    raw_rows = sorted(remaining, key=lambda row: stable_key(args.seed + 2, row))[: args.train_rows]
    purified_candidates = [row for row in remaining if row["score"] >= args.purified_threshold]
    purified_rows = sorted(purified_candidates, key=lambda row: stable_key(args.seed + 3, row))[: args.train_rows]
    if len(raw_rows) < args.train_rows or len(purified_rows) < args.train_rows or len(eval_rows) < 2 * args.eval_rows:
        raise RuntimeError(
            f"insufficient rows raw={len(raw_rows)} purified={len(purified_rows)} eval={len(eval_rows)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {
        "raw": write_tsv(args.output / "raw_train.tsv", raw_rows),
        "purified": write_tsv(args.output / "purified_train.tsv", purified_rows),
        "eval": write_eval_tsv(args.output / "shared_eval.tsv", valid_rows, test_rows, remaining[0]),
    }
    summary = {
        "claim": "LOCAL-PARALLEL-PURIFICATION-AB-001",
        "seed": args.seed,
        "thresholds": {"purified": args.purified_threshold, "eval": args.eval_threshold},
        "rows": {
            "raw": len(raw_rows), "purified": len(purified_rows),
            "valid": len(valid_rows), "test": len(test_rows),
        },
        "candidate_counts": {"purified": len(purified_candidates), "eval": len(eval_candidates)},
        "sha256": hashes,
        "disjoint": not (({r['line'] for r in raw_rows} | {r['line'] for r in purified_rows}) & eval_lines),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
