#!/usr/bin/env python3
"""Build nested purified training ladders from a scored manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def stable_key(seed: int, row: dict) -> str:
    payload = f"{seed}:{row['line']}:{row['zh']}\t{row['en']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_slot(line: str) -> int:
    digest = hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 1000


def force_partition(row: dict, slot: int | str) -> str:
    base = f"{row['zh']}\t{row['en']}"
    for nonce in range(1_000_000):
        line = f"{base}\tpartition_pad_{nonce}\n"
        actual = partition_slot(line)
        if (slot == "train" and actual >= 2) or actual == slot:
            return line
    raise RuntimeError(f"partition search failed slot={slot} source_line={row['line']}")


def write_lines(path: Path, lines: list[str]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", default="40000,80000,120000,160000,200000")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--eval-threshold", type=float, default=0.9999)
    parser.add_argument("--eval-rows", type=int, default=1024, help="Rows per valid/test split")
    parser.add_argument("--seed", type=int, default=14106)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if sizes != sorted(set(sizes)):
        raise ValueError("sizes must be unique and ascending")

    purified, eval_candidates = [], []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            score = float(row["score"])
            if score >= args.threshold:
                purified.append(row)
            if score >= args.eval_threshold:
                eval_candidates.append(row)
    eval_rows = sorted(eval_candidates, key=lambda row: stable_key(args.seed + 1, row))[: 2 * args.eval_rows]
    if len(eval_rows) < 2 * args.eval_rows:
        raise RuntimeError(f"insufficient eval rows: {len(eval_rows)}")
    eval_ids = {row["line"] for row in eval_rows}
    purified = [row for row in purified if row["line"] not in eval_ids]
    purified.sort(key=lambda row: stable_key(args.seed + 2, row))
    if len(purified) < sizes[-1]:
        raise RuntimeError(f"insufficient purified rows: {len(purified)} < {sizes[-1]}")

    args.output.mkdir(parents=True, exist_ok=True)
    train_lines = [force_partition(row, "train") for row in purified[: sizes[-1]]]
    hashes = {}
    for size in sizes:
        hashes[f"train_{size}"] = write_lines(args.output / f"purified_{size}.tsv", train_lines[:size])
    valid = eval_rows[: args.eval_rows]
    test = eval_rows[args.eval_rows :]
    filler = purified[sizes[-1]]
    eval_lines = [force_partition(filler, "train")]
    eval_lines.extend(force_partition(row, 0) for row in valid)
    eval_lines.extend(force_partition(row, 1) for row in test)
    hashes["eval"] = write_lines(args.output / "shared_eval.tsv", eval_lines)

    summary = {
        "claim": "LOCAL-PURIFICATION-SCALE-LADDER-001",
        "seed": args.seed,
        "sizes": sizes,
        "thresholds": {"train": args.threshold, "eval": args.eval_threshold},
        "candidate_counts": {"purified": len(purified), "eval": len(eval_candidates)},
        "eval_rows": {"valid": len(valid), "test": len(test)},
        "nested": True,
        "training_eval_disjoint": not ({row["line"] for row in purified[: sizes[-1]]} & eval_ids),
        "sha256": hashes,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
