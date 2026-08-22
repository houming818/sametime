#!/usr/bin/env python3
"""Verify full non-parallel quality shards and produce a compact summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    complete_path = args.input / "run_complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing {complete_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    members = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input.glob("shard_*.done.json"))
    ]
    if not members:
        raise RuntimeError("no shard metadata")
    indexes = [item["shard"] for item in members]
    if indexes != list(range(len(members))):
        raise RuntimeError(f"non-contiguous shard indexes: {indexes}")
    expected_first = 1
    source_counts = Counter()
    flag_counts = Counter()
    score_rows = 0
    score_sum = 0.0
    histogram = [0] * 1000
    for item in members:
        if item["first_global_row"] != expected_first:
            raise RuntimeError(
                f"global interval gap at shard {item['shard']}: "
                f"{item['first_global_row']} != {expected_first}"
            )
        expected_first = item["last_global_row"] + 1
        source_counts.update(item["source_counts"])
        flag_counts.update(item["flags"])
        score_rows += item["finite_scores"]
        score_sum += item["score_sum"] or 0.0
        if item["score_histogram_1000"] is not None:
            histogram = [left + right for left, right in zip(histogram, item["score_histogram_1000"])]
        if args.verify_hashes:
            manifest = args.input / f"shard_{item['shard']:05d}.jsonl.gz"
            if file_sha256(manifest) != item["manifest_sha256"]:
                raise RuntimeError(f"hash mismatch: {manifest}")
    rows = expected_first - 1
    if rows != complete["rows"] or rows != sum(item["rows"] for item in members):
        raise RuntimeError("full row count mismatch")
    if dict(sorted(source_counts.items())) != complete["source_counts"]:
        raise RuntimeError("source-count mismatch")
    expected_score_rows = 0 if complete["mode"] == "integrity_only" else rows
    if score_rows != expected_score_rows:
        raise RuntimeError(f"finite score mismatch: {score_rows} != {expected_score_rows}")
    payload = {
        "claim": "NIO-NONPAR-FULL-C01",
        "family": complete["family"],
        "mode": complete["mode"],
        "rows": rows,
        "shards": len(members),
        "source_counts": dict(sorted(source_counts.items())),
        "flags": dict(sorted(flag_counts.items())),
        "finite_scores": score_rows,
        "mean_score": score_sum / score_rows if score_rows else None,
        "score_histogram_1000": histogram if score_rows else None,
        "hashes_verified": args.verify_hashes,
        "gates": {
            "F1_exact_rows": True,
            "F2_finite_scores": score_rows == expected_score_rows,
            "F3_contiguous_shards": True,
            "F4_hashes": args.verify_hashes,
            "F5_non_destructive": True,
        },
        "members": members,
        "warning": "Scores and flags are shadow metadata; no source row was rewritten or deleted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in payload.items() if key not in {"members", "score_histogram_1000"}}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
