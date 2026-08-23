#!/usr/bin/env python3
"""Materialize immutable training views from audited nonparallel JSONL shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--family", choices=("mono", "qa", "medical"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--reject-flags", default="mojibake,extreme_repetition",
        help="Comma-separated integrity flags excluded from the materialized view.",
    )
    args = parser.parse_args()

    summary_path = args.input / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("family") != args.family or not summary.get("hashes_verified"):
        raise RuntimeError("source summary does not satisfy the family/hash contract")
    if args.family == "mono" and args.threshold is not None:
        raise ValueError("monolingual integrity views do not have a relation threshold")
    if args.family != "mono" and args.threshold is None:
        raise ValueError("QA relation views require --threshold")

    reject_flags = {item for item in args.reject_flags.split(",") if item}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    read_rows = written_rows = 0
    rejected_score = rejected_flags = 0
    source_counts: dict[str, int] = {}

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for expected_shard, member in enumerate(summary["members"]):
            if member["shard"] != expected_shard:
                raise RuntimeError(f"non-contiguous shard {expected_shard}")
            shard = args.input / f"shard_{expected_shard:05d}.jsonl.gz"
            expected_hash = member.get("manifest_sha256")
            if expected_hash and sha256(shard) != expected_hash:
                raise RuntimeError(f"compressed shard hash mismatch for shard {expected_shard}")
            local_rows = 0
            with gzip.open(shard, "rt", encoding="utf-8", errors="strict") as handle:
                for line in handle:
                    row = json.loads(line)
                    local_rows += 1
                    read_rows += 1
                    flags = row.get("flags", {})
                    if any(flags.get(flag, False) for flag in reject_flags):
                        rejected_flags += 1
                        continue
                    if args.threshold is not None and float(row["score"]) < args.threshold:
                        rejected_score += 1
                        continue
                    source = str(row["source"])
                    source_counts[source] = source_counts.get(source, 0) + 1
                    if args.family == "mono":
                        record = {
                            "id": f"{source}:{row['row_id']}",
                            "text": str(row["left"]) + str(row["right"]),
                            "source": source,
                        }
                    else:
                        record = {
                            "id": f"{source}:{row['row_id']}",
                            "prompt": str(row["left"]),
                            "response": str(row["right"]),
                            "source": source,
                            "score": float(row["score"]),
                        }
                    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written_rows += 1
            if local_rows != member["rows"]:
                raise RuntimeError(f"row mismatch in shard {expected_shard}")
        output.flush()
        os.fsync(output.fileno())

    if read_rows != summary["rows"]:
        raise RuntimeError(f"source total mismatch: {read_rows} != {summary['rows']}")
    os.replace(temporary, args.output)
    payload = {
        "schema": "nio.materialized-view.v1",
        "dataset_id": args.dataset_id,
        "parent_claim": summary["claim"],
        "family": args.family,
        "threshold": args.threshold,
        "reject_flags": sorted(reject_flags),
        "source_rows": read_rows,
        "rows": written_rows,
        "rejected_score": rejected_score,
        "rejected_flags": rejected_flags,
        "source_counts": source_counts,
        "source_summary_sha256": sha256(summary_path),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
