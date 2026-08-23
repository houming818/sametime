#!/usr/bin/env python3
"""Materialize a two-column training view from immutable scored shards."""

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--expected-rows", type=int, default=7_304_358)
    parser.add_argument("--expected-shards", type=int, default=57)
    args = parser.parse_args()

    source_summary = args.input / "summary.json"
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    members = summary.get("members", [])
    if len(members) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(members)}")
    if summary.get("rows_accepted") != args.expected_rows:
        raise RuntimeError("source summary accepted-row count differs from contract")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    rows = 0
    previous_line = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for index, member in enumerate(members):
            if member["shard"] != index:
                raise RuntimeError(f"non-contiguous shard at index {index}")
            shard = args.input / f"shard_{index:05d}.accepted.tsv.gz"
            if sha256(shard) != member["accepted_sha256"]:
                raise RuntimeError(f"accepted shard hash mismatch: {index}")
            local_rows = 0
            with gzip.open(shard, "rt", encoding="utf-8", errors="strict") as handle:
                for raw in handle:
                    parts = raw.rstrip("\n").split("\t", 3)
                    if len(parts) != 4:
                        raise RuntimeError(f"malformed accepted row in shard {index}")
                    line_no, score = int(parts[0]), float(parts[1])
                    zh, en = parts[2].strip(), parts[3].strip()
                    if line_no <= previous_line:
                        raise RuntimeError("source line identities are not strictly increasing")
                    if score < args.threshold or not zh or not en:
                        raise RuntimeError(f"invalid accepted row at source line {line_no}")
                    output.write(f"{zh}\t{en}\n")
                    previous_line = line_no
                    local_rows += 1
                    rows += 1
            if local_rows != member["accepted"]:
                raise RuntimeError(f"accepted count mismatch in shard {index}")
        output.flush()
        os.fsync(output.fileno())

    if rows != args.expected_rows:
        raise RuntimeError(f"materialized row mismatch: {rows} != {args.expected_rows}")
    os.replace(temporary, args.output)
    payload = {
        "dataset_id": "NioView-ZHEN-S098-14M-v1",
        "parent": summary["dataset_id"],
        "policy": "two-column-materialized-high-confidence-view",
        "threshold": args.threshold,
        "rows": rows,
        "shards": len(members),
        "first_source_line": members[0]["first_line"],
        "last_source_line": members[-1]["last_line"],
        "source_summary_sha256": sha256(source_summary),
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
