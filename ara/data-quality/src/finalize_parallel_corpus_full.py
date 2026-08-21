#!/usr/bin/env python3
"""Validate full-corpus score shards and write a content-addressed summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=14_170_275)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.input.glob("shard_*.done.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise RuntimeError("no completed score shards")
    indexes = [item["shard"] for item in summaries]
    if indexes != list(range(len(indexes))):
        raise RuntimeError(f"non-contiguous shards: {indexes}")
    rows = sum(item["rows"] for item in summaries)
    if rows != args.expected_rows:
        raise RuntimeError(f"row mismatch: {rows} != {args.expected_rows}")
    payload = {
        "dataset_id": "NioScore-ZHEN-14M-v1",
        "parent": "WMT-Massive-ZHEN-14M",
        "scorer": "/home/nio/models/bge-reranker-v2-m3",
        "acceptance_rule": "score>=0.98",
        "policy": "non-destructive-shadow-filter",
        "rows_scored": rows,
        "rows_accepted": sum(item["accepted"] for item in summaries),
        "shards": len(summaries),
        "members": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "members"}), flush=True)


if __name__ == "__main__":
    main()
