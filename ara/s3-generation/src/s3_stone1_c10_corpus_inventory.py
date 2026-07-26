#!/usr/bin/env python3
"""Build the frozen io-local corpus inventory for STONE-1 C10."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            count += chunk.count(b"\n")
    return count


def parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def classify(path: Path, root: Path) -> tuple[str, str, bool]:
    relative = path.relative_to(root).as_posix()
    name = path.name
    lower = relative.lower()
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        return "excluded", "hidden_or_cache", False
    if name.endswith((".lock", ".incomplete", ".metadata")):
        return "excluded", "incomplete_download", False
    if lower.startswith(("derived/", "distillation/", "treeheap_checkpoints/")):
        return "excluded", "generated_artifact", False
    if name in {"sp_bpe.model", "sp_bpe.vocab", "sp_bpe_massive.model", "sp_bpe_massive.vocab"}:
        return "excluded", "tokenizer", False
    if lower.startswith("wmt_massive/") or lower.startswith("wmt17/") or "translation2019zh" in lower:
        return "parallel", "train" if "valid" not in lower else "validation", True
    if any(part in lower for part in ("belle_zh", "baike2018qa", "chinese-medical-dialogue-data")):
        split = "validation" if any(tag in lower for tag in ("valid", "test", "sample_test")) else "train"
        return "instruction_qa", split, path.suffix.lower() in {".json", ".jsonl", ".csv"}
    if any(part in lower for part in ("new2016zh", "webtext2019zh", "wiki_zh", "zhihu-kol")):
        split = "validation" if any(tag in lower for tag in ("valid", "testa")) else "train"
        usable = path.suffix.lower() in {".json", ".jsonl", ".parquet"} or name.startswith("wiki_")
        return "raw_continuation", split, usable
    return "excluded", "unclassified", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/nio/datasets")
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_stone1_c10_corpus_inventory",
    )
    parser.add_argument("--hash-train-files", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = []
    totals: dict[str, dict[str, int]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        category, split, eligible = classify(path, root)
        size = path.stat().st_size
        rows = None
        if eligible:
            rows = parquet_rows(path) if path.suffix.lower() == ".parquet" else count_lines(path)
        record = {
            "path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": size,
            "category": category,
            "split": split,
            "eligible": eligible,
            "physical_rows": rows,
        }
        if eligible and (args.hash_train_files or split != "train"):
            record["sha256"] = sha256(path)
        files.append(record)
        key = f"{category}:{split}"
        bucket = totals.setdefault(key, {"files": 0, "bytes": 0, "physical_rows": 0})
        bucket["files"] += 1
        bucket["bytes"] += size
        if rows is not None:
            bucket["physical_rows"] += rows
        if eligible:
            print(json.dumps({"file": record["relative_path"], "rows": rows, "bytes": size}), flush=True)

    eligible_train_bytes = sum(
        row["bytes"] for row in files if row["eligible"] and row["split"] == "train"
    )
    eligible_train_rows = sum(
        row["physical_rows"] or 0
        for row in files
        if row["eligible"] and row["split"] == "train"
    )
    summary = {
        "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
        "host": socket.gethostname(),
        "root": str(root),
        "root_bytes": sum(row["bytes"] for row in files),
        "eligible_train_bytes": eligible_train_bytes,
        "eligible_train_physical_rows": eligible_train_rows,
        "totals": totals,
        "files": files,
        "known_incompleteness": [
            "Zhihu-KOL shard 00000 is absent from the materialized dataset",
            "physical row counts precede content filtering and packing",
        ],
        "elapsed_sec": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "eligible_train_bytes", "eligible_train_physical_rows", "totals", "elapsed_sec"
    )}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
