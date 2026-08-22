#!/usr/bin/env python3
"""Stream non-parallel corpora into restartable quality-metadata shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nonparallel_quality_pilot import family_sources, flags, score_pairs


HISTOGRAM_BINS = 1000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(row: tuple[str, str, str, str]) -> str:
    return f"{row[0]}\t{row[1]}"


def score_histogram(scores: list[float]) -> list[int]:
    histogram = [0] * HISTOGRAM_BINS
    for value in scores:
        index = min(HISTOGRAM_BINS - 1, max(0, int(value * HISTOGRAM_BINS)))
        histogram[index] += 1
    return histogram


def validate_existing(done_path: Path, rows, shard: int, first_global: int) -> None:
    metadata = json.loads(done_path.read_text(encoding="utf-8"))
    manifest = done_path.with_name(f"shard_{shard:05d}.jsonl.gz")
    expected = {
        "rows": len(rows),
        "first_global_row": first_global,
        "last_global_row": first_global + len(rows) - 1,
        "first_source_key": source_key(rows[0]),
        "last_source_key": source_key(rows[-1]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"resume mismatch in {done_path}: {key}={metadata.get(key)!r} != {value!r}"
            )
    if not manifest.is_file() or file_sha256(manifest) != metadata["manifest_sha256"]:
        raise RuntimeError(f"resume hash mismatch for {manifest}")


def write_shard(
    output: Path,
    family: str,
    mode: str,
    shard: int,
    first_global: int,
    rows,
    scores,
    started: float,
) -> dict:
    stem = f"shard_{shard:05d}"
    temporary = output / f".{stem}.jsonl.gz.tmp"
    manifest = output / f"{stem}.jsonl.gz"
    done = output / f"{stem}.done.json"
    content_digest = hashlib.sha256()
    flag_counts = Counter()
    source_counts = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for offset, (row, value) in enumerate(zip(rows, scores)):
            row_flags = flags(row[2], row[3])
            flag_counts.update(name for name, state in row_flags.items() if state)
            source_counts[row[0]] += 1
            record = {
                "global_row": first_global + offset,
                "source": row[0],
                "row_id": row[1],
                "left": row[2],
                "right": row[3],
                "score": value,
                "flags": row_flags,
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            content_digest.update(line.encode("utf-8"))
    temporary.replace(manifest)
    finite_scores = [float(value) for value in scores if value is not None]
    if finite_scores and not all(math.isfinite(value) for value in finite_scores):
        raise RuntimeError(f"non-finite score in shard {shard}")
    payload = {
        "claim": "NIO-NONPAR-FULL-C01",
        "family": family,
        "mode": mode,
        "shard": shard,
        "rows": len(rows),
        "first_global_row": first_global,
        "last_global_row": first_global + len(rows) - 1,
        "first_source_key": source_key(rows[0]),
        "last_source_key": source_key(rows[-1]),
        "source_counts": dict(sorted(source_counts.items())),
        "flags": {name: flag_counts[name] for name in flags("a", "b")},
        "finite_scores": len(finite_scores),
        "score_min": min(finite_scores) if finite_scores else None,
        "score_max": max(finite_scores) if finite_scores else None,
        "score_sum": sum(finite_scores) if finite_scores else None,
        "score_histogram_1000": score_histogram(finite_scores) if finite_scores else None,
        "manifest_sha256": file_sha256(manifest),
        "content_sha256": content_digest.hexdigest(),
        "elapsed_total_seconds": time.time() - started,
    }
    done.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "shard_complete",
                "family": family,
                "shard": shard,
                "rows": len(rows),
                "total_rows": payload["last_global_row"],
                "elapsed_total_seconds": payload["elapsed_total_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mono", "qa", "medical"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="/home/nio/models/bge-reranker-v2-m3")
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    mode = "integrity_only" if args.family == "mono" else "relation_score"
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tokenizer = model = None
    if mode == "relation_score":
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, torch_dtype=torch.float16, local_files_only=True
        ).to(args.device).eval()

    rows = []
    shard = 0
    total = 0
    source_counts = Counter()

    def flush() -> None:
        nonlocal rows, shard
        if not rows:
            return
        first_global = total - len(rows) + 1
        done = args.output / f"shard_{shard:05d}.done.json"
        if done.exists():
            validate_existing(done, rows, shard, first_global)
            print(json.dumps({"event": "shard_skip", "shard": shard}), flush=True)
        else:
            if mode == "relation_score":
                values = score_pairs(
                    model,
                    tokenizer,
                    [(row[2], row[3]) for row in rows],
                    args.batch,
                    args.max_length,
                    args.device,
                )
            else:
                values = [None] * len(rows)
            write_shard(
                args.output, args.family, mode, shard, first_global, rows, values, started
            )
        rows = []
        shard += 1

    stop = False
    for _, source_rows in family_sources(args.family):
        for row in source_rows:
            if args.max_rows and total >= args.max_rows:
                stop = True
                break
            rows.append(row)
            total += 1
            source_counts[row[0]] += 1
            if len(rows) >= args.shard_rows:
                flush()
        if stop:
            break
    flush()

    complete = {
        "claim": "NIO-NONPAR-FULL-C01",
        "family": args.family,
        "mode": mode,
        "model": args.model if mode == "relation_score" else None,
        "rows": total,
        "shards": shard,
        "source_counts": dict(sorted(source_counts.items())),
        "max_rows": args.max_rows,
        "source_exhausted": args.max_rows == 0,
        "seconds": time.time() - started,
    }
    (args.output / "run_complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **complete}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
