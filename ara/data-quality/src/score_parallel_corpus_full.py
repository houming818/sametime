#!/usr/bin/env python3
"""Score a parallel corpus in restartable shards without rewriting the source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CJK = re.compile(r"[\u3400-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MOJIBAKE = ("锟", "閿", "脙", "脗", "Ã", "Â")


def row_flags(zh: str, en: str) -> dict[str, bool]:
    ratio = len(en) / max(1, len(zh))
    return {
        "mojibake": any(mark in zh or mark in en for mark in MOJIBAKE),
        "direction_suspect": (
            len(CJK.findall(zh)) / max(1, len(zh)) < 0.20
            or len(LATIN.findall(en)) / max(1, len(en)) < 0.35
        ),
        "length_ratio_suspect": ratio < 0.5 or ratio > 8.0,
        "number_mismatch": sorted(NUMBER.findall(zh)) != sorted(NUMBER.findall(en)),
    }


@torch.inference_mode()
def score(model, tokenizer, pairs, batch_size: int, max_length: int, device: str):
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[1] for item in batch],
            [item[2] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")
        ):
            logits = model(**encoded).logits.reshape(-1)
        output.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shard(output: Path, index: int, rows, scores, threshold: float, started: float):
    stem = f"shard_{index:05d}"
    manifest_tmp = output / f".{stem}.jsonl.gz.tmp"
    accepted_tmp = output / f".{stem}.accepted.tsv.gz.tmp"
    manifest = output / f"{stem}.jsonl.gz"
    accepted = output / f"{stem}.accepted.tsv.gz"
    metadata = output / f"{stem}.done.json"
    accepted_count = 0
    flag_counts = {name: 0 for name in row_flags("中", "a")}
    with gzip.open(manifest_tmp, "wt", encoding="utf-8", newline="\n") as mout, gzip.open(
        accepted_tmp, "wt", encoding="utf-8", newline="\n"
    ) as aout:
        for (line_no, zh, en), value in zip(rows, scores):
            flags = row_flags(zh, en)
            for name, state in flags.items():
                flag_counts[name] += int(state)
            record = {"line": line_no, "zh": zh, "en": en, "score": value, "flags": flags}
            mout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if value >= threshold:
                aout.write(f"{line_no}\t{value:.9g}\t{zh}\t{en}\n")
                accepted_count += 1
    manifest_tmp.replace(manifest)
    accepted_tmp.replace(accepted)
    payload = {
        "shard": index,
        "first_line": rows[0][0],
        "last_line": rows[-1][0],
        "rows": len(rows),
        "accepted": accepted_count,
        "threshold": threshold,
        "flags": flag_counts,
        "manifest_sha256": sha256(manifest),
        "accepted_sha256": sha256(accepted),
        "elapsed_total_seconds": time.time() - started,
    }
    metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "shard_complete", **payload}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="/home/nio/models/bge-reranker-v2-m3")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-download", action="store_true",
        help="Permit model downloads; run this only through the configured proxy",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16, local_files_only=local_only
    ).to(args.device).eval()
    rows, valid, shard_index = [], 0, 0
    with Path(args.data).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            valid += 1
            if args.max_rows and valid > args.max_rows:
                break
            rows.append((line_no, parts[0].strip(), parts[1].strip()))
            if len(rows) < args.shard_rows:
                continue
            done = args.output / f"shard_{shard_index:05d}.done.json"
            if not done.exists():
                write_shard(
                    args.output, shard_index, rows,
                    score(model, tokenizer, rows, args.batch, args.max_length, args.device),
                    args.threshold, started,
                )
            else:
                print(json.dumps({"event": "shard_skip", "shard": shard_index}), flush=True)
            rows, shard_index = [], shard_index + 1
    if rows:
        done = args.output / f"shard_{shard_index:05d}.done.json"
        if not done.exists():
            write_shard(
                args.output, shard_index, rows,
                score(model, tokenizer, rows, args.batch, args.max_length, args.device),
                args.threshold, started,
            )
    print(json.dumps({"event": "complete", "valid_rows": valid, "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
