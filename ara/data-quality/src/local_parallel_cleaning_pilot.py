#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MOJIBAKE = ("�", "锟斤拷", "鈦", "Ã", "Â", "æ", "å")
CJK = re.compile(r"[\u3400-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_rows(path: Path, wanted: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    valid = 0
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            zh, en = parts[0].strip(), parts[1].strip()
            valid += 1
            row = (line_no, zh, en)
            if len(reservoir) < wanted:
                reservoir.append(row)
            else:
                index = rng.randrange(valid)
                if index < wanted:
                    reservoir[index] = row
    reservoir.sort(key=lambda row: row[0])
    for line_no, zh, en in reservoir:
        digest.update(f"{line_no}\t{zh}\t{en}\n".encode("utf-8"))
    return reservoir, valid, digest.hexdigest()


def flags(zh: str, en: str, seen: Counter):
    key = hashlib.sha1((zh + "\0" + en).encode()).hexdigest()
    seen[key] += 1
    zh_cjk = len(CJK.findall(zh)) / max(1, len(zh))
    en_latin = len(LATIN.findall(en)) / max(1, len(en))
    ratio = len(en) / max(1, len(zh))
    return {
        "mojibake": any(mark in zh or mark in en for mark in MOJIBAKE),
        "direction_suspect": zh_cjk < 0.20 or en_latin < 0.35,
        "length_ratio_suspect": ratio < 0.5 or ratio > 8.0,
        "duplicate": seen[key] > 1,
        "number_mismatch": sorted(NUMBER.findall(zh)) != sorted(NUMBER.findall(en)),
    }


def shuffled_targets(rows, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for index, (_, zh, en) in enumerate(rows):
        bucket = min(12, int(math.log2(max(1, len(en)))))
        buckets[bucket].append(index)
    output = [None] * len(rows)
    for indexes in buckets.values():
        targets = [rows[index][2] for index in indexes]
        if len(targets) > 1:
            shift = rng.randrange(1, len(targets))
            targets = targets[shift:] + targets[:shift]
        for index, target in zip(indexes, targets):
            output[index] = target
    return output


@torch.inference_mode()
def score_pairs(model, tokenizer, pairs, batch, max_length, device):
    scores = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        encoded = tokenizer(
            [row[0] for row in chunk], [row[1] for row in chunk],
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            logits = model(**encoded).logits.reshape(-1)
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return scores


def auc(positive, negative):
    values = [(value, 1) for value in positive] + [(value, 0) for value in negative]
    values.sort(key=lambda row: row[0])
    rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in values[index:end])
        index = end
    return (rank_sum - len(positive) * (len(positive) + 1) / 2) / max(1, len(positive) * len(negative))


def quantiles(values):
    ordered = sorted(values)
    return {str(q): ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] for q in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=14101)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    started = time.time()
    rows, corpus_valid, sample_hash = sample_rows(Path(args.data), args.rows, args.seed)
    shuffled = shuffled_targets(rows, args.seed + 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(args.device).eval()
    observed = score_pairs(model, tokenizer, [(zh, en) for _, zh, en in rows], args.batch, args.max_length, args.device)
    negative = score_pairs(model, tokenizer, [(zh, en) for (_, zh, _), en in zip(rows, shuffled)], args.batch, args.max_length, args.device)
    seen = Counter()
    records = []
    reject_threshold = sorted(observed)[max(0, int(.01 * len(observed)) - 1)]
    gray_threshold = sorted(observed)[max(0, int(.10 * len(observed)) - 1)]
    for row, score, control in zip(rows, observed, negative):
        line_no, zh, en = row
        row_flags = flags(zh, en, seen)
        bucket = "reject_candidate" if score <= reject_threshold else "gray" if score <= gray_threshold else "clean_candidate"
        records.append({"line": line_no, "zh": zh, "en": en, "score": score, "shuffle_score": control, "bucket": bucket, "flags": row_flags})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    review = sorted(records, key=lambda row: row["score"])
    review_set = review[:200] + review[len(review)//2-50:len(review)//2+50] + review[-100:]
    with (output / "manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for record in review_set:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    elapsed = time.time() - started
    summary = {
        "claim": "LOCAL-PARALLEL-CLEANING-PILOT-001",
        "model": args.model, "rows": len(rows), "corpus_valid_rows": corpus_valid,
        "sample_sha256": sample_hash, "seed": args.seed,
        "observed_quantiles": quantiles(observed), "shuffle_quantiles": quantiles(negative),
        "shuffle_auc": auc(observed, negative),
        "candidate_thresholds_unvalidated": {"reject": reject_threshold, "gray": gray_threshold},
        "buckets": dict(Counter(row["bucket"] for row in records)),
        "flags": {name: sum(row["flags"][name] for row in records) for name in records[0]["flags"]},
        "seconds": elapsed, "pairs_scored_per_second": 2 * len(rows) / elapsed,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "warning": "Observed rows are not ground-truth positives; thresholds require manual calibration before filtering.",
    }
    dump(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
