#!/usr/bin/env python3
"""Qwen calibration labels over source- and score-stratified shadow pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_qwen_judge_smoke import (
    FAMILIES,
    file_sha256,
    generate_one,
    parse_exact_json,
    prompt_for,
)


RELATIONS = {"matched", "partial", "mismatch", "uncertain"}
QUALITIES = {"usable", "noisy", "corrupt", "uncertain"}
RISKS = {"ordinary", "medical_unverified"}


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_family(rows: list[dict], wanted: int, seed: int) -> list[dict]:
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    strata = {}
    for source, source_rows in by_source.items():
        source_rows.sort(key=lambda row: (float(row["score"]), row["row_id"]))
        total = len(source_rows)
        for index, row in enumerate(source_rows):
            score_bin = min(9, index * 10 // total)
            item = dict(row)
            item["score_bin"] = score_bin
            item["source_rank_percentile"] = index / max(1, total - 1)
            strata.setdefault((source, score_bin), []).append(item)

    keys = sorted(strata)
    base, remainder = divmod(wanted, len(keys))
    rng = random.Random(seed)
    sample = []
    for index, key in enumerate(keys):
        quota = base + int(index < remainder)
        candidates = list(strata[key])
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise RuntimeError(f"stratum {key} has {len(candidates)} rows, needs {quota}")
        sample.extend(candidates[:quota])
    rng.shuffle(sample)
    if len(sample) != wanted:
        raise RuntimeError(f"sampled {len(sample)}/{wanted}")
    return sample


def sample_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['family']}\t{row['source']}\t{row['row_id']}\t{row['score']:.12g}\n".encode("utf-8")
        )
    return digest.hexdigest()


def validate(parsed: dict | None, family: str) -> bool:
    if parsed is None or set(parsed) != {
        "relation", "text_quality", "domain_risk", "reason_code", "reason_zh"
    }:
        return False
    if parsed["relation"] not in RELATIONS or parsed["text_quality"] not in QUALITIES:
        return False
    if parsed["domain_risk"] not in RISKS:
        return False
    expected_risk = "medical_unverified" if family == "medical" else "ordinary"
    return (
        parsed["domain_risk"] == expected_risk
        and isinstance(parsed["reason_code"], str)
        and isinstance(parsed["reason_zh"], str)
        and bool(parsed["reason_code"].strip())
        and bool(parsed["reason_zh"].strip())
    )


def derived(parsed: dict | None) -> dict[str, bool]:
    if parsed is None:
        return {"strict_usable": False, "acceptable": False, "mismatch": False}
    return {
        "strict_usable": parsed["relation"] == "matched" and parsed["text_quality"] == "usable",
        "acceptable": parsed["relation"] in {"matched", "partial"} and parsed["text_quality"] != "corrupt",
        "mismatch": parsed["relation"] == "mismatch",
    }


def bin_report(items: list[dict]) -> list[dict]:
    report = []
    for score_bin in range(10):
        bucket = [item for item in items if item["score_bin"] == score_bin]
        valid = [item for item in bucket if item["schema_valid"]]
        report.append({
            "score_bin": score_bin,
            "rows": len(bucket),
            "mean_bge": sum(item["bge_score"] for item in bucket) / max(1, len(bucket)),
            "schema_valid": len(valid),
            "strict_usable_rate": sum(item["derived"]["strict_usable"] for item in valid) / max(1, len(valid)),
            "acceptable_rate": sum(item["derived"]["acceptable"] for item in valid) / max(1, len(valid)),
            "mismatch_rate": sum(item["derived"]["mismatch"] for item in valid) / max(1, len(valid)),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-family", type=int, default=500)
    parser.add_argument("--seed", type=int, default=15103)
    parser.add_argument("--max-input", type=int, default=4096)
    parser.add_argument("--max-output", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    samples = []
    source_hashes_before = {}
    for family in FAMILIES:
        family_dir = args.pool_root / f"{family}_shadow100k_seed15103"
        path = family_dir / "manifest.jsonl"
        scorer_summary = json.loads((family_dir / "summary.json").read_text(encoding="utf-8"))
        if scorer_summary.get("rows") != 100_000 or not all(scorer_summary.get("gates", {}).values()):
            raise RuntimeError(f"{family} scorer gate did not pass: {scorer_summary.get('gates')}")
        source_hashes_before[str(path)] = file_sha256(path)
        rows = read_manifest(path)
        if len(rows) != 100_000:
            raise RuntimeError(f"{family} pool has {len(rows)} rows")
        family_sample = sample_family(rows, args.rows_per_family, args.seed + len(samples))
        for row in family_sample:
            row["family"] = family
        samples.extend(family_sample)
        del rows

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    judgments = []
    for position, record in enumerate(samples):
        raw = generate_one(model, tokenizer, prompt_for(record), args.max_input, args.max_output)
        parsed = parse_exact_json(raw)
        schema_valid = validate(parsed, record["family"])
        judgments.append({
            "position": position,
            "family": record["family"],
            "source": record["source"],
            "row_id": record["row_id"],
            "score_bin": record["score_bin"],
            "source_rank_percentile": record["source_rank_percentile"],
            "bge_score": record["score"],
            "flags": record["flags"],
            "left": record["left"],
            "right": record["right"],
            "raw_output": raw,
            "parsed": parsed,
            "schema_valid": schema_valid,
            "derived": derived(parsed if schema_valid else None),
        })
        if (position + 1) % 25 == 0:
            print(json.dumps({"done": position + 1, "total": len(samples)}), flush=True)

    source_hashes_after = {path: file_sha256(Path(path)) for path in source_hashes_before}
    schema_valid = sum(item["schema_valid"] for item in judgments)
    reports = {
        family: bin_report([item for item in judgments if item["family"] == family])
        for family in FAMILIES
    }
    summary = {
        "claims": ["NIO-NONPAR-CAL-C01", "NIO-NONPAR-LADDER-C01"],
        "model": str(args.model),
        "seed": args.seed,
        "rows": len(judgments),
        "rows_per_family": args.rows_per_family,
        "sample_sha256": sample_sha256(samples),
        "schema_valid": schema_valid,
        "schema_valid_rate": schema_valid / len(judgments),
        "source_hashes": source_hashes_before,
        "source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "bin_reports": reports,
        "label_counts": {
            family: dict(Counter(
                item["parsed"]["relation"] if item["schema_valid"] else "invalid"
                for item in judgments if item["family"] == family
            )) for family in FAMILIES
        },
        "gates": {
            "C0_exact_rows": len(judgments) == 1500,
            "C1_schema_1470": schema_valid >= 1470,
            "C2_medical_unverified": all(
                item["schema_valid"] and item["parsed"]["domain_risk"] == "medical_unverified"
                for item in judgments if item["family"] == "medical"
            ),
            "C3_source_unchanged": source_hashes_before == source_hashes_after,
            "C4_all_bins_reported": all(len(report) == 10 and all(row["rows"] > 0 for row in report) for report in reports.values()),
        },
        "seconds": time.time() - started,
        "warning": "Qwen calibration labels are not human ground truth or medical certification.",
    }
    with (args.output / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for item in judgments:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
