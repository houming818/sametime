#!/usr/bin/env python3
"""Build nested, coverage-preserving selected and raw-control data orders."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


FAMILY_SIZES = {
    "mono": (10_000, 20_000, 40_000, 80_000),
    "qa": (10_000, 20_000, 40_000, 80_000),
    "medical": (5_000, 10_000, 20_000, 40_000),
}
HARD_FLAGS = ("mojibake", "extreme_repetition", "too_short")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict) -> tuple[str, str]:
    return row["source"], row["row_id"]


def length_bin(row: dict) -> int:
    return min(16, int(math.log2(max(1, len(row["left"]) + len(row["right"])))))


def stable_random(seed: int, family: str, row: dict) -> float:
    digest = hashlib.sha256(
        f"{seed}\t{family}\t{row['source']}\t{row['row_id']}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def coverage_order(rows: list[dict], family: str, seed: int, selected: bool) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        strata[(row["source"], length_bin(row))].append(row)

    ranked = []
    for stratum, bucket in strata.items():
        if selected:
            bucket.sort(key=lambda row: (
                sum(bool(row["flags"].get(name)) for name in HARD_FLAGS),
                -float(row["score"]),
                row["row_id"],
            ))
        else:
            bucket.sort(key=lambda row: (stable_random(seed, family, row), row["row_id"]))
        total = len(bucket)
        for index, row in enumerate(bucket):
            item = dict(row)
            item["length_bin"] = stratum[1]
            item["within_stratum_rank"] = index
            item["within_stratum_percentile"] = (index + 0.5) / total
            ranked.append(item)
    ranked.sort(key=lambda row: (
        row["within_stratum_percentile"], row["source"], row["length_bin"], row["row_id"]
    ))
    return ranked


def distribution(rows: list[dict], field) -> dict[str, float]:
    counts = Counter(str(field(row)) for row in rows)
    total = len(rows)
    return {key: value / total for key, value in sorted(counts.items())}


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    middle = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in keys}

    def kl(one, two):
        return sum(value * math.log(value / two[key]) for key, value in one.items() if value > 0)

    return (kl(left, middle) + kl(right, middle)) / 2


def quantiles(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        str(q): values[min(len(values) - 1, int(q * (len(values) - 1)))]
        for q in (0, .1, .25, .5, .75, .9, 1)
    }


def prefix_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['source']}\t{row['row_id']}\t{float(row['score']):.12g}\n".encode("utf-8")
        )
    return digest.hexdigest()


def write_order(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for rank, row in enumerate(rows):
            item = dict(row)
            item["global_rank"] = rank
            line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def audit_metrics(prefix: list[dict], audit: dict[tuple[str, str], dict]) -> dict:
    items = [audit[row_key(row)] for row in prefix if row_key(row) in audit]
    valid = [item for item in items if item["schema_valid"]]
    return {
        "audited_rows": len(items),
        "valid_rows": len(valid),
        "strict_usable_rate": sum(item["derived"]["strict_usable"] for item in valid) / max(1, len(valid)),
        "acceptable_rate": sum(item["derived"]["acceptable"] for item in valid) / max(1, len(valid)),
        "mismatch_rate": sum(item["derived"]["mismatch"] for item in valid) / max(1, len(valid)),
    }


def tier_metrics(prefix: list[dict], pool_source, pool_length, audit) -> dict:
    source = distribution(prefix, lambda row: row["source"])
    lengths = distribution(prefix, length_bin)
    scores = [float(row["score"]) for row in prefix]
    return {
        "rows": len(prefix),
        "prefix_sha256": prefix_sha(prefix),
        "mean_bge": sum(scores) / len(scores),
        "bge_quantiles": quantiles(scores),
        "flags": {
            name: sum(bool(row["flags"].get(name)) for row in prefix)
            for name in prefix[0]["flags"]
        },
        "source_distribution": source,
        "length_distribution": lengths,
        "source_js_vs_pool": js_divergence(source, pool_source),
        "length_js_vs_pool": js_divergence(lengths, pool_length),
        "unique_cjk_chars": len({char for row in prefix for char in row["left"] + row["right"] if "\u3400" <= char <= "\u9fff"}),
        "qwen_audit": audit_metrics(prefix, audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--qwen-judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=15104)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    judgments = read_jsonl(args.qwen_judgments)
    reports = {}
    for family, sizes in FAMILY_SIZES.items():
        source_path = args.pool_root / f"{family}_shadow100k_seed15103" / "manifest.jsonl"
        rows = read_jsonl(source_path)
        if len(rows) != 100_000:
            raise RuntimeError(f"{family} expected 100000 rows, got {len(rows)}")
        if len({row_key(row) for row in rows}) != len(rows):
            raise RuntimeError(f"{family} has duplicate source keys")
        family_audit = {
            row_key(item): item for item in judgments
            if item["family"] == family
        }
        if len(family_audit) != 500:
            raise RuntimeError(f"{family} expected 500 Qwen rows, got {len(family_audit)}")

        selected = coverage_order(rows, family, args.seed, selected=True)
        raw = coverage_order(rows, family, args.seed, selected=False)
        if {row_key(row) for row in selected} != {row_key(row) for row in raw}:
            raise RuntimeError(f"{family} orders do not cover the same rows")

        family_dir = args.output / family
        family_dir.mkdir(parents=True, exist_ok=True)
        selected_sha = write_order(family_dir / "selected_order.jsonl.gz", selected)
        raw_sha = write_order(family_dir / "raw_control_order.jsonl.gz", raw)
        pool_source = distribution(rows, lambda row: row["source"])
        pool_length = distribution(rows, length_bin)
        tiers = {}
        for size in sizes:
            selected_metrics = tier_metrics(selected[:size], pool_source, pool_length, family_audit)
            raw_metrics = tier_metrics(raw[:size], pool_source, pool_length, family_audit)
            tiers[str(size)] = {"selected": selected_metrics, "raw_control": raw_metrics}

        adjacent_quality_wins = []
        for size in sizes:
            item = tiers[str(size)]
            selected_audit = item["selected"]["qwen_audit"]
            raw_audit = item["raw_control"]["qwen_audit"]
            adjacent_quality_wins.append(
                selected_audit["strict_usable_rate"] > raw_audit["strict_usable_rate"]
                and selected_audit["mismatch_rate"] < raw_audit["mismatch_rate"]
            )
        report = {
            "family": family,
            "seed": args.seed,
            "pool_rows": len(rows),
            "source_path": str(source_path),
            "selected_order_sha256": selected_sha,
            "raw_control_order_sha256": raw_sha,
            "sizes": list(sizes),
            "tiers": tiers,
            "gates": {
                "L0_exact_rows_and_unique": len(selected) == len(raw) == 100_000,
                "L0_nested_by_prefix": True,
                "L1_source_js": all(tiers[str(size)]["selected"]["source_js_vs_pool"] <= 0.02 for size in sizes),
                "L2_length_js": all(tiers[str(size)]["selected"]["length_js_vs_pool"] <= 0.02 for size in sizes),
                "L3_L4_two_adjacent_quality_wins": any(
                    adjacent_quality_wins[index] and adjacent_quality_wins[index + 1]
                    for index in range(len(adjacent_quality_wins) - 1)
                ),
            },
            "warning": "Qwen audit rates are calibration-model estimates, not human ground truth.",
        }
        (family_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports[family] = report
        print(json.dumps({"family": family, "gates": report["gates"]}), flush=True)

    overall = {
        "claims": ["NIO-NONPAR-CAL-C01", "NIO-NONPAR-LADDER-C01"],
        "seed": args.seed,
        "families": {family: report["gates"] for family, report in reports.items()},
        "training_ladder_authorized": {
            family: all(report["gates"].values()) for family, report in reports.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
