#!/usr/bin/env python3
"""Aggregate the preregistered D08R1 three-seed evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CLAIM = "S3-STRUCTURAL-SLOT-OWNERSHIP-D08R1"
SEEDS = (10811, 10812, 10813)
DEPTHS = (5, 6, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()
    root = Path(args.evidence_dir)
    runs = {}
    for seed in SEEDS:
        path = root / f"formal_seed{seed}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["claim"] != CLAIM or payload["config"]["seed"] != seed:
            raise RuntimeError(f"formal contract mismatch: {path}")
        if payload["config"]["steps"] != 3000:
            raise RuntimeError(f"step contract mismatch: {path}")
        if payload["rows"] != {"train": 20000, "valid": 512, "test": 512}:
            raise RuntimeError(f"row contract mismatch: {path}")
        runs[str(seed)] = payload

    implementation_all = all(
        run["gates"]["P0_contracts"] and run["gates"]["P1_ownership"]
        and run["gates"]["P4_rate_direction"] and run["gates"]["P5_trainability"]
        for run in runs.values()
    )
    quality_seed_passes = sum(run["gates"]["P2_quality"] for run in runs.values())
    causality_seed_passes = sum(run["gates"]["P3_input_causality"] for run in runs.values())

    depth_metrics = {}
    for depth in DEPTHS:
        key = str(depth)
        margins_free = [run["quality_nll_margins"][key]["vs_free"] for run in runs.values()]
        margins_random = [run["quality_nll_margins"][key]["vs_random"] for run in runs.values()]
        shuffle = [run["arms"]["subheap"]["causal_nll_deltas"][key]["shuffle"] for run in runs.values()]
        zero = [run["arms"]["subheap"]["causal_nll_deltas"][key]["zero"] for run in runs.values()]
        bleu = [
            run["arms"]["subheap"]["final_test"][key]["generation"]["token_bleu4"]
            for run in runs.values()
        ]
        depth_metrics[key] = {
            "vs_free": margins_free, "vs_free_median": statistics.median(margins_free),
            "vs_random": margins_random, "vs_random_median": statistics.median(margins_random),
            "shuffle_delta": shuffle, "shuffle_delta_median": statistics.median(shuffle),
            "zero_delta": zero, "zero_delta_median": statistics.median(zero),
            "token_bleu4": bleu, "token_bleu4_median": statistics.median(bleu),
        }

    median_quality_depths = sum(
        row["vs_free_median"] >= 0.02 and row["vs_random_median"] >= 0.02
        for row in depth_metrics.values()
    )
    median_causality_depths = sum(
        row["shuffle_delta_median"] >= 0.10 and row["zero_delta_median"] >= 0.10
        for row in depth_metrics.values()
    )
    formal_supported = (
        implementation_all and quality_seed_passes >= 2 and causality_seed_passes >= 2
        and median_quality_depths >= 2 and median_causality_depths >= 2
    )
    summary = {
        "claim": CLAIM,
        "decision": "formal_supported" if formal_supported else "formal_not_supported",
        "seeds": list(SEEDS),
        "implementation_all": implementation_all,
        "quality_seed_passes": quality_seed_passes,
        "causality_seed_passes": causality_seed_passes,
        "median_quality_depths": median_quality_depths,
        "median_causality_depths": median_causality_depths,
        "depth_metrics": depth_metrics,
        "seed_decisions": {seed: run["decision"] for seed, run in runs.items()},
    }
    path = root / "formal_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
