#!/usr/bin/env python3
"""Aggregate C12 arms and enforce the matched-experiment identity gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("c10", "read", "read_up")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.run_dir)
    summaries = {
        arm: json.loads((root / arm / "summary.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    stream_hashes = {row["stream_sha256"] for row in summaries.values()}
    parent_hashes = {row["parent_state_sha256"] for row in summaries.values()}
    row_hashes = {
        split: {row["row_sha256"][split] for row in summaries.values()}
        for split in ("train", "valid", "test")
    }
    read_initializations = {
        summaries[arm]["initial_read_state_sha256"] for arm in ("read", "read_up")
    }
    p0 = {
        "one_parent_state": len(parent_hashes) == 1,
        "one_training_stream": len(stream_hashes) == 1,
        "one_row_set_per_split": all(len(values) == 1 for values in row_hashes.values()),
        "matched_read_initialization": len(read_initializations) == 1,
        "all_finite": all(row["finite"] for row in summaries.values()),
    }

    read = summaries["read"]
    read_up = summaries["read_up"]
    read_depths = read["depth_ablation_deltas"][:-1]
    p1 = {
        "read_leaf_only_delta_ge_0_05": read["intervention_deltas"].get("leaf_only", 0.0) >= 0.05,
        "read_two_nonleaf_depths_ge_0_01": sum(value >= 0.01 for value in read_depths) >= 2,
        "read_up_bypass_delta": read_up["intervention_deltas"].get("bypass_up", 0.0),
    }
    metrics = {
        arm: {
            "test_nll": row["test"]["nll"],
            "token_bleu4": row["generation"]["token_bleu4"],
            "adjacent_repetition_rate": row["generation"]["adjacent_repetition_rate"],
            "mean_max_token_share": row["generation"]["mean_max_token_share"],
            "parameters": row["parameters"],
            "seconds": row["seconds"],
        }
        for arm, row in summaries.items()
    }
    comparison = {
        "claim": "S3-MULTILEVEL-READ-ABLATION-C12",
        "arms": list(ARMS),
        "p0": p0,
        "p0_pass": all(p0.values()),
        "p1": p1,
        "p1_pass": p1["read_leaf_only_delta_ge_0_05"] and p1["read_two_nonleaf_depths_ge_0_01"],
        "metrics": metrics,
        "delta_read_minus_c10": {
            key: metrics["read"][key] - metrics["c10"][key]
            for key in ("test_nll", "token_bleu4", "adjacent_repetition_rate", "mean_max_token_share", "seconds")
        },
        "delta_read_up_minus_read": {
            key: metrics["read_up"][key] - metrics["read"][key]
            for key in ("test_nll", "token_bleu4", "adjacent_repetition_rate", "mean_max_token_share", "seconds")
        },
    }
    output = Path(args.output) if args.output else root / "comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False), flush=True)
    if not comparison["p0_pass"]:
        raise SystemExit("C12 experimental identity gate failed")


if __name__ == "__main__":
    main()
