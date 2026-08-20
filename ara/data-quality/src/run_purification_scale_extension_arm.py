#!/usr/bin/env python3
"""Validate the nested dataset identity, then run one extension arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_EXISTING_HASHES = {
    "train_40000": "826a6823ccd7d3b739e2a378682d9cf72c05eb866d997148467c3e507395cdae",
    "train_80000": "de897be34c8cabb4e3beee53619efddf2083d85c30848fcc28b8354eb2d5b95b",
    "train_120000": "3f2ba077db60f0645e195f5fa696572aab8d0b94aa65d5eceb944c638044520a",
    "train_160000": "f3c729d1802113077e5445cc04ecd15ebdc0465572867f066053f4449615fb3f",
    "train_200000": "72c4c5a76aecb5b3fc52f9b0157940ca781e0a6c24b17dfa2d531eecdcfde2ac",
    "eval": "094965f1252eab5682595f991f135c5949bc0d9283424e2f38a1251176a968e7",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(300000, 400000, 500000), required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    dataset_dir = Path(
        "ara/data-quality/evidence/purification_scale_ladder/dataset_seed14106"
    )
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    hashes = summary["sha256"]
    mismatches = {
        key: {"expected": expected, "actual": hashes.get(key)}
        for key, expected in EXPECTED_EXISTING_HASHES.items()
        if hashes.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"dataset identity mismatch: {json.dumps(mismatches)}")
    if args.rows not in summary["sizes"]:
        raise RuntimeError(f"missing registered dataset arm: {args.rows}")

    command = [
        sys.executable,
        "ara/s3-generation/src/s3_bounded_annealing_fold_c13_train.py",
        "--arm", "ref_zero",
        "--config-checkpoint",
        "ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/"
        "pilot_seed10101/pretrain/checkpoint_best.pt",
        "--evidence-dir",
        f"ara/data-quality/evidence/purification_scale_ladder/formal_seed14108/{args.rows}",
        "--wmt-data", str(dataset_dir / f"purified_{args.rows}.tsv"),
        "--eval-wmt-data", str(dataset_dir / "shared_eval.tsv"),
        "--seed", "14108",
        "--steps", str(args.steps),
        "--train-rows", str(args.rows),
        "--eval-rows", "1024",
        "--batch-size", "16",
        "--lr", "0.002",
        "--log-every", str(max(1, args.steps // 5)),
        "--generation-rows", "1024",
    ]
    print(json.dumps({"validated": True, "rows": args.rows, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
