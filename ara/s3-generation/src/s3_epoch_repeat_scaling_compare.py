#!/usr/bin/env python3
"""Compare complete paired repeat-corpus learning curves without early stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_CURSOR = 7_304_358
SAFETY_GATES = (
    "finite", "source_frozen", "model_updated", "optimizer_loaded",
    "step_advanced", "cursor_advanced", "cursor_bounded", "reload",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    arms = {}
    for name in ("treeheap-63m", "treeheap-106m"):
        summary = read(root / name / "summary.json")
        wakes = read_jsonl(root / name / "wakes.jsonl")
        arms[name] = {"summary": summary, "wakes": wakes}

    base = arms["treeheap-63m"]["summary"]
    scale = arms["treeheap-106m"]["summary"]
    contracts = {
        name: read(root / name / "contract.json")
        for name in ("treeheap-63m", "treeheap-106m")
    }
    matched = {
        "target_cursor": base["cursor"] == scale["cursor"] == TARGET_CURSOR,
        "source": contracts["treeheap-63m"]["source_sha256"] == contracts["treeheap-106m"]["source_sha256"],
        "warm_start": contracts["treeheap-63m"]["warm_start_sha256"] == contracts["treeheap-106m"]["warm_start_sha256"],
        "parallel_data": contracts["treeheap-63m"]["parallel_sha256"] == contracts["treeheap-106m"]["parallel_sha256"],
        "start_cursor": contracts["treeheap-63m"]["start"]["cursor"] == contracts["treeheap-106m"]["start"]["cursor"] == 400_488,
        "batch": contracts["treeheap-63m"]["config"]["batch_size"] == contracts["treeheap-106m"]["config"]["batch_size"],
        "direction": contracts["treeheap-63m"]["direction_flip"] == contracts["treeheap-106m"]["direction_flip"] == 0,
    }
    safety = {
        name: all(bool(row["summary"]["gates"].get(gate)) for gate in SAFETY_GATES)
        for name, row in arms.items()
    }
    base_nll = base["valid"]["mean_nll"]
    scale_nll = scale["valid"]["mean_nll"]
    base_gen = base["generation"]
    scale_gen = scale["generation"]
    result = {
        "claim": "S3-EPOCH-REPEAT-SCALING-E01",
        "mode": "complete-second-pass-paired",
        "experiment_valid": all(matched.values()) and all(safety.values()),
        "matched_contract": matched,
        "safety": safety,
        "metrics": {
            "base_nll": base_nll,
            "scale_nll": scale_nll,
            "nll_gain_63m_minus_106m": base_nll - scale_nll,
            "base_bleu4": base_gen["bleu4_median"],
            "scale_bleu4": scale_gen["bleu4_median"],
            "bleu4_gain": scale_gen["bleu4_median"] - base_gen["bleu4_median"],
            "base_repetition": base_gen["repetition_max"],
            "scale_repetition": scale_gen["repetition_max"],
            "base_nonempty": base_gen["nonempty_min"],
            "scale_nonempty": scale_gen["nonempty_min"],
            "scale_extra_logit_gain": scale["extra_logit_gain"],
        },
        "curves": {
            name: [{
                "step": wake["step"], "cursor": wake["cursor"],
                "nll": wake["valid"]["mean_nll"],
                "bleu4": wake["generation"]["bleu4_median"],
                "repetition": wake["generation"]["repetition_max"],
                "nonempty": wake["generation"]["nonempty_min"],
            } for wake in row["wakes"]]
            for name, row in arms.items()
        },
        "interpretation_rule": (
            "Interpret the complete curves; temporary metric reversals are observations, "
            "not failed runs. Completion does not automatically authorize 150M."
        ),
    }
    write = root / "comparison.json"
    write.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["experiment_valid"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()

