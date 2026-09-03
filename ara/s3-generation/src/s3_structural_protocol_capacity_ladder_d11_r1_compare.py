#!/usr/bin/env python3
"""Complete the fixed-budget D11 comparison without metric-based early stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SAFETY_GATES = (
    "steps_complete",
    "finite",
    "input_causality",
    "structure",
    "source_frozen",
    "trainable_updated",
    "reload",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safety(summary: dict) -> dict[str, bool]:
    return {name: bool(summary["gates"].get(name)) for name in SAFETY_GATES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    base = read(root / "treeheap-63m" / "summary.json")
    scale = read(root / "treeheap-106m" / "summary.json")
    base_contract = read(root / "treeheap-63m" / "contract.json")
    scale_contract = read(root / "treeheap-106m" / "contract.json")

    initial_delta = abs(
        base["initial_valid"]["mean_nll"] - scale["initial_valid"]["mean_nll"]
    )
    contract = {
        "same_source": base_contract["source_sha256"] == scale_contract["source_sha256"],
        "same_warm_start": base_contract["warm_start_sha256"] == scale_contract["warm_start_sha256"],
        "same_data": base_contract["parallel_sha256"] == scale_contract["parallel_sha256"],
        "same_seed": base_contract["seed"] == scale_contract["seed"],
        "same_steps": base["step"] == scale["step"] == 25_000,
        "same_cursor": base["cursor"] == scale["cursor"],
        "initial_function_parity": initial_delta <= 1e-8,
        "parameter_order": scale["parameters"]["total"] > base["parameters"]["total"],
    }
    base_safety = safety(base)
    scale_safety = safety(scale)
    experiment_valid = (
        all(contract.values()) and all(base_safety.values()) and all(scale_safety.values())
    )

    base_nll = base["best_valid"]["mean_nll"]
    scale_nll = scale["best_valid"]["mean_nll"]
    base_generation = base["best_generation"]
    scale_generation = scale["best_generation"]
    nll_gain = base_nll - scale_nll
    bleu_gain = scale_generation["bleu4_median"] - base_generation["bleu4_median"]
    repetition_delta = (
        scale_generation["repetition_max"] - base_generation["repetition_max"]
    )
    nonempty_delta = scale_generation["nonempty_min"] - base_generation["nonempty_min"]
    quality = {
        "nll_gain_ge_0_03": nll_gain >= 0.03,
        "bleu_noninferior": bleu_gain >= -0.50,
        "repetition_noninferior": repetition_delta <= 0.02,
        "nonempty_noninferior": nonempty_delta >= 0.0,
    }
    supported = experiment_valid and all(quality.values())
    decision = (
        "invalid_experiment"
        if not experiment_valid
        else "first_scale_rung_supported"
        if supported
        else "first_scale_rung_not_supported"
    )
    result = {
        "claim": "S3-STRUCTURAL-PROTOCOL-CAPACITY-LADDER-D11-R1",
        "mode": "formal-fixed-budget-recovery",
        "decision": decision,
        "experiment_valid": experiment_valid,
        "contract": contract,
        "base_safety": base_safety,
        "scale_safety": scale_safety,
        "quality_gates": quality,
        "metrics": {
            "initial_nll_delta": initial_delta,
            "base_nll": base_nll,
            "scale_nll": scale_nll,
            "nll_gain": nll_gain,
            "base_bleu4": base_generation["bleu4_median"],
            "scale_bleu4": scale_generation["bleu4_median"],
            "bleu4_gain": bleu_gain,
            "base_repetition": base_generation["repetition_max"],
            "scale_repetition": scale_generation["repetition_max"],
            "repetition_delta": repetition_delta,
            "base_nonempty": base_generation["nonempty_min"],
            "scale_nonempty": scale_generation["nonempty_min"],
            "nonempty_delta": nonempty_delta,
            "base_parameters": base["parameters"],
            "scale_parameters": scale["parameters"],
        },
        "next": "analyze the completed rung; do not automatically start 150M",
    }
    output = root / "comparison_r1.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not experiment_valid:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
