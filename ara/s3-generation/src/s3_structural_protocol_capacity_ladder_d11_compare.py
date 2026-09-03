#!/usr/bin/env python3
"""Compare the matched D11 baseline and first capacity rung."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    base = read(root / "treeheap-63m" / "summary.json")
    scale = read(root / "treeheap-106m" / "summary.json")
    base_contract = read(root / "treeheap-63m" / "contract.json")
    scale_contract = read(root / "treeheap-106m" / "contract.json")
    initial_delta = abs(
        base["initial_valid"]["mean_nll"] - scale["initial_valid"]["mean_nll"]
    )
    nll_gain = base["best_valid"]["mean_nll"] - scale["best_valid"]["mean_nll"]
    bleu_gain = (
        scale["best_generation"]["bleu4_median"]
        - base["best_generation"]["bleu4_median"]
    )
    repetition_delta = (
        scale["best_generation"]["repetition_max"]
        - base["best_generation"]["repetition_max"]
    )
    contract = {
        "same_source": base_contract["source_sha256"] == scale_contract["source_sha256"],
        "same_warm_start": base_contract["warm_start_sha256"] == scale_contract["warm_start_sha256"],
        "same_data": base_contract["parallel_sha256"] == scale_contract["parallel_sha256"],
        "same_seed": base_contract["seed"] == scale_contract["seed"],
        "same_steps": base["step"] == scale["step"],
        "same_cursor": base["cursor"] == scale["cursor"],
        "initial_function_parity": initial_delta <= 1e-8,
        "parameter_order": scale["parameters"]["total"] > base["parameters"]["total"],
        "arms_passed": base["passed"] and scale["passed"],
    }
    if args.mode == "smoke":
        quality = {"smoke_only": True}
        supported = all(contract.values())
    else:
        quality = {
            "nll_gain_ge_0_03": nll_gain >= 0.03,
            "bleu_noninferior": bleu_gain >= -0.50,
            "repetition_noninferior": repetition_delta <= 0.02,
        }
        supported = all(contract.values()) and all(quality.values())
    result = {
        "claim": "S3-STRUCTURAL-PROTOCOL-CAPACITY-LADDER-D11",
        "mode": args.mode,
        "decision": "first_scale_rung_supported" if supported else "first_scale_rung_not_supported",
        "contract": contract, "quality_gates": quality,
        "metrics": {
            "initial_nll_delta": initial_delta,
            "base_nll": base["best_valid"]["mean_nll"],
            "scale_nll": scale["best_valid"]["mean_nll"],
            "nll_gain": nll_gain,
            "base_bleu4": base["best_generation"]["bleu4_median"],
            "scale_bleu4": scale["best_generation"]["bleu4_median"],
            "bleu4_gain": bleu_gain,
            "repetition_delta": repetition_delta,
            "base_parameters": base["parameters"],
            "scale_parameters": scale["parameters"],
        },
        "next": (
            "preregister the adjacent 150M rung"
            if supported and args.mode == "formal"
            else "stop automatic scale-up and inspect the failed gates"
        ),
    }
    (root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not supported:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
