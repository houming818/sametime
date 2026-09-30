#!/usr/bin/env python3
"""Post-hoc baselines for the A19 recursive context codec experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


EPS = 1e-12


def pair_nll(probability: torch.Tensor, counts: torch.Tensor) -> float:
    log_probability = probability.clamp_min(EPS).log()
    return float((-(counts * log_probability).sum() / counts.sum()).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    formal = Path(args.formal)
    material = json.loads((formal / "material.json").read_text(encoding="utf-8"))
    config = material["config"]
    payload = torch.load(args.counts, map_location="cpu", weights_only=False)
    original_train = payload["train"].to(torch.float64)
    sealed_test = payload["test"].to(torch.float64)

    generator = torch.Generator(device="cpu").manual_seed(int(config["split_seed"]))
    fit_ratio = float(config["fit_ratio"])
    fit = torch.binomial(
        original_train,
        torch.full_like(original_train, fit_ratio),
        generator=generator,
    )
    alpha = float(config["alpha"])
    width = fit.shape[1]

    uniform = torch.full((width,), 1.0 / width, dtype=torch.float64)
    global_probability = (fit.sum(dim=0) + alpha) / (fit.sum() + alpha * width)
    row_probability = (fit + alpha) / (fit.sum(dim=1, keepdim=True) + alpha * width)

    uniform_nll = pair_nll(uniform.unsqueeze(0), sealed_test)
    global_nll = pair_nll(global_probability.unsqueeze(0), sealed_test)
    row_nll = pair_nll(row_probability, sealed_test)

    rows = []
    for dimension in (2, 4, 8, 16):
        result = json.loads((formal / f"dimension_{dimension}.json").read_text(encoding="utf-8"))
        full_nll = float(result["final"]["test"]["nll"])
        shuffled_nll = float(result["controls"]["shuffled_root_test"]["nll"])
        total_gain = uniform_nll - full_nll
        root_conditioning_gain = shuffled_nll - full_nll
        rows.append(
            {
                "dimension": dimension,
                "full_test_nll": full_nll,
                "shuffled_root_test_nll": shuffled_nll,
                "uniform_to_full_gain": total_gain,
                "root_conditioning_gain": root_conditioning_gain,
                "root_conditioning_fraction_of_total_gain": (
                    root_conditioning_gain / total_gain if total_gain > 0.0 else 0.0
                ),
                "full_minus_global_prior_nll": full_nll - global_nll,
                "full_minus_direct_row_field_nll": full_nll - row_nll,
            }
        )

    audit = {
        "status": "post_hoc_diagnostic_not_preregistered_gate",
        "uniform_test_nll": uniform_nll,
        "uniform_test_ppl": math.exp(uniform_nll),
        "global_context_prior_test_nll": global_nll,
        "global_context_prior_test_ppl": math.exp(global_nll),
        "direct_row_probability_field_test_nll": row_nll,
        "direct_row_probability_field_test_ppl": math.exp(row_nll),
        "global_prior_gain_over_uniform": uniform_nll - global_nll,
        "direct_row_gain_over_global_prior": global_nll - row_nll,
        "dimensions": rows,
        "interpretation_boundary": (
            "The global prior is the best measured address-only baseline under the registered "
            "smoothing. The direct row field is a non-recursive information ceiling, not a fair "
            "TreeHeap competitor."
        ),
    }
    out = Path(args.out)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
