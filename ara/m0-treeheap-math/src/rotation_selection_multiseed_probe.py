#!/usr/bin/env python3
"""Multi-seed drift audit for the rotation-selection probe."""

from __future__ import annotations

import argparse
import json
import pathlib
from types import SimpleNamespace

from rotation_selection_evolution_probe import run_probe


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> dict:
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if seeds != [1, 7, 19, 42, 73, 99, 314, 2026]:
        raise ValueError("seeds must match the preregistered list")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        seed_out = out / f"seed_{seed}"
        run_args = SimpleNamespace(
            out=str(seed_out),
            capacity=127,
            state_dim=4,
            candidates=24,
            steps=1500,
            batch_size=128,
            validation_batch_size=256,
            mask_rate=0.25,
            rho=0.92,
            temperature=0.5,
            warmup_steps=150,
            decoder_lr=0.003,
            gate_lr=0.03,
            seed=seed,
        )
        result = run_probe(run_args)
        structured = result["structured"]
        iid = result["iid"]
        structured_exact = structured["groups"]["exact"]
        iid_exact = iid["groups"]["exact"]
        iid_random = iid["groups"]["random"]
        rows.append(
            {
                "seed": seed,
                "structured_winner_kind": structured["winner"]["kind"],
                "structured_exact_mass": structured_exact["probability_mass"],
                "structured_edge_loss_pearson": structured[
                    "edge_preservation_loss_pearson"
                ],
                "iid_winner_kind": iid["winner"]["kind"],
                "iid_exact_mass": iid_exact["probability_mass"],
                "iid_edge_loss_pearson": iid["edge_preservation_loss_pearson"],
                "iid_exact_random_loss_gap": abs(
                    iid_exact["mean_validation_mse"]
                    - iid_random["mean_validation_mse"]
                ),
                "echo_max_inverse_error": result["exact_echo"][
                    "max_inverse_error"
                ],
            }
        )

    structured_winner_count = sum(
        row["structured_winner_kind"] == "exact" for row in rows
    )
    iid_exact_winner_count = sum(row["iid_winner_kind"] == "exact" for row in rows)
    aggregate = {
        "seed_count": len(rows),
        "structured_exact_winner_count": structured_winner_count,
        "structured_exact_mass_mean": mean(
            [row["structured_exact_mass"] for row in rows]
        ),
        "structured_exact_mass_min": min(
            row["structured_exact_mass"] for row in rows
        ),
        "structured_edge_loss_pearson_mean": mean(
            [row["structured_edge_loss_pearson"] for row in rows]
        ),
        "iid_exact_winner_count": iid_exact_winner_count,
        "iid_winner_histogram": {
            kind: sum(row["iid_winner_kind"] == kind for row in rows)
            for kind in ("exact", "mild", "random")
        },
        "iid_exact_mass_mean": mean([row["iid_exact_mass"] for row in rows]),
        "iid_exact_random_loss_gap_mean": mean(
            [row["iid_exact_random_loss_gap"] for row in rows]
        ),
        "iid_edge_loss_pearson_mean": mean(
            [row["iid_edge_loss_pearson"] for row in rows]
        ),
        "max_echo_inverse_error": max(
            row["echo_max_inverse_error"] for row in rows
        ),
    }
    gates = {
        "C1_structured_winners": structured_winner_count >= 7,
        "C2_structured_mass_mean": aggregate["structured_exact_mass_mean"] >= 0.90,
        "C3_structured_mass_min": aggregate["structured_exact_mass_min"] >= 0.75,
        "C4_structured_correlation": (
            aggregate["structured_edge_loss_pearson_mean"] <= -0.90
        ),
        "C5_iid_winners": iid_exact_winner_count <= 4,
        "C6_iid_mass_mean": aggregate["iid_exact_mass_mean"] <= 0.50,
        "C7_iid_loss_tie": aggregate["iid_exact_random_loss_gap_mean"] <= 0.02,
        "C8_iid_correlation": abs(
            aggregate["iid_edge_loss_pearson_mean"]
        )
        <= 0.40,
        "C9_echo_inverse": aggregate["max_echo_inverse_error"] < 1e-12,
    }
    summary = {
        "parent_predict": "P-ROT02-B",
        "predict": "P-ROT02-C",
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
        "gates": gates,
        "pilot_pass": all(gates.values()),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    readme = f"""# Rotation Selection Multi-Seed Evidence

Predict: `P-ROT02-C`

```text
pilot_pass                         = {summary['pilot_pass']}
structured exact winners          = {structured_winner_count}/8
structured exact mass mean/min    = {aggregate['structured_exact_mass_mean']:.6f} / {aggregate['structured_exact_mass_min']:.6f}
structured edge/loss Pearson mean = {aggregate['structured_edge_loss_pearson_mean']:.6f}
IID winner histogram              = {aggregate['iid_winner_histogram']}
IID exact mass mean               = {aggregate['iid_exact_mass_mean']:.6f}
IID exact/random loss gap mean    = {aggregate['iid_exact_random_loss_gap_mean']:.6f}
IID edge/loss Pearson mean        = {aggregate['iid_edge_loss_pearson_mean']:.6f}
max exact-echo inverse error      = {aggregate['max_echo_inverse_error']:.6g}
```
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
