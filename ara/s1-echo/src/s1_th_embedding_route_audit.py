#!/usr/bin/env python3
"""Read-only per-level routing audit for fixed-point TreeHeap checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from s1_th_embedding_fixed_point import RecurrentTreeHeap


@torch.no_grad()
def audit_run(run: Path, device: torch.device) -> Dict[str, object]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = summary["config"]
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["dim"])
    depth = int(cfg["depth"])
    rounds = int(cfg["rounds"])
    mix = float(cfg["mix"])

    model = RecurrentTreeHeap(
        torch.zeros((vocab_size, dim), device=device), depth, rounds, mix
    ).to(device)
    state_dict = torch.load(run / "treeheap_state.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    state = F.normalize(model.embedding.weight, dim=-1)
    round_rows: List[Dict[str, object]] = []
    eps = 1e-12

    for round_index in range(rounds):
        mass = torch.ones((vocab_size, 1), device=device)
        read = torch.zeros_like(state)
        internal_offset = 0
        value_offset = 0
        level_rows = []

        for level in range(depth):
            nodes = 2**level
            weight = model.route_weight[internal_offset : internal_offset + nodes]
            bias = model.route_bias[internal_offset : internal_offset + nodes]
            right = ((state @ weight.t()) / math.sqrt(dim) + bias).sigmoid()

            binary_entropy = -(
                right * (right + eps).log2()
                + (1.0 - right) * (1.0 - right + eps).log2()
            )
            effective_right = (mass * right).sum(dim=1)
            effective_branch_entropy = (mass * binary_entropy).sum(dim=1)
            saturated_mass = (
                mass * ((right < 0.05) | (right > 0.95)).to(mass.dtype)
            ).sum(dim=1)

            next_mass = torch.stack(
                (mass * (1.0 - right), mass * right), dim=-1
            ).reshape(vocab_size, -1)
            child_values = model.node_value[value_offset : value_offset + 2 * nodes]
            read = read + next_mass @ child_values

            hard = next_mass.argmax(dim=1)
            hard_counts = torch.bincount(hard, minlength=2 * nodes).to(torch.float64)
            hard_prob = hard_counts[hard_counts > 0] / vocab_size
            hard_entropy = float(
                (-(hard_prob * hard_prob.log()).sum() / math.log(2 * nodes)).item()
            )
            level_rows.append(
                {
                    "level": level + 1,
                    "hard_prefix_utilization": float(hard.unique().numel() / (2 * nodes)),
                    "hard_prefix_entropy": hard_entropy,
                    "largest_prefix_share": float((hard_counts.max() / vocab_size).item()),
                    "effective_right_mean": float(effective_right.mean().item()),
                    "effective_branch_entropy_bits": float(
                        effective_branch_entropy.mean().item()
                    ),
                    "saturated_decision_mass": float(saturated_mass.mean().item()),
                    "route_weight_rms": float(weight.square().mean().sqrt().item()),
                    "route_bias_abs_mean": float(bias.abs().mean().item()),
                    "route_bias_abs_max": float(bias.abs().max().item()),
                }
            )
            mass = next_mass
            internal_offset += nodes
            value_offset += 2 * nodes

        leaf = mass.argmax(dim=1)
        leaf_counts = torch.bincount(leaf, minlength=2**depth).to(torch.float64)
        occupied = leaf_counts[leaf_counts > 0]
        leaf_prob = occupied / vocab_size
        round_rows.append(
            {
                "round": round_index + 1,
                "leaf_utilization": float(occupied.numel() / (2**depth)),
                "leaf_entropy": float(
                    (-(leaf_prob * leaf_prob.log()).sum() / math.log(2**depth)).item()
                ),
                "largest_leaf_share": float((leaf_counts.max() / vocab_size).item()),
                "levels": level_rows,
            }
        )
        state = F.normalize((1.0 - mix) * state + mix * read / depth, dim=-1)

    return {
        "run": run.name,
        "seed": int(cfg["seed"]),
        "rounds": rounds,
        "steps": int(cfg["steps"]),
        "pair_accuracy": summary["evaluation"]["tree_pair_accuracy"],
        "baseline_pair_accuracy": summary["evaluation"]["baseline_pair_accuracy"],
        "round_profiles": round_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path(args.base)
    device = torch.device(args.device)
    result = {
        "audit": "fixed_seed_read_only_route_profile",
        "runs": [audit_run(base / name, device) for name in args.runs],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
