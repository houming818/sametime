#!/usr/bin/env python3
"""Compare local normalization, recursive energy carrying, and naive products."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


EPSILON = 1e-8


def expand_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.empty(
        left.shape[0] * 2,
        left.shape[1],
        dtype=left.dtype,
        device=left.device,
    )
    result[0::2], result[1::2] = left, right
    return result


def local_pair(left: torch.Tensor, right: torch.Tensor, product: bool = False):
    left_norm = left.norm(dim=-1, keepdim=True)
    right_norm = right.norm(dim=-1, keepdim=True)
    if product:
        scale = torch.sqrt(left_norm * right_norm + EPSILON)
    else:
        scale = torch.sqrt(left_norm.square() + right_norm.square() + EPSILON)
    parent = (left + right) / (math.sqrt(2.0) * scale)
    detail = (right - left) / (math.sqrt(2.0) * scale)
    return parent, detail, scale


def fold_local(leaf: torch.Tensor, product: bool = False):
    levels = [leaf]
    details, scales = [], []
    node = leaf
    while node.shape[0] > 1:
        node, detail, scale = local_pair(node[0::2], node[1::2], product)
        levels.append(node)
        details.append(detail)
        scales.append(scale)
    return levels, details, scales


def unfold_local(root: torch.Tensor, details, scales):
    node = root
    for detail, scale in zip(reversed(details), reversed(scales)):
        left = scale * (node - detail) / math.sqrt(2.0)
        right = scale * (node + detail) / math.sqrt(2.0)
        node = expand_pair(left, right)
    return node


def fold_carrier(leaf: torch.Tensor):
    energy = torch.sqrt(leaf.square().sum(-1, keepdim=True) + EPSILON)
    direction = leaf / energy
    direction_levels = [direction]
    energy_levels = [energy]
    details = []
    child_energies = []
    ratios = []
    while direction.shape[0] > 1:
        left_u, right_u = direction[0::2], direction[1::2]
        left_e, right_e = energy[0::2], energy[1::2]
        parent_e = torch.sqrt(left_e.square() + right_e.square())
        left_x, right_x = left_e * left_u, right_e * right_u
        parent_u = (left_x + right_x) / (math.sqrt(2.0) * parent_e)
        detail = (right_x - left_x) / (math.sqrt(2.0) * parent_e)
        child_energies.append((left_e, right_e))
        ratios.append((left_e / parent_e, right_e / parent_e))
        details.append(detail)
        direction, energy = parent_u, parent_e
        direction_levels.append(direction)
        energy_levels.append(energy)
    return direction_levels, energy_levels, details, child_energies, ratios


def unfold_carrier(root_u, root_e, details, child_energies):
    direction, energy = root_u, root_e
    for detail, (left_e, right_e) in zip(
        reversed(details), reversed(child_energies)
    ):
        left_x = energy * (direction - detail) / math.sqrt(2.0)
        right_x = energy * (direction + detail) / math.sqrt(2.0)
        left_u = left_x / left_e
        right_u = right_x / right_e
        direction = expand_pair(left_u, right_u)
        energy = expand_pair(left_e, right_e)
    return direction * energy


def carrier_path_errors(energy_levels, ratios):
    root_energy = energy_levels[-1]
    reconstructed = root_energy
    for left_ratio, right_ratio in reversed(ratios):
        reconstructed = expand_pair(
            reconstructed * left_ratio,
            reconstructed * right_ratio,
        )
    return float((reconstructed - energy_levels[0]).abs().max().detach())


def patterns(width: int, dim: int, seed: int):
    basis = torch.eye(dim, dtype=torch.float64)
    coherent = basis[0].repeat(width, 1)
    alternating = coherent.clone()
    alternating[1::2] *= -1.0
    imbalanced = coherent.clone()
    imbalanced[0::2] *= 100.0
    imbalanced[1::2] *= 0.01
    one_sided = coherent.clone()
    one_sided[1::2] = 0.0
    generator = torch.Generator().manual_seed(seed)
    random_unit = torch.randn(width, dim, generator=generator, dtype=torch.float64)
    random_unit /= random_unit.norm(dim=-1, keepdim=True)
    return {
        "coherent": coherent,
        "alternating": alternating,
        "imbalanced": imbalanced,
        "one_sided_zero": one_sided,
        "random_unit": random_unit,
        "small_coherent": coherent * 0.01,
        "large_coherent": coherent * 100.0,
    }


def gradient_metrics(root: torch.Tensor, leaf: torch.Tensor):
    probe = torch.arange(1, root.shape[-1] + 1, dtype=root.dtype)
    probe /= probe.norm()
    objective = root[0].dot(probe)
    gradient = torch.autograd.grad(objective, leaf, retain_graph=True)[0]
    return {
        "norm": float(gradient.norm().detach()),
        "max_abs": float(gradient.abs().max().detach()),
        "finite": bool(torch.isfinite(gradient).all()),
    }


def run_local(name: str, raw: torch.Tensor, product: bool):
    leaf = raw.clone().requires_grad_(True)
    levels, details, scales = fold_local(leaf, product)
    restored = unfold_local(levels[-1], details, scales)
    return {
        "arm": name,
        "closure_max_abs": float((restored - leaf).abs().max().detach()),
        "parent_norm_max": max(float(level.norm(dim=-1).max().detach()) for level in levels[1:]),
        "root_norm": float(levels[-1].norm().detach()),
        "root_gradient": gradient_metrics(levels[-1], leaf),
        "scale_by_depth": [
            {
                "min": float(x.min().detach()),
                "mean": float(x.mean().detach()),
                "max": float(x.max().detach()),
            }
            for x in scales
        ],
    }


def run_carrier(raw: torch.Tensor):
    leaf = raw.clone().requires_grad_(True)
    levels, energies, details, child_energies, ratios = fold_carrier(leaf)
    restored = unfold_carrier(
        levels[-1], energies[-1], details, child_energies
    )
    return {
        "arm": "energy_carrier",
        "closure_max_abs": float((restored - leaf).abs().max().detach()),
        "parent_norm_max": max(float(level.norm(dim=-1).max().detach()) for level in levels[1:]),
        "root_norm": float(levels[-1].norm().detach()),
        "root_gradient": gradient_metrics(levels[-1], leaf),
        "energy_by_depth": [
            {
                "min": float(x.min().detach()),
                "mean": float(x.mean().detach()),
                "max": float(x.max().detach()),
            }
            for x in energies
        ],
        "path_product_energy_error": carrier_path_errors(energies, ratios),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16211)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4)
    args = parser.parse_args()
    if args.width < 2 or args.width & (args.width - 1):
        raise ValueError("width must be a power of two")
    if args.dim < 2:
        raise ValueError("dim must be at least two")

    cases = {}
    for pattern, raw in patterns(args.width, args.dim, args.seed).items():
        cases[pattern] = {
            "local_current": run_local("local_current", raw, product=False),
            "energy_carrier": run_carrier(raw),
            "geometric_product": run_local("geometric_product", raw, product=True),
        }

    alternating = cases["alternating"]
    coherent = cases["coherent"]
    imbalanced = cases["imbalanced"]
    one_sided = cases["one_sided_zero"]
    payload = {
        "diagnostic": "STONE-2-RECURSIVE-ENERGY-CARRIER-SMOKE",
        "seed": args.seed,
        "dtype": "float64",
        "width": args.width,
        "dim": args.dim,
        "cases": cases,
        "checks": {
            "local_reproduces_cancellation_explosion": (
                alternating["local_current"]["root_gradient"]["norm"] > 1e6
            ),
            "carrier_cancellation_gradient_below_10": (
                alternating["energy_carrier"]["root_gradient"]["norm"] < 10.0
            ),
            "carrier_cancellation_same_order_as_coherent": (
                alternating["energy_carrier"]["root_gradient"]["norm"]
                / coherent["energy_carrier"]["root_gradient"]["norm"] < 10.0
            ),
            "carrier_closure_below_1e_9": all(
                row["energy_carrier"]["closure_max_abs"] < 1e-9
                for row in cases.values()
            ),
            "carrier_path_product_error_below_1e_10": all(
                row["energy_carrier"]["path_product_energy_error"] < 1e-10
                for row in cases.values()
            ),
            "geometric_product_breaks_boundedness": max(
                imbalanced["geometric_product"]["parent_norm_max"],
                one_sided["geometric_product"]["parent_norm_max"],
            ) > 10.0,
            "all_gradients_finite": all(
                arm["root_gradient"]["finite"]
                for row in cases.values()
                for arm in row.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "checks": payload["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
