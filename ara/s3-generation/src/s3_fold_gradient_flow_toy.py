#!/usr/bin/env python3
"""Controlled calculus audit of the current reference-normalized TreeHeap FOLD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


EPSILON = 1e-8


def pair_fold(left: torch.Tensor, right: torch.Tensor):
    scale = torch.sqrt(left.square().sum(-1, keepdim=True) + right.square().sum(-1, keepdim=True) + EPSILON)
    parent = (left + right) / (math.sqrt(2.0) * scale)
    detail = (right - left) / (math.sqrt(2.0) * scale)
    return parent, detail, scale


def fold_tree(leaf: torch.Tensor):
    levels = [leaf]
    details, scales = [], []
    node = leaf
    while node.shape[0] > 1:
        left, right = node[0::2], node[1::2]
        node, detail, scale = pair_fold(left, right)
        levels.append(node)
        details.append(detail)
        scales.append(scale)
    return levels, details, scales


def unfold_tree(root: torch.Tensor, details, scales):
    level = root
    levels = [level]
    for detail, scale in zip(reversed(details), reversed(scales)):
        left = scale * (level - detail) / math.sqrt(2.0)
        right = scale * (level + detail) / math.sqrt(2.0)
        expanded = torch.empty(left.shape[0] * 2, left.shape[1], dtype=left.dtype, device=left.device)
        expanded[0::2], expanded[1::2] = left, right
        level = expanded
        levels.append(level)
    return levels


def pair_jacobian_case(scale_value: float, relation: str, dim: int):
    e0 = torch.zeros(dim, dtype=torch.float64)
    e0[0] = 1.0
    e1 = torch.zeros(dim, dtype=torch.float64)
    e1[1] = 1.0
    left = scale_value * e0
    if relation == "aligned":
        right = scale_value * e0
    elif relation == "opposed":
        right = -scale_value * e0
    elif relation == "orthogonal":
        right = scale_value * e1
    else:
        raise ValueError(relation)
    vector = torch.cat((left, right)).requires_grad_(True)

    def parent_fn(value):
        parent, _, _ = pair_fold(value[:dim], value[dim:])
        return parent

    def full_fn(value):
        parent, detail, scale = pair_fold(value[:dim], value[dim:])
        return torch.cat((parent, detail, scale.reshape(1)))

    parent = parent_fn(vector)
    parent_jacobian = torch.autograd.functional.jacobian(parent_fn, vector)
    full_jacobian = torch.autograd.functional.jacobian(full_fn, vector)
    parent_sv = torch.linalg.svdvals(parent_jacobian)
    full_sv = torch.linalg.svdvals(full_jacobian)
    radial = parent_jacobian @ vector.detach()
    return {
        "scale": scale_value,
        "relation": relation,
        "parent_norm": float(parent.norm()),
        "parent_jacobian_singular_values": [float(value) for value in parent_sv],
        "parent_jacobian_operator_norm": float(parent_sv.max()),
        "parent_radial_derivative_norm": float(radial.norm()),
        "full_jacobian_singular_values": [float(value) for value in full_sv],
        "full_jacobian_condition": float(full_sv.max() / full_sv.min().clamp_min(1e-30)),
    }


def make_patterns(width: int, dim: int, seed: int):
    e = torch.eye(dim, dtype=torch.float64)
    coherent = e[0].repeat(width, 1)
    alternating = coherent.clone()
    alternating[1::2] *= -1.0
    orthogonal = torch.stack([e[index % dim] for index in range(width)])
    generator = torch.Generator().manual_seed(seed)
    random_unit = torch.randn(width, dim, generator=generator, dtype=torch.float64)
    random_unit = random_unit / random_unit.norm(dim=-1, keepdim=True)
    return {
        "coherent": coherent,
        "alternating": alternating,
        "orthogonal_cycle": orthogonal,
        "random_unit": random_unit,
        "small_coherent": coherent * 0.01,
        "large_coherent": coherent * 100.0,
    }


def recursive_case(name: str, raw_leaf: torch.Tensor):
    leaf = raw_leaf.clone().requires_grad_(True)
    levels, details, scales = fold_tree(leaf)
    restored = unfold_tree(levels[-1], details, scales)[-1]
    closure = float((restored - leaf).abs().max())
    probe = torch.arange(1, leaf.shape[1] + 1, dtype=leaf.dtype)
    probe = probe / probe.norm()
    gradients = []
    for depth, level in enumerate(levels):
        scalar = level[0].dot(probe)
        gradient = torch.autograd.grad(scalar, leaf, retain_graph=True)[0]
        gradients.append({
            "depth_from_leaf": depth,
            "width": level.shape[0],
            "state_norm": float(level.norm()),
            "leaf_gradient_norm": float(gradient.norm()),
            "active_leaf_gradient_norm": float(gradient[: 2 ** depth].norm()),
        })
    return {
        "pattern": name,
        "closure_max_abs": closure,
        "parent_norm_max": max(float(level.norm(dim=-1).max()) for level in levels[1:]),
        "scale_by_depth": [
            {
                "min": float(scale.min()),
                "mean": float(scale.mean()),
                "max": float(scale.max()),
            }
            for scale in scales
        ],
        "gradients": gradients,
    }, list(reversed(levels))


class SharedReadKernel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.first = nn.Linear(3 * dim, 2 * dim)
        self.second = nn.Linear(2 * dim, dim)

    def forward(self, query, local, depth):
        return torch.tanh(self.second(F.gelu(self.first(torch.cat((query, local, depth), dim=-1)))))


def flatten_grad(parameters):
    return torch.cat([
        parameter.grad.detach().reshape(-1) if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1)
        for parameter in parameters
    ])


def cosine_matrix(vectors):
    stacked = torch.stack(vectors)
    normalized = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return normalized @ normalized.T


def read_case(root_to_leaf, dim: int, seed: int):
    torch.manual_seed(seed)
    kernel = SharedReadKernel(dim).double()
    depth_embedding = nn.Embedding(len(root_to_leaf), dim).double()
    query0 = torch.linspace(-0.25, 0.25, dim, dtype=torch.float64)[None]
    gain = float(torch.sigmoid(torch.tensor(-4.0)))
    locals_ = [level.mean(0, keepdim=True).detach() for level in root_to_leaf]
    corrections = []
    query = query0
    for depth, local in enumerate(locals_):
        depth_state = depth_embedding.weight[depth][None]
        query = query + gain * kernel(query, local, depth_state)
        corrections.append(float((query - query0).norm()))

    parameters = list(kernel.parameters())
    direct_gradients = []
    for depth, local in enumerate(locals_):
        kernel.zero_grad(set_to_none=True)
        depth_embedding.zero_grad(set_to_none=True)
        update = kernel(query0, local, depth_embedding.weight[depth][None])
        objective = 0.5 * (query0 + gain * update).square().sum()
        objective.backward()
        direct_gradients.append(flatten_grad(parameters))
    gram = cosine_matrix(direct_gradients)
    off_diagonal = gram[~torch.eye(len(locals_), dtype=torch.bool)]
    return {
        "gain": gain,
        "depths": len(locals_),
        "correction_norm_after_each_depth": corrections,
        "direct_parameter_gradient_norms": [float(value.norm()) for value in direct_gradients],
        "gradient_cosine_matrix": [[float(value) for value in row] for row in gram],
        "off_diagonal_cosine_min": float(off_diagonal.min()),
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "off_diagonal_negative_fraction": float((off_diagonal < 0).double().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16201)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4)
    args = parser.parse_args()
    if args.width < 2 or args.width & (args.width - 1):
        raise ValueError("width must be a power of two")
    if args.dim < 2:
        raise ValueError("dim must be at least two")

    jacobians = [
        pair_jacobian_case(scale, relation, args.dim)
        for relation in ("aligned", "opposed", "orthogonal")
        for scale in (0.01, 1.0, 100.0)
    ]
    recursive = []
    read = {}
    for name, leaf in make_patterns(args.width, args.dim, args.seed).items():
        result, root_to_leaf = recursive_case(name, leaf)
        recursive.append(result)
        read[name] = read_case(root_to_leaf, args.dim, args.seed + 100)

    aligned = {row["scale"]: row for row in jacobians if row["relation"] == "aligned"}
    payload = {
        "diagnostic": "STONE-2-FOLD-GRADIENT-TOY",
        "seed": args.seed,
        "dtype": "float64",
        "width": args.width,
        "dim": args.dim,
        "jacobian_cases": jacobians,
        "recursive_cases": recursive,
        "read_cases": read,
        "checks": {
            "all_parents_bounded": all(row["parent_norm_max"] <= 1.0 + 1e-12 for row in recursive),
            "all_closure_max_abs_below_1e_9": all(row["closure_max_abs"] < 1e-9 for row in recursive),
            "aligned_parent_scale_invariant": max(
                abs(aligned[scale]["parent_norm"] - aligned[1.0]["parent_norm"])
                for scale in aligned
            ) < 1e-4,
            "aligned_jacobian_inverse_scale_approximately": abs(
                aligned[0.01]["parent_jacobian_operator_norm"]
                / aligned[1.0]["parent_jacobian_operator_norm"] - 100.0
            ) < 0.1,
            "read_has_negative_cross_depth_gradient_for_some_pattern": any(
                row["off_diagonal_negative_fraction"] > 0.0 for row in read.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "checks": payload["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
