#!/usr/bin/env python3
"""TreeHeap diff algebra probe.

This proof treats distance as a derived operation:

  diff -> norm / inner product / cosine -> finite difference -> learning signal

The goal is to define the minimum numerical toolbox needed before S1 can train
a probabilistic vector-plus encoder that writes token/world-model vectors into a
zero TreeHeap.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def depths_for_heap(n_nodes: int) -> np.ndarray:
    depths = np.zeros(n_nodes, dtype=np.int64)
    for i in range(1, n_nodes):
        depths[i] = depths[(i - 1) // 2] + 1
    return depths


def depth_weights(n_nodes: int, alpha: float) -> np.ndarray:
    return alpha ** depths_for_heap(n_nodes)


def treeheap_zero(n_nodes: int, dim: int) -> np.ndarray:
    return np.zeros((n_nodes, dim), dtype=np.float64)


def treeheap_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def treeheap_inner(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[:, None] * a * b))


def treeheap_norm(a: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(max(treeheap_inner(a, a, weights), 0.0))


def treeheap_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return treeheap_norm(treeheap_diff(a, b), weights)


def treeheap_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    denom = treeheap_norm(a, weights) * treeheap_norm(b, weights)
    if denom < eps:
        return 0.0
    return treeheap_inner(a, b, weights) / denom


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def prob_vector_plus(h: np.ndarray, x: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A minimal probabilistic vector write.

    theta controls the route distribution over heap nodes.  The vector update is
    intentionally simple:

      H'[i] = H[i] + p_i * x

    This is enough to test whether TreeHeap diff algebra supports a learning
    signal.  It is not a claim that this is the final S1 encoder.
    """

    p = softmax(theta)
    h2 = h + p[:, None] * x[None, :]
    return h2, p


def weighted_mse_loss(h: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    d = h - target
    return 0.5 * float(np.sum(weights[:, None] * d * d))


def analytic_grad_theta(h: np.ndarray, x: np.ndarray, target: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    h2, p = prob_vector_plus(h, x, theta)
    # dL/dp_i = <dL/dH'_i, dH'_i/dp_i> = weights_i * (H'_i-target_i) dot x
    g_p = weights * np.sum((h2 - target) * x[None, :], axis=1)
    # softmax Jacobian-vector product: dL/dtheta_i = p_i * (g_p_i - sum_j p_j g_p_j)
    return p * (g_p - float(np.sum(p * g_p)))


def finite_diff_grad_theta(
    h: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    theta: np.ndarray,
    weights: np.ndarray,
    eps: float,
) -> np.ndarray:
    grad = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        step = np.zeros_like(theta)
        step[i] = eps
        hp, _ = prob_vector_plus(h, x, theta + step)
        hm, _ = prob_vector_plus(h, x, theta - step)
        grad[i] = (weighted_mse_loss(hp, target, weights) - weighted_mse_loss(hm, target, weights)) / (2.0 * eps)
    return grad


def directional_finite_difference(
    h: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    """Check finite difference in TreeHeap state space."""

    lp = weighted_mse_loss(h + eps * direction, target, weights)
    lm = weighted_mse_loss(h - eps * direction, target, weights)
    numerical = (lp - lm) / (2.0 * eps)
    analytic = treeheap_inner(h - target, direction, weights)
    return {
        "finite_difference": float(numerical),
        "analytic_directional_derivative": float(analytic),
        "abs_error": float(abs(numerical - analytic)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n, d = args.nodes, args.dim
    weights = depth_weights(n, args.alpha)
    zero = treeheap_zero(n, d)
    a = rng.normal(0.0, 0.2, size=(n, d))
    b = rng.normal(0.0, 0.2, size=(n, d))
    direction = rng.normal(0.0, 0.2, size=(n, d))
    x = rng.normal(0.0, 1.0, size=(d,))

    # Make a target that has a known best route at node 3.
    target_node = 3
    target = np.zeros_like(a)
    target[target_node] = x
    theta = rng.normal(0.0, 0.1, size=(n,))

    norm_zero = treeheap_norm(zero, weights)
    norm_a = treeheap_norm(a, weights)
    dist_aa = treeheap_distance(a, a, weights)
    dist_ab = treeheap_distance(a, b, weights)
    dist_ba = treeheap_distance(b, a, weights)
    triangle_gap = treeheap_distance(a, b, weights) + treeheap_distance(b, zero, weights) - treeheap_distance(a, zero, weights)
    cos_aa = treeheap_cosine(a, a, weights)
    cos_ab = treeheap_cosine(a, b, weights)
    diff_closure_shape_ok = treeheap_diff(a, b).shape == a.shape
    anti_sym_error = float(np.max(np.abs(treeheap_diff(a, b) + treeheap_diff(b, a))))

    h_after, p = prob_vector_plus(zero, x, theta)
    initial_loss = weighted_mse_loss(h_after, target, weights)
    grad_an = analytic_grad_theta(zero, x, target, theta, weights)
    grad_fd = finite_diff_grad_theta(zero, x, target, theta, weights, args.eps)
    grad_abs_error = float(np.max(np.abs(grad_an - grad_fd)))
    grad_rel_error = float(np.linalg.norm(grad_an - grad_fd) / (np.linalg.norm(grad_fd) + 1e-12))

    # One gradient step should reduce the weighted distance to the target.
    lr = 2.0
    h_step, p_step = prob_vector_plus(zero, x, theta - lr * grad_an)
    stepped_loss = weighted_mse_loss(h_step, target, weights)
    target_prob_before = float(p[target_node])
    target_prob_after = float(p_step[target_node])

    dir_check = directional_finite_difference(a, direction, b, weights, args.eps)

    pass_checks = {
        "zero_norm_is_zero": norm_zero == 0.0,
        "self_distance_is_zero": dist_aa < 1e-12,
        "distance_symmetric": abs(dist_ab - dist_ba) < 1e-12,
        "triangle_nonnegative_gap": triangle_gap >= -1e-10,
        "cosine_self_is_one": abs(cos_aa - 1.0) < 1e-12,
        "diff_closure_shape_ok": bool(diff_closure_shape_ok),
        "diff_antisymmetry": anti_sym_error < 1e-12,
        "state_directional_derivative_matches_finite_diff": dir_check["abs_error"] < 1e-8,
        "theta_gradient_matches_finite_diff": grad_abs_error < 1e-8 and grad_rel_error < 1e-6,
        "gradient_step_reduces_loss": stepped_loss < initial_loss,
        "gradient_step_increases_target_write_prob": target_prob_after > target_prob_before,
    }

    summary = {
        "claim": "M0-DIFF-C01",
        "predict": "P-DIFF01",
        "seed": args.seed,
        "host_preference": "io.grepcode.cn",
        "config": {
            "nodes": n,
            "dim": d,
            "alpha": args.alpha,
            "eps": args.eps,
            "target_node": target_node,
        },
        "metrics": {
            "norm_zero": norm_zero,
            "norm_a": norm_a,
            "dist_aa": dist_aa,
            "dist_ab": dist_ab,
            "dist_ba": dist_ba,
            "triangle_gap": triangle_gap,
            "cos_aa": cos_aa,
            "cos_ab": cos_ab,
            "anti_sym_error": anti_sym_error,
            "directional_derivative_abs_error": dir_check["abs_error"],
            "theta_grad_abs_error": grad_abs_error,
            "theta_grad_rel_error": grad_rel_error,
            "initial_loss": initial_loss,
            "stepped_loss": stepped_loss,
            "target_prob_before": target_prob_before,
            "target_prob_after": target_prob_after,
        },
        "pass_checks": pass_checks,
        "pilot_pass": bool(all(pass_checks.values())),
        "interpretation": {
            "supported": "TreeHeap distance can be derived from diff->norm/inner product, and finite differences provide a learning signal for prob vector plus.",
            "not_proved": [
                "not a final S1 encoder",
                "not semantic world-model alignment",
                "not WMT",
                "not superiority over Transformer/MLP",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "directional_check", **dir_check}, ensure_ascii=False),
                json.dumps({"event": "theta_grad_an", "values": grad_an.tolist()}, ensure_ascii=False),
                json.dumps({"event": "theta_grad_fd", "values": grad_fd.tolist()}, ensure_ascii=False),
                json.dumps({"event": "write_prob_before", "values": p.tolist()}, ensure_ascii=False),
                json.dumps({"event": "write_prob_after", "values": p_step.tolist()}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# TreeHeap diff algebra probe",
                "",
                "This evidence validates the minimal diff algebra needed for learning:",
                "",
                "- Zero TreeHeap",
                "- subtraction / anti-symmetry",
                "- depth-weighted norm and distance",
                "- depth-weighted inner product and cosine",
                "- finite difference in TreeHeap state space",
                "- finite difference against prob vector plus route parameters",
                "",
                "This is M0 math evidence, not a language or WMT proof.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
