#!/usr/bin/env python3
"""Soft Plus probe for differentiable TreeHeap operators.

This is an M0 toy proof. It does not test language.

Question:
  Can a kernel-guided Soft Plus operator carry gradients through:

      K_write(subheap, x) -> p(address) -> sum p(a) * Plus_a(H, x) -> loss

  and collapse back to the correct hard plus address at low temperature?

The probe uses NumPy and manual gradients so the proof is transparent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPS = 1e-12


ADDRESSES = ("LL", "LR", "RL", "RR")
ADDR_META = {
    "LL": {"parent": 4.0, "root_side": -1.0, "child_side": -1.0, "depth": 2.0},
    "LR": {"parent": 4.0, "root_side": -1.0, "child_side": 1.0, "depth": 2.0},
    "RL": {"parent": 12.0, "root_side": 1.0, "child_side": -1.0, "depth": 2.0},
    "RR": {"parent": 12.0, "root_side": 1.0, "child_side": 1.0, "depth": 2.0},
}


def softmax(logits: np.ndarray, tau: float) -> np.ndarray:
    z = logits / tau
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def gold_address(key: float) -> int:
    if key < 4:
        return 0  # LL
    if key < 8:
        return 1  # LR
    if key < 12:
        return 2  # RL
    return 3  # RR


def kernel_features(keys: np.ndarray) -> np.ndarray:
    """Features seen by K_write(subheap(H, a), x).

    Shape: [num_samples, num_addresses, feature_dim].
    """

    rows = []
    for key in keys:
        key_norm = key / 16.0
        per_addr = []
        for name in ADDRESSES:
            meta = ADDR_META[name]
            parent = meta["parent"]
            parent_norm = parent / 16.0
            diff = (key - parent) / 16.0
            root_diff = (key - 8.0) / 16.0
            root_alignment = root_diff * meta["root_side"]
            child_alignment = diff * meta["child_side"]
            # These are differentiable numeric features in the kernel. They are
            # not hard route decisions; the kernel must learn how to use them.
            per_addr.append(
                [
                    1.0,
                    key_norm,
                    parent_norm,
                    diff,
                    root_diff,
                    meta["root_side"],
                    meta["child_side"],
                    meta["depth"] / 4.0,
                    abs(diff),
                    key_norm * meta["child_side"],
                    root_alignment,
                    child_alignment,
                    root_alignment + child_alignment,
                ]
            )
        rows.append(per_addr)
    return np.asarray(rows, dtype=np.float64)


def plus_features(keys: np.ndarray) -> np.ndarray:
    """Features consumed by Plus_a(H, x).

    Shape: [num_samples, num_addresses, plus_feature_dim].
    """

    rows = []
    for key in keys:
        key_norm = key / 16.0
        per_addr = []
        for idx, name in enumerate(ADDRESSES):
            meta = ADDR_META[name]
            addr_one_hot = [1.0 if idx == j else 0.0 for j in range(len(ADDRESSES))]
            per_addr.append(
                [
                    1.0,
                    key_norm,
                    meta["parent"] / 16.0,
                    meta["root_side"],
                    meta["child_side"],
                    meta["depth"] / 4.0,
                    *addr_one_hot,
                ]
            )
        rows.append(per_addr)
    return np.asarray(rows, dtype=np.float64)


def forward(
    keys: np.ndarray,
    theta: np.ndarray,
    plus_u: np.ndarray,
    true_u: np.ndarray,
    tau: float,
    route_ce_weight: float,
) -> dict:
    phi = kernel_features(keys)
    psi = plus_features(keys)
    gold = np.asarray([gold_address(float(k)) for k in keys], dtype=np.int64)

    scores = phi @ theta
    probs = softmax(scores, tau=tau)
    candidates = np.einsum("naf,df->nad", psi, plus_u)
    target = np.asarray([true_u @ psi[i, gold[i]] for i in range(len(keys))])
    pred = np.einsum("na,nad->nd", probs, candidates)

    err = pred - target
    mse_loss = float(0.5 * np.mean(err * err))
    route_loss = float(-np.log(probs[np.arange(len(keys)), gold] + EPS).mean())
    loss = mse_loss + route_ce_weight * route_loss
    acc = float((probs.argmax(axis=1) == gold).mean())
    mean_gold_prob = float(probs[np.arange(len(keys)), gold].mean())
    hard_soft_gap = float(np.mean(np.linalg.norm(pred - target, axis=1)))

    return {
        "phi": phi,
        "psi": psi,
        "gold": gold,
        "scores": scores,
        "probs": probs,
        "candidates": candidates,
        "target": target,
        "pred": pred,
        "err": err,
        "loss": loss,
        "mse_loss": mse_loss,
        "route_loss": route_loss,
        "accuracy": acc,
        "mean_gold_prob": mean_gold_prob,
        "hard_soft_gap": hard_soft_gap,
    }


def backward(
    cache: dict,
    theta: np.ndarray,
    plus_u: np.ndarray,
    tau: float,
    route_ce_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi = cache["phi"]
    psi = cache["psi"]
    probs = cache["probs"]
    candidates = cache["candidates"]
    err = cache["err"]

    n, _, out_dim = candidates.shape
    grad_pred = err / (n * out_dim)

    # pred = sum_a probs[a] * candidates[a]
    grad_candidates = probs[:, :, None] * grad_pred[:, None, :]
    grad_plus_u = np.einsum("nad,naf->df", grad_candidates, psi)

    grad_probs = np.einsum("nd,nad->na", grad_pred, candidates)
    dot = np.sum(grad_probs * probs, axis=1, keepdims=True)
    grad_scores = probs * (grad_probs - dot) / tau
    gold = cache["gold"]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(gold)), gold] = 1.0
    grad_scores += route_ce_weight * (probs - one_hot) / (len(gold) * tau)
    grad_theta = np.einsum("naf,na->f", phi, grad_scores)

    return grad_theta, grad_plus_u


def run_probe(
    seed: int,
    epochs: int,
    lr_theta: float,
    lr_plus: float,
    out_dim: int,
    route_ce_weight: float,
) -> dict:
    rng = np.random.default_rng(seed)
    keys = np.asarray([2, 3, 5, 6, 10, 11, 13, 14], dtype=np.float64)
    feature_dim = kernel_features(keys).shape[-1]
    plus_dim = plus_features(keys).shape[-1]

    theta = rng.normal(scale=0.1, size=(feature_dim,))
    true_u = rng.normal(scale=0.8, size=(out_dim, plus_dim))
    plus_u = true_u + rng.normal(scale=0.35, size=(out_dim, plus_dim))

    trace = []
    first_grad = None
    for epoch in range(epochs + 1):
        tau = max(0.25, 2.0 * (0.997**epoch))
        cache = forward(keys, theta, plus_u, true_u, tau=tau, route_ce_weight=route_ce_weight)
        grad_theta, grad_plus_u = backward(
            cache,
            theta,
            plus_u,
            tau=tau,
            route_ce_weight=route_ce_weight,
        )
        grad_theta_norm = float(np.linalg.norm(grad_theta))
        grad_plus_norm = float(np.linalg.norm(grad_plus_u))
        if first_grad is None:
            first_grad = {
                "theta": grad_theta_norm,
                "plus_u": grad_plus_norm,
            }
        if epoch in {0, 1, 5, 25, 100, 250, 500, 1000, epochs}:
            trace.append(
                {
                    "epoch": epoch,
                    "tau": tau,
                    "loss": cache["loss"],
                    "mse_loss": cache["mse_loss"],
                    "route_loss": cache["route_loss"],
                    "accuracy": cache["accuracy"],
                    "mean_gold_prob": cache["mean_gold_prob"],
                    "hard_soft_gap": cache["hard_soft_gap"],
                    "grad_theta_norm": grad_theta_norm,
                    "grad_plus_norm": grad_plus_norm,
                }
            )
        if epoch == epochs:
            break
        theta -= lr_theta * grad_theta
        plus_u -= lr_plus * grad_plus_u

    final_soft = forward(keys, theta, plus_u, true_u, tau=0.25, route_ce_weight=route_ce_weight)
    final_collapse = forward(keys, theta, plus_u, true_u, tau=0.05, route_ce_weight=route_ce_weight)
    initial = trace[0]

    pilot_pass = (
        first_grad["theta"] > 1e-10
        and first_grad["plus_u"] > 1e-10
        and final_soft["loss"] < initial["loss"]
        and final_soft["mean_gold_prob"] > initial["mean_gold_prob"]
        and final_collapse["accuracy"] == 1.0
    )

    examples = []
    for i, key in enumerate(keys):
        examples.append(
            {
                "key": int(key),
                "gold": ADDRESSES[int(final_soft["gold"][i])],
                "argmax_tau_0_25": ADDRESSES[int(final_soft["probs"][i].argmax())],
                "argmax_tau_0_05": ADDRESSES[int(final_collapse["probs"][i].argmax())],
                "gold_prob_tau_0_25": float(final_soft["probs"][i, final_soft["gold"][i]]),
                "gold_prob_tau_0_05": float(final_collapse["probs"][i, final_collapse["gold"][i]]),
            }
        )

    return {
        "seed": seed,
        "epochs": epochs,
        "lr_theta": lr_theta,
        "lr_plus": lr_plus,
        "out_dim": out_dim,
        "route_ce_weight": route_ce_weight,
        "addresses": list(ADDRESSES),
        "claims_tested": [
            "P-SOFT01 hard operator probabilistic lifting",
            "P-SOFT02 soft plus is differentiable TreeHeap plus",
            "P-SOFT03 kernel-guided soft plus can learn write/merge routes",
            "P-SOFT04 low-temperature collapse recovers hard plus address",
        ],
        "initial": initial,
        "final_tau_0_25": {
            "loss": final_soft["loss"],
            "mse_loss": final_soft["mse_loss"],
            "route_loss": final_soft["route_loss"],
            "accuracy": final_soft["accuracy"],
            "mean_gold_prob": final_soft["mean_gold_prob"],
            "hard_soft_gap": final_soft["hard_soft_gap"],
        },
        "final_tau_0_05": {
            "loss": final_collapse["loss"],
            "mse_loss": final_collapse["mse_loss"],
            "route_loss": final_collapse["route_loss"],
            "accuracy": final_collapse["accuracy"],
            "mean_gold_prob": final_collapse["mean_gold_prob"],
            "hard_soft_gap": final_collapse["hard_soft_gap"],
        },
        "first_grad_norms": first_grad,
        "trace": trace,
        "examples": examples,
        "pilot_pass": pilot_pass,
        "interpretation": (
            "This proves only a synthetic M0 gradient/collapse path. It does not "
            "prove language learning, WMT performance, or superiority to Transformer."
        ),
    }


def write_readme(summary: dict, out_dir: Path) -> None:
    rows = "\n".join(
        f"| {r['epoch']} | {r['tau']:.4f} | {r['loss']:.6g} | {r['mse_loss']:.6g} | "
        f"{r['route_loss']:.6g} | {r['accuracy']:.3f} | "
        f"{r['mean_gold_prob']:.3f} | {r['hard_soft_gap']:.6g} | "
        f"{r['grad_theta_norm']:.3e} | {r['grad_plus_norm']:.3e} |"
        for r in summary["trace"]
    )
    examples = "\n".join(
        f"| {e['key']} | {e['gold']} | {e['argmax_tau_0_25']} | {e['argmax_tau_0_05']} | "
        f"{e['gold_prob_tau_0_25']:.3f} | {e['gold_prob_tau_0_05']:.3f} |"
        for e in summary["examples"]
    )
    text = f"""# Soft Plus Probe Evidence

This is an M0 synthetic proof for kernel-guided Soft Plus.

## Verdict

`pilot_pass = {summary['pilot_pass']}`

## What Was Tested

```text
score(a) = K_write(subheap(H, a), x)
p(a) = softmax(score(a))
H_next = sum_a p(a) * Plus_a(H, x)
loss = MSE(H_next, target)
```

The proof checks whether gradients reach both:

```text
K_write parameters
Plus_a parameters
```

and whether low-temperature collapse recovers the correct hard plus address.

## First Gradient Norms

```text
dL/dK_write = {summary['first_grad_norms']['theta']}
dL/dPlus    = {summary['first_grad_norms']['plus_u']}
```

## Training Trace

| Epoch | Tau | Loss | MSE | Route CE | Accuracy | Gold Prob | Hard/Soft Gap | Grad K | Grad Plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Collapse Examples

| Key | Gold Address | Argmax tau=0.25 | Argmax tau=0.05 | Gold Prob tau=0.25 | Gold Prob tau=0.05 |
|---:|---|---|---|---:|---:|
{examples}

## Interpretation

This evidence supports only a narrow M0 claim:

```text
kernel-guided Soft Plus can be made differentiable, can receive gradient
through K_write and Plus_a, and can collapse to the hard address in this
synthetic key/address toy.
```

It does not prove language understanding, syntax induction, WMT translation,
or superiority over Transformer.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "evidence" / "soft_plus_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr-theta", type=float, default=1.0)
    parser.add_argument("--lr-plus", type=float, default=0.4)
    parser.add_argument("--out-dim", type=int, default=8)
    parser.add_argument("--route-ce-weight", type=float, default=0.25)
    args = parser.parse_args()

    summary = run_probe(
        seed=args.seed,
        epochs=args.epochs,
        lr_theta=args.lr_theta,
        lr_plus=args.lr_plus,
        out_dim=args.out_dim,
        route_ce_weight=args.route_ce_weight,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["trace"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_readme(summary, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
