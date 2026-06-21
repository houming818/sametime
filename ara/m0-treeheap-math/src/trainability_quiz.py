#!/usr/bin/env python3
"""Minimal ML trainability quiz for the TreeHeap route.

This script intentionally avoids heavy ML frameworks. It uses NumPy and manual
gradients to answer a narrow question: can the local environment and toy modules
learn standard small tasks before we attempt TreeHeap-object echo?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(y)), y] + EPS).mean())


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y).mean())


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def run_linear_regression(rng: np.random.Generator) -> dict:
    n, d, m = 256, 5, 3
    x = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d, m))
    true_b = rng.normal(size=(m,))
    y = x @ true_w + true_b

    w = rng.normal(scale=0.1, size=(d, m))
    b = np.zeros(m, dtype=np.float64)
    lr = 0.05
    losses = []
    for _ in range(800):
        pred = x @ w + b
        err = pred - y
        loss = float((err * err).mean())
        losses.append(loss)
        grad = 2.0 * err / n
        w -= lr * (x.T @ grad)
        b -= lr * grad.sum(axis=0)

    pred = x @ w + b
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return {
        "task": "linear_regression",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], EPS),
        "r2": r2,
        "pass": losses[-1] < 1e-8 and r2 > 0.999999,
    }


def run_xor(rng: np.random.Generator) -> dict:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    hidden = 8
    w1 = rng.normal(scale=0.8, size=(2, hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(scale=0.8, size=(hidden, 2))
    b2 = np.zeros(2, dtype=np.float64)
    lr = 0.2
    losses = []

    y_oh = one_hot(y, 2)
    for _ in range(5000):
        h_pre = x @ w1 + b1
        h = np.tanh(h_pre)
        logits = h @ w2 + b2
        probs = softmax(logits)
        loss = cross_entropy(probs, y)
        losses.append(loss)

        dlogits = (probs - y_oh) / len(y)
        dw2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ w2.T
        dh_pre = dh * (1.0 - h * h)
        dw1 = x.T @ dh_pre
        db1 = dh_pre.sum(axis=0)

        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

    probs = softmax(np.tanh(x @ w1 + b1) @ w2 + b2)
    acc = accuracy(probs, y)
    return {
        "task": "xor",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], EPS),
        "accuracy": acc,
        "pass": acc == 1.0 and losses[-1] < 0.02,
    }


def make_mod_data(base: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for a in range(base):
        for b in range(base):
            x = np.zeros(base * 2, dtype=np.float64)
            x[a] = 1.0
            x[base + b] = 1.0
            rows.append(x)
            labels.append((a + b) % base)
    return np.stack(rows), np.array(labels, dtype=np.int64)


def run_modular_addition(rng: np.random.Generator, base: int) -> dict:
    x, y = make_mod_data(base)
    n, d = x.shape
    hidden = 64
    w1 = rng.normal(scale=0.3, size=(d, hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(scale=0.3, size=(hidden, base))
    b2 = np.zeros(base, dtype=np.float64)
    lr = 0.25
    losses = []
    y_oh = one_hot(y, base)

    for _ in range(8000):
        h_pre = x @ w1 + b1
        h = np.tanh(h_pre)
        logits = h @ w2 + b2
        probs = softmax(logits)
        loss = cross_entropy(probs, y)
        losses.append(loss)

        dlogits = (probs - y_oh) / n
        dw2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ w2.T
        dh_pre = dh * (1.0 - h * h)
        dw1 = x.T @ dh_pre
        db1 = dh_pre.sum(axis=0)

        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

    probs = softmax(np.tanh(x @ w1 + b1) @ w2 + b2)
    acc = accuracy(probs, y)
    confusion = np.zeros((base, base), dtype=int)
    pred = probs.argmax(axis=1)
    for gold, got in zip(y, pred):
        confusion[gold, got] += 1
    return {
        "task": "modular_addition",
        "base": base,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], EPS),
        "accuracy": acc,
        "confusion": confusion.tolist(),
        "pass": acc == 1.0 and losses[-1] < 0.05,
    }


def render_readme(summary: dict) -> str:
    lines = [
        "# Trainability Quiz Evidence",
        "",
        "This is synthetic M0 evidence for a minimal learning check before TreeHeap-object echo.",
        "",
        "The quiz uses NumPy manual gradients, not PyTorch.",
        "",
        "## Verdict",
        "",
        f"`pilot_pass = {summary['pilot_pass']}`",
        "",
        "## Tasks",
        "",
        "| Task | Final Loss | Accuracy/R2 | Pass |",
        "|---|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        score = task.get("accuracy", task.get("r2"))
        lines.append(f"| `{task['task']}` | `{task['final_loss']}` | `{score}` | `{task['pass']}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The toy verifies that small trainable modules can learn linear mapping,",
            "nonlinear XOR, and a full base-8 modular addition table. This does not",
            "prove TreeHeap language learning; it is only the ML entrance exam before",
            "building trainable TreeHeap encoder/plus/decoder modules.",
            "",
        ]
    )
    return "\n".join(lines)


def run(seed: int, base: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    tasks = [
        run_linear_regression(rng),
        run_xor(rng),
        run_modular_addition(rng, base=base),
    ]
    summary = {
        "run": {
            "seed": seed,
            "base": base,
            "framework": "numpy_manual_gradients",
        },
        "tasks": tasks,
        "pilot_pass": all(task["pass"] for task in tasks),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "trainability_quiz",
    )
    args = parser.parse_args()
    summary = run(seed=args.seed, base=args.base, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
