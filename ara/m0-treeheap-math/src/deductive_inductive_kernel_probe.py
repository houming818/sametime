#!/usr/bin/env python3
"""Deductive vs inductive TreeHeap kernel proof.

Part A is deductive: operations such as mirror/conjugate/one-hot soft plus are
checked as algebraic identities.

Part B is inductive: a probabilistic kernel is trained to imitate a synthetic
world-model operation distribution. KL divergence measures whether the learned
distribution approaches the world distribution on train/test/OOD splits.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def all_paths(depth: int) -> list[str]:
    paths = [""]
    frontier = [""]
    for _ in range(depth):
        nxt: list[str] = []
        for p in frontier:
            nxt.extend([p + "L", p + "R"])
        paths.extend(nxt)
        frontier = nxt
    return paths


def candidate_paths(depth: int) -> list[str]:
    return all_paths(depth - 1)


def mirror_path(path: str) -> str:
    return "".join("R" if ch == "L" else "L" for ch in path)


def path_to_id(path: str) -> int:
    idx = 0
    for ch in path:
        idx = 2 * idx + (1 if ch == "L" else 2)
    return idx


def mirror_patch(patch: np.ndarray) -> np.ndarray:
    return patch[[0, 2, 1]]


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=-1, keepdims=True)


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(p * (np.log(p + EPS) - np.log(q + EPS)), axis=-1)


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(p + EPS), axis=-1)


def path_features(path: str, max_depth: int) -> np.ndarray:
    bits = np.zeros(max_depth, dtype=np.float64)
    for i, ch in enumerate(path[:max_depth]):
        bits[i] = -1.0 if ch == "L" else 1.0
    depth = len(path) / max(1, max_depth)
    left_ratio = path.count("L") / max(1, len(path))
    right_ratio = path.count("R") / max(1, len(path))
    return np.concatenate([[depth, left_ratio, right_ratio], bits])


@dataclass
class Dataset:
    query: np.ndarray
    patches: np.ndarray
    path_feat: np.ndarray
    address_ids: np.ndarray
    p_world: np.ndarray
    paths: list[list[str]]


def make_dataset(
    rng: np.random.Generator,
    *,
    n: int,
    depth: int,
    feature_depth: int,
    temperature: float,
    query_scale: float,
    patch_scale: float,
) -> Dataset:
    paths = candidate_paths(depth)
    queries = []
    patches = []
    path_feats = []
    address_ids = []
    p_worlds = []
    path_rows = []

    for _ in range(n):
        q = rng.normal(0.0, query_scale, size=3)
        row_patches = rng.normal(0.0, patch_scale, size=(len(paths), 3))

        # Ensure the distribution is not always uniform by making one or two
        # candidates partially match the query.
        hit = int(rng.integers(0, len(paths)))
        row_patches[hit] = q + rng.normal(0.0, 0.18 * patch_scale, size=3)
        if rng.random() < 0.35:
            near = int(rng.integers(0, len(paths)))
            row_patches[near] = q + rng.normal(0.0, 0.45 * patch_scale, size=3)

        dist = np.sum((row_patches - q[None, :]) ** 2, axis=1)
        p_world = stable_softmax((-dist / temperature)[None, :])[0]

        queries.append(q)
        patches.append(row_patches)
        path_feats.append(np.vstack([path_features(p, feature_depth) for p in paths]))
        address_ids.append(np.array([path_to_id(p) for p in paths], dtype=np.int64))
        p_worlds.append(p_world)
        path_rows.append(list(paths))

    return Dataset(
        query=np.asarray(queries, dtype=np.float64),
        patches=np.asarray(patches, dtype=np.float64),
        path_feat=np.asarray(path_feats, dtype=np.float64),
        address_ids=np.asarray(address_ids, dtype=np.int64),
        p_world=np.asarray(p_worlds, dtype=np.float64),
        paths=path_rows,
    )


def raw_features(ds: Dataset, include_path: bool) -> np.ndarray:
    n, m, _ = ds.patches.shape
    q = np.repeat(ds.query[:, None, :], m, axis=1)
    feats = [q, ds.patches]
    if include_path:
        feats.append(ds.path_feat)
    return np.concatenate(feats, axis=-1)


class AddressPrior:
    def __init__(self, max_address_id: int):
        self.w = np.zeros(max_address_id + 1, dtype=np.float64)

    def scores(self, ds: Dataset) -> np.ndarray:
        ids = np.clip(ds.address_ids, 0, len(self.w) - 1)
        return self.w[ids]

    def train(self, train: Dataset, *, epochs: int, lr: float) -> list[dict]:
        trace = []
        for epoch in range(epochs + 1):
            scores = self.scores(train)
            q = stable_softmax(scores)
            loss = float(kl_div(train.p_world, q).mean())
            if epoch in {0, 1, 5, 25, 100, 250, 500, epochs}:
                trace.append({"epoch": epoch, "train_kl": loss})
            if epoch == epochs:
                break
            grad_scores = (q - train.p_world) / len(train.p_world)
            grad = np.zeros_like(self.w)
            for ids_row, grad_row in zip(train.address_ids, grad_scores):
                np.add.at(grad, ids_row, grad_row)
            self.w -= lr * grad
        return trace


class LinearKernel:
    def __init__(self, dim: int, rng: np.random.Generator):
        self.w = rng.normal(0.0, 0.02, size=dim)
        self.b = 0.0

    def scores_from_x(self, x: np.ndarray) -> np.ndarray:
        return np.tensordot(x, self.w, axes=([-1], [0])) + self.b

    def train(self, x: np.ndarray, p_world: np.ndarray, *, epochs: int, lr: float, l2: float) -> list[dict]:
        trace = []
        n = x.shape[0]
        for epoch in range(epochs + 1):
            scores = self.scores_from_x(x)
            q = stable_softmax(scores)
            loss = float(kl_div(p_world, q).mean())
            if epoch in {0, 1, 5, 25, 100, 250, 500, epochs}:
                trace.append({"epoch": epoch, "train_kl": loss})
            if epoch == epochs:
                break
            grad_scores = (q - p_world) / n
            self.w -= lr * (np.tensordot(grad_scores, x, axes=([0, 1], [0, 1])) + l2 * self.w)
            self.b -= lr * float(grad_scores.sum())
        return trace


class MLPKernel:
    def __init__(self, dim: int, hidden: int, rng: np.random.Generator):
        self.w1 = rng.normal(0.0, 0.12, size=(dim, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.12, size=hidden)
        self.b2 = 0.0

    def scores_from_x(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_pre = np.tensordot(x, self.w1, axes=([-1], [0])) + self.b1
        h = np.tanh(h_pre)
        scores = np.tensordot(h, self.w2, axes=([-1], [0])) + self.b2
        return scores, h

    def train(self, x: np.ndarray, p_world: np.ndarray, *, epochs: int, lr: float, l2: float) -> list[dict]:
        trace = []
        n = x.shape[0]
        for epoch in range(epochs + 1):
            scores, h = self.scores_from_x(x)
            q = stable_softmax(scores)
            loss = float(kl_div(p_world, q).mean())
            if epoch in {0, 1, 5, 25, 100, 250, 500, epochs}:
                trace.append({"epoch": epoch, "train_kl": loss})
            if epoch == epochs:
                break
            grad_scores = (q - p_world) / n
            grad_w2 = np.tensordot(grad_scores, h, axes=([0, 1], [0, 1])) + l2 * self.w2
            grad_b2 = float(grad_scores.sum())
            grad_h = grad_scores[..., None] * self.w2
            grad_pre = grad_h * (1.0 - h * h)
            grad_w1 = np.tensordot(x, grad_pre, axes=([0, 1], [0, 1])) + l2 * self.w1
            grad_b1 = grad_pre.sum(axis=(0, 1))
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
        return trace


def evaluate_scores(scores: np.ndarray, p_world: np.ndarray) -> dict:
    q = stable_softmax(scores)
    return {
        "kl": float(kl_div(p_world, q).mean()),
        "cross_entropy": float((-np.sum(p_world * np.log(q + EPS), axis=-1)).mean()),
        "top1_agreement": float((np.argmax(q, axis=-1) == np.argmax(p_world, axis=-1)).mean()),
        "entropy_error": float(np.abs(entropy(q) - entropy(p_world)).mean()),
        "calibration_l1": float(np.abs(q - p_world).mean()),
    }


def oracle_scores(ds: Dataset, temperature: float) -> np.ndarray:
    dist = np.sum((ds.patches - ds.query[:, None, :]) ** 2, axis=-1)
    return -dist / temperature


def deductive_checks(rng: np.random.Generator) -> dict:
    paths = all_paths(5)
    mirror_involution_ok = all(mirror_path(mirror_path(p)) == p for p in paths)

    patch_sample = rng.normal(0.0, 1.0, size=3)
    mirror_patch_involution_error = float(np.linalg.norm(mirror_patch(mirror_patch(patch_sample)) - patch_sample))

    query = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    patch_left = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    score_original = -float(np.sum((patch_left - query) ** 2))
    score_conjugate = -float(np.sum((mirror_patch(mirror_patch(patch_left)) - query) ** 2))
    conjugate_equiv_error = abs(score_original - score_conjugate)

    hard_candidates = rng.normal(0.0, 1.0, size=(4, 3))
    target = 2
    write = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    hard = hard_candidates.copy()
    hard[target] += write
    one_hot = np.zeros(4, dtype=np.float64)
    one_hot[target] = 1.0
    soft = hard_candidates + one_hot[:, None] * write[None, :]
    one_hot_soft_plus_error = float(np.linalg.norm(hard - soft))

    return {
        "mirror_involution_ok": bool(mirror_involution_ok),
        "mirror_patch_involution_error": mirror_patch_involution_error,
        "conjugate_equiv_error": conjugate_equiv_error,
        "one_hot_soft_plus_error": one_hot_soft_plus_error,
        "deductive_pass": bool(
            mirror_involution_ok
            and mirror_patch_involution_error < 1e-12
            and conjugate_equiv_error < 1e-12
            and one_hot_soft_plus_error < 1e-12
        ),
    }


def run(out: pathlib.Path, seed: int, epochs: int) -> dict:
    rng = np.random.default_rng(seed)
    temperature = 0.45
    feature_depth = 5
    train = make_dataset(rng, n=900, depth=3, feature_depth=feature_depth, temperature=temperature, query_scale=1.0, patch_scale=1.0)
    test = make_dataset(rng, n=300, depth=3, feature_depth=feature_depth, temperature=temperature, query_scale=1.0, patch_scale=1.0)
    ood = make_dataset(rng, n=300, depth=5, feature_depth=feature_depth, temperature=temperature, query_scale=1.25, patch_scale=1.15)

    max_addr = max(int(train.address_ids.max()), int(test.address_ids.max()), int(ood.address_ids.max()))
    addr = AddressPrior(max_addr)
    addr_trace = addr.train(train, epochs=epochs, lr=0.8)

    x_train_raw = raw_features(train, include_path=False)
    x_test_raw = raw_features(test, include_path=False)
    x_ood_raw = raw_features(ood, include_path=False)
    x_train_tree = raw_features(train, include_path=True)
    x_test_tree = raw_features(test, include_path=True)
    x_ood_tree = raw_features(ood, include_path=True)

    linear = LinearKernel(x_train_raw.shape[-1], rng)
    linear_trace = linear.train(x_train_raw, train.p_world, epochs=epochs, lr=0.35, l2=1e-4)

    mlp = MLPKernel(x_train_raw.shape[-1], hidden=32, rng=rng)
    mlp_trace = mlp.train(x_train_raw, train.p_world, epochs=epochs, lr=0.35, l2=1e-4)

    tree_mlp = MLPKernel(x_train_tree.shape[-1], hidden=40, rng=rng)
    tree_trace = tree_mlp.train(x_train_tree, train.p_world, epochs=epochs, lr=0.35, l2=1e-4)

    def model_metrics(name: str, score_fn) -> dict:
        return {
            "train": evaluate_scores(score_fn(train), train.p_world),
            "test": evaluate_scores(score_fn(test), test.p_world),
            "ood": evaluate_scores(score_fn(ood), ood.p_world),
        }

    metrics = {
        "address_prior": model_metrics("address_prior", lambda ds: addr.scores(ds)),
        "linear_raw": {
            "train": evaluate_scores(linear.scores_from_x(x_train_raw), train.p_world),
            "test": evaluate_scores(linear.scores_from_x(x_test_raw), test.p_world),
            "ood": evaluate_scores(linear.scores_from_x(x_ood_raw), ood.p_world),
        },
        "mlp_raw": {
            "train": evaluate_scores(mlp.scores_from_x(x_train_raw)[0], train.p_world),
            "test": evaluate_scores(mlp.scores_from_x(x_test_raw)[0], test.p_world),
            "ood": evaluate_scores(mlp.scores_from_x(x_ood_raw)[0], ood.p_world),
        },
        "treeheap_prob_kernel": {
            "train": evaluate_scores(tree_mlp.scores_from_x(x_train_tree)[0], train.p_world),
            "test": evaluate_scores(tree_mlp.scores_from_x(x_test_tree)[0], test.p_world),
            "ood": evaluate_scores(tree_mlp.scores_from_x(x_ood_tree)[0], ood.p_world),
        },
        "oracle_fixed_kernel": {
            "train": evaluate_scores(oracle_scores(train, temperature), train.p_world),
            "test": evaluate_scores(oracle_scores(test, temperature), test.p_world),
            "ood": evaluate_scores(oracle_scores(ood, temperature), ood.p_world),
        },
    }

    deductive = deductive_checks(rng)
    inductive_pass = bool(
        metrics["mlp_raw"]["test"]["kl"] < metrics["linear_raw"]["test"]["kl"]
        and metrics["treeheap_prob_kernel"]["ood"]["kl"] < metrics["address_prior"]["ood"]["kl"]
        and metrics["treeheap_prob_kernel"]["ood"]["top1_agreement"] > 0.85
    )
    summary = {
        "seed": seed,
        "epochs": epochs,
        "claim": "M0-SOFT-C09",
        "predict": "P-SOFT05-KL",
        "deductive": deductive,
        "inductive": {
            "world_model": "P_W(a|H,q)=softmax(-||patch(a)-query||^2 / temperature)",
            "temperature": temperature,
            "train_examples": len(train.p_world),
            "test_examples": len(test.p_world),
            "ood_examples": len(ood.p_world),
            "metrics": metrics,
            "inductive_pass": inductive_pass,
        },
        "pilot_pass": bool(deductive["deductive_pass"] and inductive_pass),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for name, trace in [
            ("address_prior", addr_trace),
            ("linear_raw", linear_trace),
            ("mlp_raw", mlp_trace),
            ("treeheap_prob_kernel", tree_trace),
        ]:
            for row in trace:
                f.write(json.dumps({"model": name, **row}, sort_keys=True) + "\n")

    readme = f"""# Deductive vs Inductive Kernel Probe Evidence

Verdict: `pilot_pass = {summary['pilot_pass']}`

## Deductive Proof

| Check | Result |
|---|---:|
| mirror involution | {deductive['mirror_involution_ok']} |
| mirror patch involution error | {deductive['mirror_patch_involution_error']:.6e} |
| conjugate equivalence error | {deductive['conjugate_equiv_error']:.6e} |
| one-hot soft plus error | {deductive['one_hot_soft_plus_error']:.6e} |

These are algebraic checks. They are expected to hold by definition when the
operators are correctly specified.

## Inductive Proof

The world model defines an operation distribution:

```text
P_W(a|H,q) = softmax(-||patch(a)-query||^2 / temperature)
```

Models are trained to imitate `P_W` and evaluated with KL divergence.

| Model | Train KL | Test KL | OOD KL | OOD top1 |
|---|---:|---:|---:|---:|
| address_prior | {metrics['address_prior']['train']['kl']:.6f} | {metrics['address_prior']['test']['kl']:.6f} | {metrics['address_prior']['ood']['kl']:.6f} | {metrics['address_prior']['ood']['top1_agreement']:.3f} |
| linear_raw | {metrics['linear_raw']['train']['kl']:.6f} | {metrics['linear_raw']['test']['kl']:.6f} | {metrics['linear_raw']['ood']['kl']:.6f} | {metrics['linear_raw']['ood']['top1_agreement']:.3f} |
| mlp_raw | {metrics['mlp_raw']['train']['kl']:.6f} | {metrics['mlp_raw']['test']['kl']:.6f} | {metrics['mlp_raw']['ood']['kl']:.6f} | {metrics['mlp_raw']['ood']['top1_agreement']:.3f} |
| treeheap_prob_kernel | {metrics['treeheap_prob_kernel']['train']['kl']:.6f} | {metrics['treeheap_prob_kernel']['test']['kl']:.6f} | {metrics['treeheap_prob_kernel']['ood']['kl']:.6f} | {metrics['treeheap_prob_kernel']['ood']['top1_agreement']:.3f} |
| oracle_fixed_kernel | {metrics['oracle_fixed_kernel']['train']['kl']:.6f} | {metrics['oracle_fixed_kernel']['test']['kl']:.6f} | {metrics['oracle_fixed_kernel']['ood']['kl']:.6f} | {metrics['oracle_fixed_kernel']['ood']['top1_agreement']:.3f} |

## Interpretation

This pilot separates two proof types:

```text
deductive proof: algebraic operations hold by definition
inductive proof: trainable parameters reduce KL to imitate a world distribution
```

It does not prove language ability or WMT translation. It is a controlled proof
that probability kernels can be evaluated as inductive learners with KL/OOD KL.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=700)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seed=args.seed, epochs=args.epochs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
