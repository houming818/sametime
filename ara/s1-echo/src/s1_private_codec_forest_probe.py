#!/usr/bin/env python3
"""S1 private-codec TreeHeap forest proof.

This is a small, explicit ARA proof for SPR-048:

  * Theta is a learnable parameter TreeHeap forest.
  * Q/kernel requests do not contain the answer.
  * Scalar loss writes into Theta.arr[i].
  * Two heads compose serially on a held-out intermediate state.

The proof is intentionally toy-sized. It is not a natural-language proof.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = ["rice", "noodle", "apple", "mango", "amoxicillin", "ibuprofen", "stone", "car"]
IDX = {name: i for i, name in enumerate(VOCAB)}
FOOD = ["rice", "noodle", "apple", "mango"]
FRUIT = ["apple", "mango"]
SOURCE_BUCKETS = {
    "food": ["rice", "noodle", "apple", "mango"],
    "rice_apple": ["rice", "apple", "stone"],
    "noodle_mango": ["noodle", "mango", "car"],
}


def distribution(names: list[str]) -> torch.Tensor:
    y = torch.zeros(len(VOCAB), dtype=torch.float32)
    for name in names:
        y[IDX[name]] = 1.0
    return y / y.sum().clamp_min(1.0)


TARGET_FOOD = distribution(FOOD)


def kl_like_loss(log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy against a probability bucket."""
    return -(target * log_probs).sum()


class ParameterTreeHeapHead(nn.Module):
    """A 7-node parameter TreeHeap with a fixed recursive read kernel."""

    def __init__(self, dim: int, alpha: float = 0.5, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.arr = nn.Parameter(torch.randn(7, dim, generator=gen) * 0.05)
        self.alpha = alpha

    def read_node(self, idx: int) -> torch.Tensor:
        left = 2 * idx + 1
        right = 2 * idx + 2
        out = self.arr[idx]
        if left < self.arr.shape[0]:
            out = out + self.alpha * self.read_node(left)
        if right < self.arr.shape[0]:
            out = out + self.alpha * self.read_node(right)
        return out

    def root_logits(self) -> torch.Tensor:
        return self.read_node(0)


class PrivateCodecForest(nn.Module):
    def __init__(self, dim: int, seed: int):
        super().__init__()
        self.sources = nn.ModuleDict(
            {name: ParameterTreeHeapHead(dim, seed=seed + 10 + k) for k, name in enumerate(SOURCE_BUCKETS)}
        )
        self.fruit_filter = ParameterTreeHeapHead(dim, seed=seed + 100)
        self.constant = ParameterTreeHeapHead(dim, seed=seed + 101)

    def source_distribution(self, name: str) -> torch.Tensor:
        return F.softmax(self.sources[name].root_logits(), dim=-1)

    def fruit_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.fruit_filter.root_logits())

    def constant_distribution(self) -> torch.Tensor:
        return F.softmax(self.constant.root_logits(), dim=-1)

    def apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.fruit_mask()
        return y / y.sum().clamp_min(1e-8)


def filter_train_examples(seed: int) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    """Examples for learning an input-dependent fruit filter.

    The exact FOOD bucket is intentionally not included; it is the held-out
    composition target.
    """

    base = [
        (["apple", "stone", "car"], ["apple"], "apple_noise"),
        (["mango", "stone", "ibuprofen"], ["mango"], "mango_noise"),
        (["apple", "mango", "stone", "car"], ["apple", "mango"], "fruits_noise"),
        (["rice", "apple", "ibuprofen"], ["apple"], "rice_apple_drug"),
        (["noodle", "mango", "amoxicillin"], ["mango"], "noodle_mango_drug"),
        (["apple", "mango", "amoxicillin", "ibuprofen"], ["apple", "mango"], "fruit_drugs"),
    ]
    rng = random.Random(seed)
    rows = []
    for names, target, label in base:
        x = distribution(names)
        rows.append((x, distribution(target), label))

    # Add smooth mixtures so the filter cannot solve only one-hot set strings.
    for k in range(12):
        weights = torch.zeros(len(VOCAB), dtype=torch.float32)
        chosen = rng.sample(VOCAB, rng.randint(3, 6))
        for name in chosen:
            weights[IDX[name]] = rng.random() + 0.05
        weights = weights / weights.sum()
        fruit_names = [name for name in FRUIT if weights[IDX[name]] > 0]
        if not fruit_names:
            continue
        target = torch.zeros(len(VOCAB), dtype=torch.float32)
        for name in fruit_names:
            target[IDX[name]] = weights[IDX[name]]
        target = target / target.sum()
        rows.append((weights, target, f"smooth_{k}"))
    return rows


def bucket_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    eps = 1e-9
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    ce = float(-(target * torch.log(pred.clamp_min(eps))).sum())
    l1 = float((pred - target).abs().sum())
    top2 = [VOCAB[i] for i in torch.topk(pred, 2).indices.tolist()]
    target_support = [VOCAB[i] for i, v in enumerate(target.tolist()) if v > 1e-6]
    pred_support = [VOCAB[i] for i, v in enumerate(pred.tolist()) if v > 0.12]
    return {
        "cross_entropy": ce,
        "l1": l1,
        "top2": top2,
        "pred_support_gt_0_12": pred_support,
        "target_support": target_support,
        "probabilities": {name: float(pred[i]) for i, name in enumerate(VOCAB)},
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    forest = PrivateCodecForest(len(VOCAB), args.seed)
    initial = {
        "source_arr_l2": {name: float(head.arr.detach().norm()) for name, head in forest.sources.items()},
        "filter_arr_l2": float(forest.fruit_filter.arr.detach().norm()),
        "constant_arr_l2": float(forest.constant.arr.detach().norm()),
    }
    examples = filter_train_examples(args.seed)
    opt = torch.optim.Adam(forest.parameters(), lr=args.lr)
    trace = []

    for epoch in range(args.epochs + 1):
        source_loss = torch.tensor(0.0)
        for name, bucket in SOURCE_BUCKETS.items():
            pred = forest.source_distribution(name)
            source_loss = source_loss + kl_like_loss(torch.log(pred.clamp_min(1e-9)), distribution(bucket))
        source_loss = source_loss / len(SOURCE_BUCKETS)

        filter_loss = torch.tensor(0.0)
        constant_loss = torch.tensor(0.0)
        for x, y, _ in examples:
            out = forest.apply_filter(x)
            filter_loss = filter_loss + kl_like_loss(torch.log(out.clamp_min(1e-9)), y)
            const = forest.constant_distribution()
            constant_loss = constant_loss + kl_like_loss(torch.log(const.clamp_min(1e-9)), y)
        filter_loss = filter_loss / len(examples)
        constant_loss = constant_loss / len(examples)
        loss = source_loss + filter_loss + constant_loss

        if epoch in {0, 1, 2, 5, 10, 25, 50, 100, 250, 500, args.epochs}:
            with torch.no_grad():
                heldout_rows = heldout_composition_metrics(forest)
                trace.append(
                    {
                        "epoch": epoch,
                        "loss": float(loss),
                        "source_loss": float(source_loss),
                        "filter_loss": float(filter_loss),
                        "constant_loss": float(constant_loss),
                        "heldout_mean_ce": heldout_rows["mean_ce"],
                        "heldout_all_top2_correct": heldout_rows["all_top2_correct"],
                        "food_top2": bucket_metrics(forest.source_distribution("food"), TARGET_FOOD)["top2"],
                    }
                )

        if epoch == args.epochs:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        heldout = heldout_composition_metrics(forest)
        no_training = PrivateCodecForest(len(VOCAB), args.seed + 1000)
        untrained_heldout = heldout_composition_metrics(no_training)
        constant_heldout = constant_composition_metrics(forest)

        final = {
            "sources": {
                name: bucket_metrics(forest.source_distribution(name), distribution(bucket))
                for name, bucket in SOURCE_BUCKETS.items()
            },
            "heldout_serial_compositions": heldout,
            "constant_baseline_compositions": constant_heldout,
            "untrained_forest_compositions": untrained_heldout,
        }
        gradients_moved_theta = {
            "source_arr_delta_l2": {
                name: float(head.arr.detach().norm()) - initial["source_arr_l2"][name]
                for name, head in forest.sources.items()
            },
            "filter_arr_delta_l2": float((forest.fruit_filter.arr.detach()).norm()) - initial["filter_arr_l2"],
            "constant_arr_delta_l2": float((forest.constant.arr.detach()).norm()) - initial["constant_arr_l2"],
        }

    pass_checks = {
        "source_ce_low": all(row["cross_entropy"] < args.max_food_ce for row in final["sources"].values()),
        "heldout_composition_ce_low": final["heldout_serial_compositions"]["mean_ce"] < args.max_heldout_ce,
        "heldout_support_correct": final["heldout_serial_compositions"]["all_support_correct"],
        "constant_baseline_worse": final["constant_baseline_compositions"]["mean_ce"]
        > final["heldout_serial_compositions"]["mean_ce"] + args.min_baseline_gap,
        "untrained_worse": final["untrained_forest_compositions"]["mean_ce"]
        > final["heldout_serial_compositions"]["mean_ce"] + args.min_baseline_gap,
        "theta_moved": all(abs(v) > 0.1 for v in gradients_moved_theta["source_arr_delta_l2"].values())
        and abs(gradients_moved_theta["filter_arr_delta_l2"]) > 0.1,
    }

    return {
        "claim": "S1-PRIVATE-CODEC-C01",
        "predict": "P-S1-PRIVATE-CODEC01",
        "host": args.host_label,
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "vocab": VOCAB,
            "food_target": FOOD,
            "fruit_target": FRUIT,
            "filter_train_examples": [
                {
                    "label": label,
                    "input_support": [VOCAB[i] for i, v in enumerate(x.tolist()) if v > 1e-6],
                    "target_support": [VOCAB[i] for i, v in enumerate(y.tolist()) if v > 1e-6],
                }
                for x, y, label in examples
            ],
            "heldout_compositions": {
                name: f"fruit_filter({name}_distribution)" for name in SOURCE_BUCKETS
            },
            "direct_heldout_loss_used": False,
        },
        "metrics": final,
        "theta_movement": gradients_moved_theta,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "trace": trace,
        "interpretation": {
            "supported_if_pass": "Scalar loss wrote information into a parameter TreeHeap forest, and two heads composed serially on a held-out intermediate bucket.",
            "not_proved": [
                "not natural language semantics",
                "not unsupervised ontology discovery",
                "not WMT translation",
                "not Transformer superiority",
            ],
        },
    }


def fruit_target_for_bucket(names: list[str]) -> torch.Tensor:
    fruits = [name for name in names if name in FRUIT]
    return distribution(fruits)


def heldout_composition_metrics(forest: PrivateCodecForest) -> dict[str, object]:
    rows = {}
    ces = []
    support_ok = []
    top2_ok = []
    for name, bucket in SOURCE_BUCKETS.items():
        pred = forest.apply_filter(forest.source_distribution(name))
        target = fruit_target_for_bucket(bucket)
        row = bucket_metrics(pred, target)
        rows[name] = row
        ces.append(row["cross_entropy"])
        support_ok.append(set(row["pred_support_gt_0_12"]) == set(row["target_support"]))
        topk = row["top2"][: len(row["target_support"])]
        top2_ok.append(set(topk) == set(row["target_support"]))
    return {
        "by_source": rows,
        "mean_ce": float(sum(ces) / len(ces)),
        "all_support_correct": all(support_ok),
        "all_top2_correct": all(top2_ok),
    }


def constant_composition_metrics(forest: PrivateCodecForest) -> dict[str, object]:
    rows = {}
    ces = []
    support_ok = []
    top2_ok = []
    pred = forest.constant_distribution()
    for name, bucket in SOURCE_BUCKETS.items():
        target = fruit_target_for_bucket(bucket)
        row = bucket_metrics(pred, target)
        rows[name] = row
        ces.append(row["cross_entropy"])
        support_ok.append(set(row["pred_support_gt_0_12"]) == set(row["target_support"]))
        topk = row["top2"][: len(row["target_support"])]
        top2_ok.append(set(topk) == set(row["target_support"]))
    return {
        "by_source": rows,
        "mean_ce": float(sum(ces) / len(ces)),
        "all_support_correct": all(support_ok),
        "all_top2_correct": all(top2_ok),
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["trace"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    readme = f"""# S1 Private Codec Forest Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
source_food_ce = {summary['metrics']['sources']['food']['cross_entropy']:.6f}
heldout_serial_mean_ce = {summary['metrics']['heldout_serial_compositions']['mean_ce']:.6f}
constant_baseline_mean_ce = {summary['metrics']['constant_baseline_compositions']['mean_ce']:.6f}
untrained_composition_mean_ce = {summary['metrics']['untrained_forest_compositions']['mean_ce']:.6f}
heldout_support_correct = {summary['metrics']['heldout_serial_compositions']['all_support_correct']}
```

## Meaning

`Theta_food` and `Theta_filter` are separate parameter TreeHeaps. The direct
held-out composition loss is not used during training:

```text
direct_heldout_loss_used = {summary['config']['direct_heldout_loss_used']}
```

The test asks whether:

```text
K_filter(K_food(H0; Theta_food); Theta_filter)
```

recovers the fruit bucket from the learned food bucket.

## Boundary

This is a toy learning-mechanism proof. It is not natural-language semantics,
WMT translation, or unsupervised world-model discovery.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_private_codec_forest_probe")
    parser.add_argument("--seed", type=int, default=4801)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--max-food-ce", type=float, default=1.45)
    parser.add_argument("--max-heldout-ce", type=float, default=0.80)
    parser.add_argument("--min-baseline-gap", type=float, default=0.15)
    parser.add_argument("--host-label", default="local")
    args = parser.parse_args()

    summary = train(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
