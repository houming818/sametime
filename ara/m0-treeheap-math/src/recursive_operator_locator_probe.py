#!/usr/bin/env python3
"""Test whether one local TreeHeap route kernel extrapolates by recursion."""
from __future__ import annotations

import argparse
import json
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import algebraic_operator_codec_probe as base


ACTIONS = ["stop", "left", "right"]


def generate_level_samples(count: int, levels: list[int], seed: int) -> list[base.Sample]:
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        target_level = rng.choice(levels)
        op_depth = rng.randint(1, 2)
        minimum_tree_depth = target_level + op_depth
        tree_depth = rng.randint(max(5, minimum_tree_depth), base.MAX_TREE_DEPTH)
        address = rng.randint(2 ** target_level, 2 ** (target_level + 1) - 1)
        op = rng.randrange(len(base.OPS))
        root = rng.randint(1, 7)
        target = base.legal_heap(tree_depth, root)
        size = 2 ** (tree_depth + 1) - 1
        delta_index = rng.randrange(len(base.DELTAS))
        if op == 0:
            corrupted = base.mirror_subtree(target, address, op_depth, size)
            delta_index = 0
        else:
            corrupted = base.plus_subtree(target, address, op_depth, base.DELTAS[delta_index], size)
        samples.append(base.Sample(corrupted, target, size, tree_depth, op, address, op_depth, delta_index))
    return samples


def local_input(features: torch.Tensor, size: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Read current/left/right content features; no target/path feature enters."""
    batch = torch.arange(features.shape[0], device=features.device)
    node_index = (nodes - 1).clamp(0, base.MAX_NODES - 1)
    left = nodes * 2
    right = left + 1
    left_index = (left - 1).clamp(0, base.MAX_NODES - 1)
    right_index = (right - 1).clamp(0, base.MAX_NODES - 1)
    current_f = features[batch, node_index, :-1]
    left_f = features[batch, left_index, :-1]
    right_f = features[batch, right_index, :-1]
    left_valid = (left <= size).to(features.dtype)[:, None]
    right_valid = (right <= size).to(features.dtype)[:, None]
    left_f = left_f * left_valid
    right_f = right_f * right_valid
    return torch.cat([
        current_f, left_f, right_f,
        left_f - current_f, right_f - current_f,
        left_valid, right_valid,
    ], dim=-1)


class RecursiveLocator(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        feature_dim = 5 + 2 * (base.MAX_OP_DEPTH + 1)  # exclude absolute node level
        self.route_kernel = nn.Sequential(
            nn.Linear(feature_dim * 5 + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(ACTIONS)),
        )
        self.program_kernel = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.op_head = nn.Linear(hidden, len(base.OPS))
        self.delta_head = nn.Linear(hidden, len(base.DELTAS))

    def route(self, features, size, nodes):
        return self.route_kernel(local_input(features, size, nodes))

    def program(self, features, nodes):
        rows = torch.arange(features.shape[0], device=features.device)
        current = features[rows, (nodes - 1).clamp(0, base.MAX_NODES - 1), :-1]
        hidden = self.program_kernel(current)
        return self.op_head(hidden), self.delta_head(hidden)


class OneShotStructural(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        feature_dim = 5 + 2 * (base.MAX_OP_DEPTH + 1) + 1
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, features, valid):
        return self.net(features).squeeze(-1).masked_fill(~valid, -1e9)


class FlatAddress(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(base.MAX_NODES, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, base.MAX_NODES),
        )

    def forward(self, arr, size):
        scale = arr[:, 1:].abs().amax(-1, keepdim=True).clamp_min(1)
        logits = self.net(arr[:, 1:] / scale)
        valid = torch.arange(base.MAX_NODES, device=arr.device)[None] < size[:, None]
        return logits.masked_fill(~valid, -1e9)


def oracle_actions(target: torch.Tensor, nodes: torch.Tensor, step: int) -> torch.Tensor:
    target_level = torch.floor(torch.log2(target.to(torch.float32))).to(torch.long)
    shift = (target_level - step - 1).clamp_min(0)
    next_bit = torch.bitwise_and(torch.bitwise_right_shift(target, shift), 1)
    return torch.where(target.eq(nodes), torch.zeros_like(target), 1 + next_bit)


def depth_from_features(features: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(features.shape[0], device=features.device)
    node_index = (nodes - 1).clamp(0, base.MAX_NODES - 1)
    logits = torch.stack([
        features[rows, node_index, 5 + 2 * depth]
        for depth in range(1, base.MAX_OP_DEPTH + 1)
    ], dim=-1)
    return logits.argmax(-1) + 1


def train_epoch(recursive, one_shot, flat, loader, optimizers, device):
    recursive.train(); one_shot.train(); flat.train()
    totals = {"recursive": 0.0, "one_shot": 0.0, "flat": 0.0}; batches = 0
    for arr, _, size, op, address0, _, delta in loader:
        arr, size = arr.to(device), size.to(device)
        target = address0.to(device) + 1
        op, delta = op.to(device), delta.to(device)
        features, valid = base.algebra_features(arr, size)

        nodes = torch.ones_like(target)
        active = torch.ones_like(target, dtype=torch.bool)
        route_loss = torch.zeros((), device=device)
        decisions = 0
        for step in range(3):  # train target levels are only 0..2
            if not bool(active.any()):
                break
            labels = oracle_actions(target, nodes, step)
            logits = recursive.route(features, size, nodes)
            route_loss = route_loss + F.cross_entropy(logits[active], labels[active])
            decisions += 1
            move = active & labels.ne(0)
            nodes = torch.where(move, nodes * 2 + labels - 1, nodes)
            active = active & labels.ne(0)
        route_loss = route_loss / max(decisions, 1)
        op_logits, delta_logits = recursive.program(features, target)
        program_loss = F.cross_entropy(op_logits, op)
        plus = op.eq(1)
        if bool(plus.any()):
            program_loss = program_loss + F.cross_entropy(delta_logits[plus], delta[plus])
        recursive_loss = route_loss + program_loss
        optimizers["recursive"].zero_grad(set_to_none=True)
        recursive_loss.backward(); optimizers["recursive"].step()

        one_loss = F.cross_entropy(one_shot(features.detach(), valid), target - 1)
        optimizers["one_shot"].zero_grad(set_to_none=True)
        one_loss.backward(); optimizers["one_shot"].step()

        flat_loss = F.cross_entropy(flat(arr, size), target - 1)
        optimizers["flat"].zero_grad(set_to_none=True)
        flat_loss.backward(); optimizers["flat"].step()

        totals["recursive"] += float(recursive_loss.item())
        totals["one_shot"] += float(one_loss.item())
        totals["flat"] += float(flat_loss.item())
        batches += 1
    return {key + "_loss": value / batches for key, value in totals.items()}


@torch.no_grad()
def evaluate(recursive, one_shot, flat, samples, device):
    recursive.eval(); one_shot.eval(); flat.eval()
    loader = DataLoader(base.HeapDataset(samples), batch_size=256, shuffle=False)
    totals = {
        "samples": 0, "teacher_actions": 0, "teacher_correct": 0,
        "route": 0, "restore": 0, "one_shot": 0, "flat": 0,
        "steps": 0, "stopped": 0,
    }
    action_confusion = torch.zeros(3, 3, dtype=torch.long)
    examples = []
    for arr, target_arr, size, op, address0, depth0, delta in loader:
        arr, size = arr.to(device), size.to(device)
        target_address = address0.to(device) + 1
        features, valid = base.algebra_features(arr, size)

        # Teacher-forced local accuracy measures the reusable rule itself.
        teacher_nodes = torch.ones_like(target_address)
        teacher_active = torch.ones_like(target_address, dtype=torch.bool)
        max_gold_level = int(torch.floor(torch.log2(target_address.float())).max().item())
        for step in range(max_gold_level + 1):
            if not bool(teacher_active.any()):
                break
            labels = oracle_actions(target_address, teacher_nodes, step)
            predictions = recursive.route(features, size, teacher_nodes).argmax(-1)
            mask = teacher_active
            totals["teacher_actions"] += int(mask.sum())
            totals["teacher_correct"] += int(predictions[mask].eq(labels[mask]).sum())
            for gold, pred in zip(labels[mask].cpu(), predictions[mask].cpu()):
                action_confusion[int(gold), int(pred)] += 1
            move = mask & labels.ne(0)
            teacher_nodes = torch.where(move, teacher_nodes * 2 + labels - 1, teacher_nodes)
            teacher_active = teacher_active & labels.ne(0)

        # Autonomous inference uses its own previous decisions.
        nodes = torch.ones_like(target_address)
        done = torch.zeros_like(target_address, dtype=torch.bool)
        invalid = torch.zeros_like(target_address, dtype=torch.bool)
        executed = torch.zeros_like(target_address)
        for _ in range(base.MAX_TREE_DEPTH + 1):
            active = ~done & ~invalid
            if not bool(active.any()):
                break
            actions = recursive.route(features, size, nodes).argmax(-1)
            executed += active.to(executed.dtype)
            stop = active & actions.eq(0)
            move = active & actions.ne(0)
            next_nodes = nodes * 2 + actions - 1
            invalid = invalid | (move & next_nodes.gt(size))
            nodes = torch.where(move, next_nodes, nodes)
            done = done | stop

        route_ok = done & ~invalid & nodes.eq(target_address)
        totals["route"] += int(route_ok.sum())
        totals["stopped"] += int(done.sum())
        totals["steps"] += int(executed.sum())
        totals["one_shot"] += int((one_shot(features, valid).argmax(-1) + 1).eq(target_address).sum())
        totals["flat"] += int((flat(arr, size).argmax(-1) + 1).eq(target_address).sum())

        safe_nodes = torch.minimum(nodes, size).clamp_min(1)
        pred_op, pred_delta = recursive.program(features, safe_nodes)
        pred_op, pred_delta = pred_op.argmax(-1), pred_delta.argmax(-1)
        pred_depth = depth_from_features(features, safe_nodes)
        for row in range(arr.shape[0]):
            restored = base.execute_inverse(
                arr[row].cpu().numpy(), int(pred_op[row]), int(safe_nodes[row]),
                int(pred_depth[row]), int(pred_delta[row]), int(size[row]),
            )
            exact = np.array_equal(
                restored[:int(size[row]) + 1],
                target_arr[row].numpy()[:int(size[row]) + 1],
            )
            totals["restore"] += int(exact)
            if len(examples) < 5:
                examples.append({
                    "gold_address": int(target_address[row]),
                    "pred_address": int(nodes[row]),
                    "gold_level": base.level(int(target_address[row])),
                    "stopped": bool(done[row]), "route_exact": bool(route_ok[row]),
                    "restore_exact": exact,
                })

        totals["samples"] += arr.shape[0]

    n = totals["samples"]
    local_p = totals["teacher_correct"] / totals["teacher_actions"]
    mean_steps = totals["steps"] / n
    return {
        "teacher_forced_action_accuracy": local_p,
        "autonomous_route_exact": totals["route"] / n,
        "exact_restore": totals["restore"] / n,
        "one_shot_structural_address_accuracy": totals["one_shot"] / n,
        "flat_address_accuracy": totals["flat"] / n,
        "stopped_fraction": totals["stopped"] / n,
        "mean_executed_steps": mean_steps,
        "p_power_mean_steps": local_p ** mean_steps,
        "action_confusion_gold_rows": action_confusion.tolist(),
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/m0-treeheap-math/evidence/recursive_operator_locator_probe")
    ap.add_argument("--train-samples", type=int, default=12000)
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    train_samples = generate_level_samples(args.train_samples, [0, 1, 2], args.seed)
    splits = {
        "iid_levels_0_2": generate_level_samples(args.test_samples, [0, 1, 2], args.seed + 1),
        "ood_level_3": generate_level_samples(args.test_samples, [3], args.seed + 2),
        "ood_level_4": generate_level_samples(args.test_samples, [4], args.seed + 3),
        "ood_level_5": generate_level_samples(args.test_samples, [5], args.seed + 4),
    }
    loader = DataLoader(base.HeapDataset(train_samples), batch_size=args.batch, shuffle=True)
    recursive = RecursiveLocator(args.hidden).to(device)
    one_shot = OneShotStructural(args.hidden).to(device)
    flat = FlatAddress(args.hidden).to(device)
    optimizers = {
        "recursive": torch.optim.AdamW(recursive.parameters(), lr=args.lr),
        "one_shot": torch.optim.AdamW(one_shot.parameters(), lr=args.lr),
        "flat": torch.optim.AdamW(flat.parameters(), lr=args.lr),
    }
    started = time.time(); trace = []
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, **train_epoch(recursive, one_shot, flat, loader, optimizers, device)}
        trace.append(row); print(json.dumps(row), flush=True)
    results = {name: evaluate(recursive, one_shot, flat, samples, device) for name, samples in splits.items()}
    ood_pass = all(results[f"ood_level_{level}"]["autonomous_route_exact"] >= 0.80 for level in (3, 4, 5))
    baseline_pass = all(
        results[f"ood_level_{level}"]["autonomous_route_exact"] > max(
            results[f"ood_level_{level}"]["one_shot_structural_address_accuracy"],
            results[f"ood_level_{level}"]["flat_address_accuracy"],
        ) for level in (3, 4, 5)
    )
    restore_strong_pass = all(results[f"ood_level_{level}"]["exact_restore"] >= 0.75 for level in (3, 4, 5))
    summary = {
        "claim": "M0-RECUR-C01", "host": socket.gethostname(),
        "seconds": time.time() - started, "config": vars(args),
        "parameters": {
            "recursive": sum(p.numel() for p in recursive.parameters()),
            "one_shot_structural": sum(p.numel() for p in one_shot.parameters()),
            "flat_address": sum(p.numel() for p in flat.parameters()),
        },
        "gates": {"ood_route_gate": ood_pass, "beats_baselines": baseline_pass, "strong_restore_gate": restore_strong_pass},
        "results": results,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace) + "\n", encoding="utf8")
    (out / "README.md").write_text("# Recursive Operator Locator Probe\n\nSee `summary.json`.\n", encoding="utf8")
    torch.save({"state_dict": recursive.state_dict(), "config": vars(args)}, out / "checkpoint_recursive.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
