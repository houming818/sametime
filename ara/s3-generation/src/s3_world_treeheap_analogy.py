#!/usr/bin/env python3
"""Finite-world DIFF -> TRANSPORT -> APPLY analogy proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PAD = 0
PEOPLE = tuple(range(1, 13))
FOODS = tuple(range(13, 25))
NUMBERS = tuple(range(25, 33))
EAT, BUY, COST, LIKE, UNIT = range(33, 38)
VOCAB = 38
RELATIONS = ("food_replace", "number_shift", "person_transport", "role_swap")


def names() -> Dict[int, str]:
    out = {PAD: "_", EAT: "EAT", BUY: "BUY", COST: "COST", LIKE: "LIKE", UNIT: "UNIT"}
    out.update({token: f"P{index}" for index, token in enumerate(PEOPLE)})
    out.update({token: f"FOOD{index}" for index, token in enumerate(FOODS)})
    out.update({token: f"N{index}" for index, token in enumerate(NUMBERS)})
    return out


TOKEN_NAMES = names()


def seq(*values: int) -> List[int]:
    return list(values) + [PAD] * (8 - len(values))


def food_statement(template: str, person: int, food: int, number: int) -> List[int]:
    if template == "eat":
        return seq(person, EAT, food)
    if template == "buy":
        return seq(person, BUY, food, number)
    if template == "price":
        return seq(food, COST, number, UNIT)
    raise ValueError(template)


def split_key(values: Tuple) -> str:
    text = "|".join(map(str, values)).encode()
    return hashlib.sha256(text).hexdigest()


@dataclass
class Quadruple:
    a: List[int]
    b: List[int]
    c: List[int]
    d: List[int]
    relation: int
    key: str


class FiniteWorld:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def sample_one(self, split: str) -> Quadruple:
        while True:
            relation = self.rng.randrange(len(RELATIONS))
            if relation == 0:
                source, target = self.rng.sample(FOODS, 2)
                p1, p2 = self.rng.sample(PEOPLE, 2)
                n1, n2 = self.rng.choice(NUMBERS), self.rng.choice(NUMBERS)
                ta, tc = self.rng.sample(("eat", "buy", "price"), 2)
                a = food_statement(ta, p1, source, n1)
                b = food_statement(ta, p1, target, n1)
                c = food_statement(tc, p2, source, n2)
                d = food_statement(tc, p2, target, n2)
                identity = (relation, source, target, p1, p2, n1, n2, ta, tc)
            elif relation == 1:
                delta = self.rng.choice((-1, 1))
                valid = [i for i in range(len(NUMBERS)) if 0 <= i + delta < len(NUMBERS)]
                ni, mi = self.rng.choice(valid), self.rng.choice(valid)
                food1, food2 = self.rng.sample(FOODS, 2)
                p1, p2 = self.rng.sample(PEOPLE, 2)
                ta, tc = self.rng.sample(("buy", "price"), 2)
                n, n_next = NUMBERS[ni], NUMBERS[ni + delta]
                m, m_next = NUMBERS[mi], NUMBERS[mi + delta]
                a = food_statement(ta, p1, food1, n)
                b = food_statement(ta, p1, food1, n_next)
                c = food_statement(tc, p2, food2, m)
                d = food_statement(tc, p2, food2, m_next)
                identity = (relation, delta, ni, mi, food1, food2, p1, p2, ta, tc)
            elif relation == 2:
                source, target, other1, other2 = self.rng.sample(PEOPLE, 4)
                # Relation must move from subject role in A:B to object role in C:D.
                a, b = seq(source, LIKE, other1), seq(target, LIKE, other1)
                c, d = seq(other2, LIKE, source), seq(other2, LIKE, target)
                identity = (relation, source, target, other1, other2)
            else:
                p, q, r, s = self.rng.sample(PEOPLE, 4)
                a, b = seq(p, LIKE, q), seq(q, LIKE, p)
                c, d = seq(r, LIKE, s), seq(s, LIKE, r)
                identity = (relation, p, q, r, s)
            key = split_key(identity)
            is_valid = int(key[:8], 16) % 5 == 0
            if (split == "valid") == is_valid:
                return Quadruple(a, b, c, d, relation, key)

    def batch(self, size: int, split: str, device: torch.device):
        rows = [self.sample_one(split) for _ in range(size)]
        tensors = [torch.tensor([getattr(row, field) for row in rows], device=device) for field in ("a", "b", "c", "d")]
        relation = torch.tensor([row.relation for row in rows], device=device)
        return (*tensors, relation, rows)


class TreeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, dim)
        self.position = nn.Parameter(torch.randn(8, dim) * 0.02)
        self.depth = nn.Embedding(4, dim)
        self.fold = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def from_leaves(self, leaves: torch.Tensor) -> List[torch.Tensor]:
        levels = [leaves]
        node = leaves
        depth = 1
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            d = self.depth.weight[depth].view(1, 1, -1).expand_as(left)
            node = self.norm(self.fold(torch.cat((left, right, d), dim=-1)))
            levels.append(node)
            depth += 1
        return levels

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        return self.from_leaves(self.embedding(tokens) + self.position[None])


class WorldTreeHeap(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.nodes = nn.Parameter(torch.randn(15, dim) * 0.08)
        self.route = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 3))

    def forward(self, query: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = self.nodes
        if mode == "zero":
            nodes = torch.zeros_like(nodes)
        elif mode == "shuffle":
            nodes = nodes.flip(0)
        batch, dim = query.shape
        frontier = [(0, query.new_ones(batch))]
        output = query.new_zeros(batch, dim)
        entropy = query.new_zeros(batch)
        for depth in range(4):
            following = []
            for index, mass in frontier:
                node = nodes[index].expand(batch, -1)
                if depth == 3:
                    output = output + mass[:, None] * node
                    continue
                left_index, right_index = 2 * index + 1, 2 * index + 2
                left = nodes[left_index].expand(batch, -1)
                right = nodes[right_index].expand(batch, -1)
                probability = F.softmax(self.route(torch.cat((query, node, left, right), dim=-1)), dim=-1)
                entropy = entropy - mass * (probability.clamp_min(1e-9) * probability.clamp_min(1e-9).log()).sum(-1)
                output = output + (mass * probability[:, 0])[:, None] * node
                following.append((left_index, mass * probability[:, 1]))
                following.append((right_index, mass * probability[:, 2]))
            frontier = following
        return output, entropy


class WorldAnalogyTreeHeap(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.encoder = TreeEncoder(dim)
        self.world = WorldTreeHeap(dim)
        self.query = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.relation_depth = nn.Embedding(4, dim)
        self.diff = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.transport = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.apply_kernel = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.apply_gate = nn.Linear(3 * dim, 1)
        self.relation_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, VOCAB)

    def forward(self, a, b, c, world_mode: str = "full"):
        ha, hb, hc = self.encoder(a), self.encoder(b), self.encoder(c)
        query = self.query(torch.cat((ha[-1][:, 0], hb[-1][:, 0], hc[-1][:, 0]), dim=-1))
        world, entropy = self.world(query, world_mode)
        relation = []
        for depth, (left, right) in enumerate(zip(ha, hb)):
            w = world[:, None].expand_as(left)
            d = self.relation_depth.weight[depth].view(1, 1, -1).expand_as(left)
            relation.append(self.relation_norm(self.diff(torch.cat((left, right, right - left, w, d), dim=-1))))
        path_relation = []
        for leaf in range(8):
            values = [relation[depth][:, leaf // (2 ** depth)] for depth in range(4)]
            path_relation.append(torch.stack(values).mean(0))
        path_relation = torch.stack(path_relation, dim=1)
        c_root = hc[-1][:, 0][:, None].expand_as(hc[0])
        w = world[:, None].expand_as(hc[0])
        transported = self.transport(torch.cat((path_relation, hc[0], c_root, w, path_relation * hc[0]), dim=-1))
        apply_input = torch.cat((hc[0], transported, w), dim=-1)
        delta = self.apply_kernel(apply_input)
        gate = torch.sigmoid(self.apply_gate(apply_input))
        output_leaf = self.output_norm(hc[0] + gate * delta)
        output_levels = self.encoder.from_leaves(output_leaf)
        return self.decoder(output_leaf), output_levels, {"world": world, "route_entropy": entropy, "gate": gate}

    def echo(self, tokens):
        levels = self.encoder(tokens)
        return self.decoder(levels[0])


class FlatAnalogyTransformer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, dim)
        self.position = nn.Parameter(torch.randn(24, dim) * 0.02)
        self.segment = nn.Embedding(3, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.decoder = nn.Linear(dim, VOCAB)

    def forward(self, a, b, c):
        tokens = torch.cat((a, b, c), dim=1)
        segments = torch.arange(3, device=tokens.device).repeat_interleave(8)
        state = self.embedding(tokens) + self.position[None] + self.segment(segments)[None]
        state = self.transformer(state)
        return self.decoder(state[:, 16:24])


def weighted_ce(logits, target):
    weight = logits.new_ones(VOCAB)
    weight[PAD] = 0.2
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), weight=weight)


def lexical_baseline(a, b, c):
    output = c.clone()
    for row in range(a.shape[0]):
        original = c[row].clone()
        for position in torch.where(a[row] != b[row])[0]:
            output[row][original == a[row, position]] = b[row, position]
    return output


@torch.no_grad()
def evaluate(tree, flat, world: FiniteWorld, args, device):
    tree.eval(); flat.eval()
    total = 0
    stats = {name: {"token": 0, "exact": 0, "count": 0} for name in RELATIONS}
    totals = {name: 0 for name in ("tree_token", "tree_exact", "flat_token", "flat_exact", "lex_token", "lex_exact", "zero_token", "shuffle_token", "wrong_token", "preserve", "preserve_total")}
    samples = []
    while total < args.valid_examples:
        size = min(args.batch, args.valid_examples - total)
        a, b, c, d, relation, rows = world.batch(size, "valid", device)
        logits, _, _ = tree(a, b, c)
        zero_logits, _, _ = tree(a, b, c, "zero")
        shuffle_logits, _, _ = tree(a, b, c, "shuffle")
        wrong_logits, _, _ = tree(a, b.roll(1, 0), c)
        flat_logits = flat(a, b, c)
        pred, zero, shuffled, wrong = logits.argmax(-1), zero_logits.argmax(-1), shuffle_logits.argmax(-1), wrong_logits.argmax(-1)
        flat_pred, lex = flat_logits.argmax(-1), lexical_baseline(a, b, c)
        for prefix, value in (("tree", pred), ("flat", flat_pred), ("lex", lex)):
            totals[f"{prefix}_token"] += int(value.eq(d).sum())
            totals[f"{prefix}_exact"] += int(value.eq(d).all(-1).sum())
        totals["zero_token"] += int(zero.eq(d).sum())
        totals["shuffle_token"] += int(shuffled.eq(d).sum())
        totals["wrong_token"] += int(wrong.eq(d).sum())
        preserve_mask = d.eq(c)
        totals["preserve"] += int(pred[preserve_mask].eq(d[preserve_mask]).sum())
        totals["preserve_total"] += int(preserve_mask.sum())
        for index, name in enumerate(RELATIONS):
            mask = relation.eq(index)
            stats[name]["token"] += int(pred[mask].eq(d[mask]).sum())
            stats[name]["exact"] += int(pred[mask].eq(d[mask]).all(-1).sum())
            stats[name]["count"] += int(mask.sum())
        if len(samples) < 8:
            for i in range(min(size, 8 - len(samples))):
                show = lambda x: " ".join(TOKEN_NAMES[int(v)] for v in x if int(v) != PAD)
                samples.append({"relation": RELATIONS[int(relation[i])], "A": show(a[i]), "B": show(b[i]), "C": show(c[i]), "gold_D": show(d[i]), "tree_D": show(pred[i]), "flat_D": show(flat_pred[i]), "lexical_D": show(lex[i])})
        total += size
    slots = total * 8
    result = {
        "tree": {"token_accuracy": totals["tree_token"] / slots, "sequence_exact": totals["tree_exact"] / total},
        "flat": {"token_accuracy": totals["flat_token"] / slots, "sequence_exact": totals["flat_exact"] / total},
        "lexical": {"token_accuracy": totals["lex_token"] / slots, "sequence_exact": totals["lex_exact"] / total},
        "interventions": {
            "zero_world_token_accuracy": totals["zero_token"] / slots,
            "shuffle_world_token_accuracy": totals["shuffle_token"] / slots,
            "wrong_B_token_accuracy": totals["wrong_token"] / slots,
        },
        "non_target_preservation": totals["preserve"] / max(1, totals["preserve_total"]),
        "relations": {name: {"token_accuracy": row["token"] / max(1, row["count"] * 8), "sequence_exact": row["exact"] / max(1, row["count"]), "count": row["count"]} for name, row in stats.items()},
        "samples": samples,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_world_treeheap_analogy_smoke")
    parser.add_argument("--seed", type=int, default=71503)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--valid-examples", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_world = FiniteWorld(args.seed)
    valid_world = FiniteWorld(args.seed + 10000)
    tree = WorldAnalogyTreeHeap(args.dim).to(device)
    flat = FlatAnalogyTransformer(args.dim).to(device)
    tree_opt = torch.optim.AdamW(tree.parameters(), lr=args.lr)
    flat_opt = torch.optim.AdamW(flat.parameters(), lr=args.lr)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        tree.train(); flat.train()
        a, b, c, d, _, _ = train_world.batch(args.batch, "train", device)
        logits, output_levels, extra = tree(a, b, c)
        target_levels = tree.encoder(d)
        task = (step - 1) % 3
        if task == 0:
            loss = weighted_ce(logits, d)
            task_name = "analogy_text"
        elif task == 1:
            loss = sum(F.mse_loss(out, target.detach()) for out, target in zip(output_levels, target_levels)) / len(output_levels)
            task_name = "analogy_state"
        else:
            echo_tokens = torch.cat((a, b, c, d), dim=0)
            loss = weighted_ce(tree.echo(echo_tokens), echo_tokens)
            task_name = "private_echo"
        tree_opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(tree.parameters(), 2.0); tree_opt.step()
        flat_loss = weighted_ce(flat(a, b, c), d)
        flat_opt.zero_grad(set_to_none=True); flat_loss.backward(); torch.nn.utils.clip_grad_norm_(flat.parameters(), 2.0); flat_opt.step()
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "task": task_name, "tree_loss": float(loss.detach()), "flat_loss": float(flat_loss.detach()), "world_route_entropy": float(extra["route_entropy"].mean().detach()), "apply_gate": float(extra["gate"].mean().detach()), "elapsed_sec": time.time() - started}
            history.append(row); print(json.dumps(row), flush=True)
    metrics = evaluate(tree, flat, valid_world, args, device)
    tree_acc = metrics["tree"]["token_accuracy"]
    noncopy = min(metrics["relations"][name]["sequence_exact"] for name in ("number_shift", "role_swap"))
    gates = {
        "P1_full_and_noncopy": metrics["tree"]["sequence_exact"] >= 0.90 and noncopy >= 0.80,
        "P2_lexical_margin": metrics["tree"]["sequence_exact"] - metrics["lexical"]["sequence_exact"] >= 0.20,
        "P3_world_causal": tree_acc - max(metrics["interventions"]["zero_world_token_accuracy"], metrics["interventions"]["shuffle_world_token_accuracy"]) >= 0.10,
        "P4_wrong_relation": tree_acc - metrics["interventions"]["wrong_B_token_accuracy"] >= 0.30,
        "P5_selective": metrics["non_target_preservation"] >= 0.95,
        "P6_flat_reported": True,
    }
    status = "supported_single_seed" if all(gates.values()) else ("partial_analogy_no_world" if gates["P1_full_and_noncopy"] and not gates["P3_world_causal"] else "not_supported_by_smoke")
    summary = {
        "claim": "S3-WORLD-ANALOGY-C01", "host": socket.gethostname(), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config": vars(args), "tree_parameters": sum(p.numel() for p in tree.parameters()),
        "flat_parameters": sum(p.numel() for p in flat.parameters()), "metrics": metrics,
        "gates": gates, "status": status, "elapsed_sec": time.time() - started,
        "boundary": "Finite symbolic world only; no natural-language semantic or architecture-superiority claim."
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
    torch.save({"tree": tree.state_dict(), "flat": flat.state_dict(), "config": vars(args)}, output / "checkpoint.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
