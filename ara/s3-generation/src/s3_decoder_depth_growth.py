#!/usr/bin/env python3
"""Variable-arity TreeHeap depth-growth pilot on real Chinese continuation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_annealed_frontier_pretrain as data


TRAIN_ARITIES = (2, 4, 8)
ALL_ARITIES = tuple(range(2, 9))
VARIANTS = ("tree", "random", "flat")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def loader(args, split: str, seed: int, batch: int | None = None) -> DataLoader:
    dataset = data.FutureBlocks(
        Path(args.data_root), args.spm_model, split, seed, args.context, args.future
    )
    return DataLoader(
        dataset, batch_size=batch or args.batch, collate_fn=data.collate,
        num_workers=0, pin_memory=args.device.startswith("cuda"),
    )


class FoldKernel(nn.Module):
    def __init__(self, dim: int, max_arity: int):
        super().__init__()
        self.slot = nn.Embedding(max_arity, dim)
        self.child = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.score = nn.Linear(dim, 1)
        self.out = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, children: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        arity = children.shape[2]
        slots = self.slot.weight[:arity][None, None]
        encoded = self.child(children + slots)
        scores = self.score(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        weight = F.softmax(scores, dim=-1)
        pooled = (weight[..., None] * encoded).sum(2)
        return self.norm(pooled + self.out(pooled))


class ReadKernel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        score = (self.q(query)[:, None] * self.k(nodes)).sum(-1) / math.sqrt(nodes.shape[-1])
        context = (F.softmax(score, -1)[..., None] * self.v(nodes)).sum(1)
        return self.norm(query + self.update(torch.cat((query, context), -1)))


class DepthGrowthModel(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, max_arity: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.fold = FoldKernel(dim, max_arity)
        self.flat_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.reader = ReadKernel(dim)
        self.target_embedding = nn.Embedding(vocab, dim)
        self.cell = nn.GRUCell(dim, hidden)
        self.query = nn.Linear(hidden, dim)
        self.output = nn.Linear(hidden + dim, vocab)
        self.hidden = hidden

    @staticmethod
    def permutation(width: int, arity: int, level: int, device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(910_003 + width * 97 + arity * 1009 + level * 7919)
        return torch.randperm(width, generator=generator).to(device)

    def tree_levels(self, leaves: torch.Tensor, arity: int, random_links: bool) -> List[torch.Tensor]:
        node = leaves
        bottom = [node]
        level = 0
        while node.shape[1] > 1:
            if random_links:
                node = node[:, self.permutation(node.shape[1], arity, level, node.device)]
            width = node.shape[1]
            groups = math.ceil(width / arity)
            padded = groups * arity
            mask = torch.arange(padded, device=node.device)[None] < width
            if padded > width:
                node = F.pad(node, (0, 0, 0, padded - width))
            children = node.reshape(node.shape[0], groups, arity, node.shape[-1])
            child_mask = mask.reshape(1, groups, arity).expand(node.shape[0], -1, -1)
            node = self.fold(children, child_mask)
            bottom.append(node)
            level += 1
        return list(reversed(bottom))

    def flat_levels(self, leaves: torch.Tensor, arity: int) -> List[torch.Tensor]:
        counts = []
        width = leaves.shape[1]
        while True:
            counts.append(width)
            if width == 1:
                break
            width = math.ceil(width / arity)
        levels = []
        source = leaves.transpose(1, 2)
        for count in reversed(counts):
            pooled = F.adaptive_avg_pool1d(source, count).transpose(1, 2)
            levels.append(self.flat_projection(pooled))
        return levels

    def levels(self, source: torch.Tensor, arity: int, variant: str) -> List[torch.Tensor]:
        leaves = self.embedding(source)
        if variant == "tree":
            return self.tree_levels(leaves, arity, False)
        if variant == "random":
            return self.tree_levels(leaves, arity, True)
        if variant == "flat":
            return self.flat_levels(leaves, arity)
        raise ValueError(variant)

    def read(self, query: torch.Tensor, levels: Sequence[torch.Tensor], dose: int, intervention: str) -> torch.Tensor:
        root = levels[0]
        for depth, nodes in enumerate(levels[: dose + 1]):
            if intervention == "repeated_root":
                nodes = root.expand(-1, nodes.shape[1], -1)
            query = self.reader(query, nodes)
        return query

    def teacher(self, source, target, bos: int, arity: int, dose: int, variant: str, intervention: str = "native"):
        effective = "random" if intervention == "shuffled_links" else variant
        levels = self.levels(source, arity, effective)
        state = source.new_zeros((source.shape[0], self.hidden), dtype=self.embedding.weight.dtype)
        previous = torch.full((source.shape[0],), bos, device=source.device, dtype=torch.long)
        output = []
        for step in range(target.shape[1]):
            state = self.cell(self.target_embedding(previous), state)
            read = self.read(self.query(state), levels, dose, intervention)
            output.append(self.output(torch.cat((state, read), -1)))
            previous = target[:, step]
        return torch.stack(output, 1), len(levels)

    @torch.no_grad()
    def greedy(self, source, bos: int, eos: int, length: int, arity: int, dose: int, variant: str):
        levels = self.levels(source, arity, variant)
        state = source.new_zeros((source.shape[0], self.hidden), dtype=self.embedding.weight.dtype)
        previous = torch.full((source.shape[0],), bos, device=source.device, dtype=torch.long)
        output = []
        for _ in range(length):
            state = self.cell(self.target_embedding(previous), state)
            read = self.read(self.query(state), levels, dose, "native")
            previous = self.output(torch.cat((state, read), -1)).argmax(-1)
            output.append(previous)
        return torch.stack(output, 1)


def loss_of(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), target.flatten())


@torch.no_grad()
def evaluate(model, args, variant: str, arity: int, dose: int, intervention: str = "native") -> float:
    model.eval()
    stream = loader(args, "valid", args.seed + 50_000, args.eval_batch)
    total_loss = total_tokens = 0
    for batch_number, (source, target) in enumerate(stream, 1):
        source, target = source.to(args.device), target.to(args.device)
        logits, _ = model.teacher(source, target, args.bos, arity, dose, variant, intervention)
        total_loss += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
        total_tokens += target.numel()
        if batch_number >= args.eval_batches:
            break
    return total_loss / total_tokens


def train_variant(args, variant: str, output: Path) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = DepthGrowthModel(args.vocab, args.dim, args.hidden, args.max_arity).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    stream = iter(loader(args, "train", args.seed + sum(map(ord, variant))))
    rng = random.Random(args.seed + 7000 + sum(map(ord, variant)))
    trace = []
    finite = True
    started = time.time()
    for step in range(1, args.steps + 1):
        source, target = next(stream)
        source, target = source.to(args.device), target.to(args.device)
        arity = rng.choice(TRAIN_ARITIES)
        # Number of frontiers is determined by arity and fixed context width.
        depth_count = 1
        width = args.context
        while width > 1:
            width = math.ceil(width / arity)
            depth_count += 1
        dose = rng.randrange(depth_count)
        logits, _ = model.teacher(source, target, args.bos, arity, dose, variant)
        loss = loss_of(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"variant": variant, "step": step, "arity": arity, "dose": dose,
                   "train_nll": float(loss.detach()), "elapsed_sec": time.time() - started}
            trace.append(row)
            print(json.dumps(row), flush=True)
        if not finite:
            raise RuntimeError(f"non-finite training in {variant} at step {step}")

    matrix = {}
    for arity in ALL_ARITIES:
        width = args.context
        depth_count = 1
        while width > 1:
            width = math.ceil(width / arity)
            depth_count += 1
        matrix[str(arity)] = [evaluate(model, args, variant, arity, dose) for dose in range(depth_count)]
    torch.save({"variant": variant, "state_dict": model.state_dict(), "config": vars(args)}, output / f"checkpoint_{variant}.pt")
    return {"parameters": sum(p.numel() for p in model.parameters()), "finite": finite,
            "seconds": time.time() - started, "trace": trace, "nll_by_arity_depth": matrix}, model


def compressed_mean(row: Sequence[float]) -> float:
    values = row[:-1] if len(row) > 1 else row
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_decoder_depth_growth_pilot")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--future", type=int, default=32)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--max-arity", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=72063)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    args.vocab = sp.get_piece_size() + 1
    args.bos, args.eos = sp.bos_id(), sp.eos_id()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {}
    models = {}
    for variant in VARIANTS:
        results[variant], models[variant] = train_variant(args, variant, output)
        atomic_json(output / "status.json", {"completed_variants": list(results), "elapsed_sec": time.time() - started})
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    tree = models["tree"]
    interventions = {}
    for arity in ALL_ARITIES:
        native = results["tree"]["nll_by_arity_depth"][str(arity)]
        interventions[str(arity)] = {
            "shuffled_links": [evaluate(tree, args, "tree", arity, d, "shuffled_links") for d in range(len(native))],
            "repeated_root": [evaluate(tree, args, "tree", arity, d, "repeated_root") for d in range(len(native))],
        }

    seen_growth = []
    for arity in TRAIN_ARITIES:
        row = results["tree"]["nll_by_arity_depth"][str(arity)]
        seen_growth.append(row[0] - min(row[:-1]))
    tree_mid = sum(compressed_mean(results["tree"]["nll_by_arity_depth"][str(k)]) for k in TRAIN_ARITIES) / 3
    random_mid = sum(compressed_mean(results["random"]["nll_by_arity_depth"][str(k)]) for k in TRAIN_ARITIES) / 3
    flat_mid = sum(compressed_mean(results["flat"]["nll_by_arity_depth"][str(k)]) for k in TRAIN_ARITIES) / 3
    link_damage = []
    for arity in TRAIN_ARITIES:
        native = results["tree"]["nll_by_arity_depth"][str(arity)]
        shuffled = interventions[str(arity)]["shuffled_links"]
        link_damage.append(compressed_mean(shuffled) - compressed_mean(native))
    unseen_wins = 0
    unseen_rows = {}
    for arity in (3, 5, 6, 7):
        t = compressed_mean(results["tree"]["nll_by_arity_depth"][str(arity)])
        r = compressed_mean(results["random"]["nll_by_arity_depth"][str(arity)])
        f = compressed_mean(results["flat"]["nll_by_arity_depth"][str(arity)])
        advantage = min(r, f) - t
        unseen_rows[str(arity)] = {"tree": t, "random": r, "flat": f, "advantage": advantage}
        unseen_wins += int(advantage >= 0.02)

    sample_source, _ = next(iter(loader(args, "valid", args.seed + 50_000, 16)))
    decoded = [sp.decode(row.tolist()) for row in sample_source]
    character_lengths = sorted(len(row) for row in decoded)
    gates = {
        "P1_depth_growth": sum(seen_growth) / len(seen_growth) >= 0.05,
        "P2_structural_advantage": min(random_mid, flat_mid) - tree_mid >= 0.02,
        "P3_link_causality": sum(link_damage) / len(link_damage) >= 0.02,
        "P4_unseen_arity": unseen_wins >= 2,
        "P5_long_context": character_lengths[len(character_lengths) // 2] >= 100,
        "P6_finite": all(row["finite"] for row in results.values()),
    }
    summary = {
        "claim_id": "S3-DECODER-DEPTH-GROWTH-C01",
        "status": "completed_one_seed_pilot",
        "host": socket.gethostname(),
        "config": vars(args),
        "results": results,
        "interventions": interventions,
        "derived": {
            "seen_root_to_best_compressed_gain": seen_growth,
            "mean_compressed_nll": {"tree": tree_mid, "random": random_mid, "flat": flat_mid},
            "tree_advantage_over_best_control": min(random_mid, flat_mid) - tree_mid,
            "mean_shuffled_link_damage": sum(link_damage) / len(link_damage),
            "unseen_arity": unseen_rows,
            "context_character_lengths": character_lengths,
        },
        "gates": gates,
        "elapsed_sec": time.time() - started,
        "boundary": "One-seed real-text pilot; no semantic, superiority, world-model, or consciousness claim.",
    }
    atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Decoder depth-growth pilot\n\nSee `summary.json` for the complete matrix and gates.\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": summary["derived"], "gates": gates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
