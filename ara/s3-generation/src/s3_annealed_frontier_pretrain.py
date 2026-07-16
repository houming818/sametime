#!/usr/bin/env python3
"""Complete-context future generation under annealed TreeHeap frontiers."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as lifting
import s3_wmt_treeheap_seq2seq as base
from s3_conditional_denoising_seq2seq import MixedDocuments


VARIANTS = ("leaf_only", "uniform", "annealed")


class FutureBlocks(IterableDataset):
    def __init__(self, root: Path, spm_path: str, split: str, seed: int, context: int, future: int):
        self.root, self.spm_path, self.split, self.seed = root, spm_path, split, seed
        self.context, self.future = context, future

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        sp = spm.SentencePieceProcessor(model_file=self.spm_path)
        width = self.context + self.future
        for text in MixedDocuments(self.root, self.split, self.seed):
            ids = sp.encode(text, out_type=int)
            for start in range(0, len(ids) - width + 1, width):
                block = ids[start : start + width]
                yield (
                    torch.tensor(block[: self.context], dtype=torch.long),
                    torch.tensor(block[self.context :] + [sp.eos_id()], dtype=torch.long),
                )


def collate(batch):
    return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])


class FrontierModel(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, heap_width: int, pad: int):
        super().__init__()
        self.encoder = lifting.AdaptiveLiftingEncoder(vocab, dim, heap_width, pad, True, False)
        self.decoder = base.Decoder(vocab, dim, hidden)
        self.resolution = nn.Embedding(self.encoder.depths + 1, dim)
        nn.init.zeros_(self.resolution.weight)

    @property
    def depths(self) -> int:
        return self.encoder.depths

    def frontier(self, source: torch.Tensor, depth: int, intervention: str = "native"):
        length = torch.full((source.shape[0],), source.shape[1], device=source.device, dtype=torch.long)
        working = source
        if intervention == "sibling_swap":
            working = source.reshape(source.shape[0], -1, 2).flip(2).reshape_as(source)
        _, _, _, levels, masks = self.encoder.states(working, length)
        nodes, valid = levels[depth], masks[depth]
        if intervention == "source_shuffle":
            nodes, valid = nodes.roll(1, dims=0), valid.roll(1, dims=0)
        elif intervention == "root_zero":
            nodes = torch.zeros_like(nodes)
        elif intervention not in ("native", "sibling_swap"):
            raise ValueError(intervention)
        return nodes + self.resolution.weight[depth], valid

    def teacher(self, source: torch.Tensor, target: torch.Tensor, bos: int, depth: int, intervention: str = "native"):
        nodes, valid = self.frontier(source, depth, intervention)
        return self.decoder.teacher(nodes, valid, target, bos), None

    def greedy(self, source: torch.Tensor, bos: int, eos: int, max_len: int, depth: int, intervention: str = "native"):
        nodes, valid = self.frontier(source, depth, intervention)
        return self.decoder.greedy(nodes, valid, bos, eos, max_len), None


def clean(ids: Sequence[int], eos: int, pad: int) -> List[int]:
    answer = []
    for token in ids:
        if int(token) in (eos, pad): break
        answer.append(int(token))
    return answer


def repeat_fraction(ids: Sequence[int]) -> float:
    if len(ids) < 2: return 0.0
    return sum(ids[i] == ids[i - 1] for i in range(1, len(ids))) / (len(ids) - 1)


@torch.no_grad()
def evaluate(model: FrontierModel, loader, args, sp, depth: int, intervention: str = "native", generate: bool = False) -> dict:
    model.eval()
    loss_sum = tokens = greedy_hit = greedy_tokens = exact = samples = 0
    outputs: List[Tuple[int, ...]] = []
    repeats = 0.0
    examples: List[dict] = []
    for batch_no, (source, target) in enumerate(loader, 1):
        source, target = source.to(args.device), target.to(args.device)
        logits, _ = model.teacher(source, target, args.bos, depth, intervention)
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
        tokens += target.numel()
        if generate:
            predicted, _ = model.greedy(source, args.bos, args.eos, target.shape[1], depth, intervention)
            if predicted.shape[1] < target.shape[1]:
                predicted = F.pad(predicted, (0, target.shape[1] - predicted.shape[1]), value=args.pad)
            predicted = predicted[:, : target.shape[1]]
            greedy_hit += int(predicted.eq(target).sum())
            greedy_tokens += target.numel()
            for row in range(source.shape[0]):
                reference = clean(target[row].cpu().tolist(), args.eos, args.pad)
                output = clean(predicted[row].cpu().tolist(), args.eos, args.pad)
                outputs.append(tuple(output)); repeats += repeat_fraction(output)
                exact += int(output == reference); samples += 1
                if len(examples) < 8:
                    examples.append({"context": sp.decode(source[row].cpu().tolist()), "reference": sp.decode(reference), "generated": sp.decode(output)})
        if batch_no >= args.eval_batches: break
    nll = loss_sum / max(1, tokens)
    return {
        "nll": nll, "ppl": math.exp(min(20, nll)),
        "greedy_token_accuracy": greedy_hit / max(1, greedy_tokens) if generate else None,
        "greedy_exact": exact / max(1, samples) if generate else None,
        "nonempty_fraction": sum(bool(row) for row in outputs) / max(1, samples) if generate else None,
        "adjacent_repeat_fraction": repeats / max(1, samples) if generate else None,
        "unique_output_fraction": len(set(outputs)) / max(1, samples) if generate else None,
        "examples": examples,
    }


def make_loader(args, split: str, seed: int):
    return DataLoader(
        FutureBlocks(Path(args.root), args.spm_model, split, seed, args.context, args.future),
        batch_size=args.batch, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def annealed_depth(step: int, total: int, depths: int, rng: random.Random) -> int:
    progress = step / max(1, total - 1)
    direction = 1.0 - 2.0 * progress
    temperature = 1.5 * (0.25 / 1.5) ** progress
    logits = [direction * depth / temperature for depth in range(depths)]
    maximum = max(logits)
    soft = [math.exp(value - maximum) for value in logits]
    soft = [value / sum(soft) for value in soft]
    probability = [0.25 / depths + 0.75 * value for value in soft]
    return rng.choices(range(depths), weights=probability)[0]


def choose_depth(name: str, step: int, args, depths: int, rng: random.Random) -> int:
    if name == "leaf_only": return depths - 1
    if name == "uniform": return rng.randrange(depths)
    return annealed_depth(step, args.steps, depths, rng)


def train_variant(name: str, args, sp, output: Path) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model = FrontierModel(args.vocab, args.dim, args.hidden, args.heap_width, args.pad).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = iter(make_loader(args, "train", args.seed))
    valid_loaders = [make_loader(args, "valid", args.seed + 1000 + depth) for depth in range(model.depths + 1)]
    initial = [evaluate(model, loader, args, sp, depth)["nll"] for depth, loader in enumerate(valid_loaders)]
    rng = random.Random(args.seed + 700 + sum(map(ord, name)))
    trace = []; best_score = float("inf"); best = None; finite = True; depth_count = [0] * (model.depths + 1)
    started = time.time()
    for step in range(args.steps):
        model.train(); source, target = next(train_iter); source, target = source.to(args.device), target.to(args.device)
        depth = choose_depth(name, step, args, model.depths + 1, rng); depth_count[depth] += 1
        logits, _ = model.teacher(source, target, args.bos, depth)
        loss = base.ce(logits, target, args.pad)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        finite = finite and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % args.valid_every == 0 or step + 1 == args.steps:
            valid = [evaluate(model, loader, args, sp, d)["nll"] for d, loader in enumerate(valid_loaders)]
            if name == "leaf_only": score = valid[-1]
            elif name == "annealed": score = valid[0]
            else: score = sum(valid) / len(valid)
            row = {"variant": name, "step": step + 1, "train_nll": float(loss.detach()), "valid_nll_root_to_leaf": valid, "depth_count": list(depth_count), "elapsed_sec": time.time() - started}
            trace.append(row); print(json.dumps(row), flush=True)
            if score < best_score:
                best_score = score
                best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(best)
    test_loaders = [make_loader(args, "test", args.seed + 2000 + depth) for depth in range(model.depths + 1)]
    frontiers = [evaluate(model, loader, args, sp, depth, generate=True) for depth, loader in enumerate(test_loaders)]
    audit_loader = make_loader(args, "test", args.seed + 3000)
    native = evaluate(model, audit_loader, args, sp, 0)["nll"]
    interventions = {}
    for intervention in ("source_shuffle", "root_zero", "sibling_swap"):
        row = evaluate(model, audit_loader, args, sp, 0, intervention)
        interventions[intervention] = {"nll": row["nll"], "damage": row["nll"] - native}
    length = torch.full((args.batch,), args.context, device=args.device, dtype=torch.long)
    source, _ = next(iter(audit_loader)); source = source.to(args.device)
    length = length[: source.shape[0]]
    leaf, _, _, levels, _ = model.encoder.states(source, length)
    difference = levels[-1] - leaf
    closure = {"state_mse": float(difference.square().mean().detach()), "state_max_abs": float(difference.abs().max().detach())}
    checkpoint = output / f"checkpoint_{name}.pt"
    torch.save({"name": name, "state_dict": best, "config": vars(args), "trace": trace}, checkpoint)
    return {
        "parameters": sum(p.numel() for p in model.parameters()), "finite_gradients": finite,
        "seconds": time.time() - started, "initial_nll_root_to_leaf": initial,
        "trace": trace, "test_root_to_leaf": frontiers, "interventions_root": interventions,
        "closure": closure, "depth_count": depth_count, "checkpoint": checkpoint.name,
    }


def spearman_depth(values: Sequence[float]) -> float:
    order = sorted(range(len(values)), key=lambda index: values[index])
    rank = [0] * len(values)
    for value, index in enumerate(order): rank[index] = value
    x = list(range(len(values))); mean_x = sum(x) / len(x); mean_y = sum(rank) / len(rank)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, rank))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in rank))
    return numerator / max(1e-12, denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_annealed_frontier_pretrain")
    parser.add_argument("--seed", type=int, default=72051)
    parser.add_argument("--context", type=int, default=64); parser.add_argument("--future", type=int, default=16); parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256); parser.add_argument("--hidden", type=int, default=256); parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000); parser.add_argument("--valid-every", type=int, default=2000); parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.context != args.heap_width: raise ValueError("context must equal heap width")
    sp = spm.SentencePieceProcessor(model_file=args.spm_model); args.pad = sp.get_piece_size(); args.vocab = args.pad + 1; args.bos = sp.bos_id(); args.eos = sp.eos_id()
    random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True); started = time.time(); results = {}
    for name in args.variants:
        results[name] = train_variant(name, args, sp, output)
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
    annealed = results["annealed"]; uniform = results["uniform"]
    nll = [row["nll"] for row in annealed["test_root_to_leaf"]]
    root = annealed["test_root_to_leaf"][0]; leaf = annealed["test_root_to_leaf"][-1]
    derived = {
        "annealed_root_gain_over_uniform": uniform["test_root_to_leaf"][0]["nll"] - root["nll"],
        "annealed_root_gap_to_leaf": root["nll"] - leaf["nll"],
        "root_source_shuffle_damage": annealed["interventions_root"]["source_shuffle"]["damage"],
        "root_sibling_swap_damage": annealed["interventions_root"]["sibling_swap"]["damage"],
        "frontier_nll_root_to_leaf": nll,
        "frontier_nll_spearman": spearman_depth(nll),
    }
    gates = {
        "R1_finite_and_learning": all(row["finite_gradients"] and min(x["nll"] for x in row["test_root_to_leaf"]) < max(row["initial_nll_root_to_leaf"]) for row in results.values()),
        "R2_annealed_root_gain": derived["annealed_root_gain_over_uniform"] >= 0.05,
        "R3_root_retains_future": derived["annealed_root_gap_to_leaf"] <= 1.0,
        "R4_root_source_causal": derived["root_source_shuffle_damage"] >= 0.20,
        "R5_address_causal": derived["root_sibling_swap_damage"] >= 0.05,
        "R6_root_generation_nondegenerate": root["nonempty_fraction"] >= 0.75 and root["adjacent_repeat_fraction"] <= 0.40 and root["unique_output_fraction"] >= 0.10,
        "R7_ordered_rate_distortion": derived["frontier_nll_spearman"] <= -0.70,
        "R8_closed": annealed["closure"]["state_mse"] < 1e-10,
    }
    summary = {"claim": "S3-ANNEAL-REAL-C01", "host": socket.gethostname(), "seconds": time.time() - started, "config": vars(args), "results": results, "derived": derived, "gates": gates, "decision": "supported" if all(gates.values()) else "partial or rejected; inspect gates"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text("# Annealed Frontier Pretraining\n\nComplete 64-token context predicts the next 16 tokens; no MASK.\n", encoding="utf-8")
    print(json.dumps({"derived": derived, "gates": gates}, indent=2), flush=True)


if __name__ == "__main__": main()
