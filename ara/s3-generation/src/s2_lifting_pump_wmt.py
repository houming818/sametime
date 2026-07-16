#!/usr/bin/env python3
"""Real WMT English->Chinese S2 over a TreeHeap lifting information pump."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base


VARIANTS = ("target_only", "flat_seq", "lifting_root", "lifting_full", "lifting_recursive")


class SharedPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(0.5 * self.net(state))


class LiftingEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int, heap_width: int, pad: int):
        super().__init__()
        if heap_width < 2 or heap_width & (heap_width - 1):
            raise ValueError("heap_width must be a power of two")
        self.embedding = nn.Embedding(vocab, dim)
        self.predictor = SharedPredictor(dim)
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad = pad

    def fold(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        pair_break_depth: int = -1,
    ):
        if src.shape[1] > self.heap_width:
            raise ValueError(f"source width {src.shape[1]} exceeds heap width {self.heap_width}")
        padded = torch.full(
            (src.shape[0], self.heap_width), self.pad,
            dtype=src.dtype, device=src.device,
        )
        padded[:, : src.shape[1]] = src
        leaf_mask = torch.arange(self.heap_width, device=src.device)[None] < length[:, None]
        leaf = self.embedding(padded) * leaf_mask[:, :, None]
        node, node_mask = leaf, leaf_mask
        details: List[torch.Tensor] = []
        masks: List[torch.Tensor] = [leaf_mask]
        depth = 0
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            lm, rm = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == pair_break_depth:
                right = right.roll(1, dims=0)
                rm = rm.roll(1, dims=0)
            detail = right - self.predictor(left)
            parent = left + 0.5 * detail
            node_mask = lm | rm
            parent = parent * node_mask[:, :, None]
            detail = detail * node_mask[:, :, None]
            details.append(detail)
            masks.append(node_mask)
            node = parent
            depth += 1
        return leaf, node[:, 0], details, masks

    def unfold(
        self,
        root: torch.Tensor,
        details: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        intervention: str = "native",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        node = root
        local_details = list(details)
        local_masks = list(masks)
        if intervention == "source_shuffle":
            node = node.roll(1, dims=0)
            local_details = [row.roll(1, dims=0) for row in local_details]
            local_masks = [row.roll(1, dims=0) for row in local_masks]
        elif intervention == "root_shuffle":
            node = node.roll(1, dims=0)
        elif intervention.startswith("detail_shuffle_"):
            depth = int(intervention.rsplit("_", 1)[1])
            local_details[depth] = local_details[depth].roll(1, dims=0)
        elif intervention != "native":
            raise ValueError(intervention)
        levels = [node[:, None]]
        level_masks = [local_masks[-1]]
        for depth in range(len(local_details) - 1, -1, -1):
            detail = local_details[depth]
            left = levels[-1] - 0.5 * detail
            right = detail + self.predictor(left)
            expanded = torch.empty(
                left.shape[0], left.shape[1] * 2, left.shape[2],
                device=left.device, dtype=left.dtype,
            )
            expanded[:, 0::2], expanded[:, 1::2] = left, right
            expanded = expanded * local_masks[depth][:, :, None]
            levels.append(expanded)
            level_masks.append(local_masks[depth])
        return levels, level_masks

    def states(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        intervention: str = "native",
        pair_break_depth: int = -1,
    ):
        leaf, root, details, masks = self.fold(src, length, pair_break_depth)
        levels, level_masks = self.unfold(root, details, masks, intervention)
        return leaf, root, details, levels, level_masks


class RecursiveDecoder(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, depths: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.query = nn.Linear(hidden, dim, bias=False)
        self.stop = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.branch = nn.Linear(hidden, dim, bias=False)
        self.depth_embedding = nn.Embedding(depths + 1, dim)
        self.cell = nn.GRUCell(2 * dim, hidden)
        self.output = nn.Linear(hidden + dim, vocab)
        self.hidden = hidden
        self.depths = depths

    def read(self, hidden: torch.Tensor, levels, masks, route_mode: str = "native"):
        query = self.query(hidden)
        active = torch.ones((hidden.shape[0], 1), device=hidden.device)
        context = torch.zeros((hidden.shape[0], levels[0].shape[-1]), device=hidden.device)
        masses: List[torch.Tensor] = []
        for depth, (nodes, valid) in enumerate(zip(levels, masks)):
            active = active * valid.to(active.dtype)
            last = depth == len(levels) - 1
            if last:
                stop_probability = torch.ones_like(active)
            elif route_mode == "force_root":
                stop_probability = torch.ones_like(active) if depth == 0 else torch.zeros_like(active)
            elif route_mode == "force_leaf":
                stop_probability = torch.zeros_like(active)
            elif route_mode == "native":
                depth_state = self.depth_embedding.weight[depth][None, None]
                q = query[:, None].expand_as(nodes)
                stop_probability = torch.sigmoid(self.stop(torch.cat((q, nodes + depth_state), dim=-1)).squeeze(-1))
            else:
                raise ValueError(route_mode)
            stopped = active * stop_probability
            context = context + (stopped[:, :, None] * nodes).sum(1)
            masses.append(stopped.sum(1).mean())
            if last:
                break
            expand = active * (1.0 - stop_probability)
            children = levels[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2, nodes.shape[2])
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            scores = (self.branch(hidden)[:, None, None] * children).sum(-1) / math.sqrt(nodes.shape[-1])
            probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
            active = (expand[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return context, torch.stack(masses)

    def teacher(self, levels, masks, target: torch.Tensor, bos: int, route_mode: str = "native"):
        hidden = levels[0].new_zeros((levels[0].shape[0], self.hidden))
        prev = torch.full((levels[0].shape[0],), bos, device=levels[0].device, dtype=torch.long)
        logits, route = [], []
        for step in range(target.shape[1]):
            context, mass = self.read(hidden, levels, masks, route_mode)
            hidden = self.cell(torch.cat((self.embedding(prev), context), dim=-1), hidden)
            logits.append(self.output(torch.cat((hidden, context), dim=-1)))
            route.append(mass)
            prev = target[:, step]
        return torch.stack(logits, dim=1), torch.stack(route).mean(0)

    def greedy(self, levels, masks, bos: int, eos: int, max_len: int, route_mode: str = "native"):
        hidden = levels[0].new_zeros((levels[0].shape[0], self.hidden))
        prev = torch.full((levels[0].shape[0],), bos, device=levels[0].device, dtype=torch.long)
        done = torch.zeros(levels[0].shape[0], device=levels[0].device, dtype=torch.bool)
        output, route = [], []
        for _ in range(max_len):
            context, mass = self.read(hidden, levels, masks, route_mode)
            hidden = self.cell(torch.cat((self.embedding(prev), context), dim=-1), hidden)
            prev = self.output(torch.cat((hidden, context), dim=-1)).argmax(-1)
            output.append(prev)
            route.append(mass)
            done |= prev.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(output, dim=1), torch.stack(route).mean(0)


class S2Model(nn.Module):
    def teacher(self, src, length, target, bos, intervention="native", route_mode="native", pair_break_depth=-1):
        raise NotImplementedError

    def greedy(self, src, length, bos, eos, max_len, intervention="native", route_mode="native"):
        raise NotImplementedError


class TargetOnly(S2Model):
    def __init__(self, vocab, dim, hidden):
        super().__init__()
        self.decoder = base.Decoder(vocab, dim, hidden)
        self.anchor = nn.Parameter(torch.zeros(dim))

    def memory(self, src):
        return self.anchor[None, None].expand(src.shape[0], 1, -1), torch.ones((src.shape[0], 1), device=src.device, dtype=torch.bool)

    def teacher(self, src, length, target, bos, **kwargs):
        memory, mask = self.memory(src)
        return self.decoder.teacher(memory, mask, target, bos), None

    def greedy(self, src, length, bos, eos, max_len, **kwargs):
        memory, mask = self.memory(src)
        return self.decoder.greedy(memory, mask, bos, eos, max_len), None


class FlatModel(S2Model):
    def __init__(self, vocab, dim, hidden):
        super().__init__()
        self.model = base.FlatSeq2Seq(vocab, dim, hidden)

    def teacher(self, src, length, target, bos, **kwargs):
        memory, mask = self.model.encode(src, length)
        return self.model.decoder.teacher(memory, mask, target, bos), None

    def greedy(self, src, length, bos, eos, max_len, **kwargs):
        memory, mask = self.model.encode(src, length)
        return self.model.decoder.greedy(memory, mask, bos, eos, max_len), None


class LiftingStatic(S2Model):
    def __init__(self, vocab, dim, hidden, heap_width, pad, variant):
        super().__init__()
        self.encoder = LiftingEncoder(vocab, dim, heap_width, pad)
        self.decoder = base.Decoder(vocab, dim, hidden)
        self.variant = variant

    def memory(self, src, length, intervention="native", pair_break_depth=-1):
        _, _, _, levels, masks = self.encoder.states(src, length, intervention, pair_break_depth)
        return (levels[0], masks[0]) if self.variant == "lifting_root" else (levels[-1], masks[-1])

    def teacher(self, src, length, target, bos, intervention="native", pair_break_depth=-1, **kwargs):
        memory, mask = self.memory(src, length, intervention, pair_break_depth)
        return self.decoder.teacher(memory, mask, target, bos), None

    def greedy(self, src, length, bos, eos, max_len, intervention="native", **kwargs):
        memory, mask = self.memory(src, length, intervention)
        return self.decoder.greedy(memory, mask, bos, eos, max_len), None


class LiftingRecursive(S2Model):
    def __init__(self, vocab, dim, hidden, heap_width, pad):
        super().__init__()
        self.encoder = LiftingEncoder(vocab, dim, heap_width, pad)
        self.decoder = RecursiveDecoder(vocab, dim, hidden, self.encoder.depths)

    def states(self, src, length, intervention="native", pair_break_depth=-1):
        return self.encoder.states(src, length, intervention, pair_break_depth)

    def teacher(self, src, length, target, bos, intervention="native", route_mode="native", pair_break_depth=-1):
        _, _, _, levels, masks = self.states(src, length, intervention, pair_break_depth)
        return self.decoder.teacher(levels, masks, target, bos, route_mode)

    def greedy(self, src, length, bos, eos, max_len, intervention="native", route_mode="native"):
        _, _, _, levels, masks = self.states(src, length, intervention)
        return self.decoder.greedy(levels, masks, bos, eos, max_len, route_mode)


def make_model(name, vocab, dim, hidden, heap_width, pad):
    if name == "target_only":
        return TargetOnly(vocab, dim, hidden)
    if name == "flat_seq":
        return FlatModel(vocab, dim, hidden)
    if name in ("lifting_root", "lifting_full"):
        return LiftingStatic(vocab, dim, hidden, heap_width, pad, name)
    if name == "lifting_recursive":
        return LiftingRecursive(vocab, dim, hidden, heap_width, pad)
    raise ValueError(name)


def finite_gradients(model):
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


@torch.no_grad()
def evaluate(model, loader, cfg, pad, bos, eos, sp, intervention="native", route_mode="native", pair_break_depth=-1, generate=False):
    model.eval()
    loss_sum = tokens = exact = nonempty = 0
    hypotheses, references, examples = [], [], []
    route_sum = None
    batches = 0
    for src, length, target, _ in loader:
        src, length, target = src.to(cfg.device), length.to(cfg.device), target.to(cfg.device)
        logits, route = model.teacher(
            src,
            length,
            target,
            bos,
            intervention=intervention,
            route_mode=route_mode,
            pair_break_depth=pair_break_depth,
        )
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad, reduction="sum"))
        tokens += int(valid.sum())
        if route is not None:
            route_sum = route.detach().float().cpu() if route_sum is None else route_sum + route.detach().float().cpu()
        batches += 1
        if generate:
            predicted, _ = model.greedy(
                src,
                length,
                bos,
                eos,
                target.shape[1],
                intervention=intervention,
                route_mode=route_mode,
            )
            predicted, src_cpu, target_cpu = predicted.cpu(), src.cpu(), target.cpu()
            for row in range(src.shape[0]):
                hyp = base.clean(predicted[row].tolist(), eos, pad)
                ref = base.clean(target_cpu[row].tolist(), eos, pad)
                hypotheses.append(hyp)
                references.append(ref)
                exact += int(hyp == ref)
                nonempty += int(len(hyp) > 0)
                if len(examples) < 12:
                    examples.append({"en": sp.decode(base.clean(src_cpu[row].tolist(), eos, pad)), "reference_zh": sp.decode(ref), "hypothesis_zh": sp.decode(hyp)})
    nll = loss_sum / max(1, tokens)
    result = {"nll": nll, "ppl": math.exp(min(20, nll)), "tokens": tokens}
    if route_sum is not None:
        mass = route_sum / max(1, batches)
        result["route_depth_mass"] = [float(value) for value in mass]
    if generate:
        result.update({"exact": exact / max(1, len(references)), "nonempty": nonempty / max(1, len(references)), "token_bleu4": base.bleu4(hypotheses, references), "examples": examples})
    return result


@torch.no_grad()
def closure_audit(model: LiftingRecursive, loader, device):
    src, length, _, _ = next(iter(loader))
    src, length = src.to(device), length.to(device)
    leaf, _, _, levels, _ = model.states(src, length)
    difference = levels[-1] - leaf
    return {"state_mse": float(difference.square().mean()), "state_max_abs": float(difference.abs().max())}


def train_variant(name, train_loader, valid_loader, test_loader, cfg, vocab, pad, bos, eos, sp, args, output):
    torch.manual_seed(args.seed + sum(map(ord, name)))
    model = make_model(name, vocab, args.dim, args.hidden, args.heap_width, pad).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trace = []
    best_nll = float("inf")
    best = None
    gradients_ok = True
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = steps = 0
        for src, length, target, _ in train_loader:
            src, length, target = src.to(cfg.device), length.to(cfg.device), target.to(cfg.device)
            logits, _ = model.teacher(src, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_ok = gradients_ok and finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        valid = evaluate(model, valid_loader, cfg, pad, bos, eos, sp)
        row = {"model": name, "epoch": epoch, "train_nll": total / max(1, steps), "valid_nll": valid["nll"], "elapsed_sec": time.time() - started}
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best)
    checkpoint = output / f"checkpoint_{name}.pt"
    torch.save({"name": name, "state_dict": best, "config": vars(args)}, checkpoint)
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": gradients_ok,
        "seconds": time.time() - started,
        "trace": trace,
        "test": evaluate(model, test_loader, cfg, pad, bos, eos, sp, generate=True),
        "checkpoint": checkpoint.name,
    }
    if name == "lifting_recursive":
        result["closure"] = closure_audit(model, test_loader, cfg.device)
        result["interventions"] = {
            "source_shuffle": evaluate(model, test_loader, cfg, pad, bos, eos, sp, intervention="source_shuffle"),
            "root_shuffle": evaluate(model, test_loader, cfg, pad, bos, eos, sp, intervention="root_shuffle"),
            "force_root": evaluate(model, test_loader, cfg, pad, bos, eos, sp, route_mode="force_root"),
            "force_leaf": evaluate(model, test_loader, cfg, pad, bos, eos, sp, route_mode="force_leaf"),
            "detail_shuffle": [evaluate(model, test_loader, cfg, pad, bos, eos, sp, intervention=f"detail_shuffle_{depth}") for depth in range(model.encoder.depths)],
            "pair_break": [evaluate(model, test_loader, cfg, pad, bos, eos, sp, pair_break_depth=depth) for depth in range(model.encoder.depths)],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_lifting_pump_wmt")
    parser.add_argument("--seed", type=int, default=71521)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--valid-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--max-scan", type=int, default=100000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--heap-width", type=int, default=32)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    cfg = base.Config(
        data=args.data, spm_model=args.spm_model, evidence_dir=args.evidence_dir,
        model="s2_lifting_pump", seed=args.seed, train_samples=args.train_samples,
        valid_samples=args.valid_samples, test_samples=args.test_samples,
        max_scan=args.max_scan, min_len=args.min_len, max_len=args.max_len,
        dim=args.dim, hidden=args.hidden, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, device=args.device, num_workers=args.num_workers,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, pieces = base.load_rows(cfg, sp)
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    loaders = [
        DataLoader(
            base.ParallelDataset(split), batch_size=args.batch_size,
            shuffle=index == 0, num_workers=args.num_workers,
            collate_fn=base.collate(pad), pin_memory=args.device.startswith("cuda"),
        )
        for index, split in enumerate(splits)
    ]
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: Dict[str, dict] = {}
    for name in args.variants:
        results[name] = train_variant(name, *loaders, cfg, vocab, pad, bos, eos, sp, args, output)
    gates = {}
    derived = {}
    if "lifting_recursive" in results:
        recursive = results["lifting_recursive"]
        native_nll = recursive["test"]["nll"]
        interventions = recursive["interventions"]
        detail_damage = [row["nll"] - native_nll for row in interventions["detail_shuffle"]]
        pair_damage = [row["nll"] - native_nll for row in interventions["pair_break"]]
        route_mass = recursive["test"].get("route_depth_mass", [])
        derived = {
            "recursive_nll": native_nll,
            "recursive_gain_over_root": results.get("lifting_root", {"test": {"nll": float("nan")}})["test"]["nll"] - native_nll,
            "recursive_gap_to_full": native_nll - results.get("lifting_full", {"test": {"nll": float("nan")}})["test"]["nll"],
            "recursive_gap_to_flat": native_nll - results.get("flat_seq", {"test": {"nll": float("nan")}})["test"]["nll"],
            "source_shuffle_damage": interventions["source_shuffle"]["nll"] - native_nll,
            "root_shuffle_damage": interventions["root_shuffle"]["nll"] - native_nll,
            "detail_shuffle_damage": detail_damage,
            "pair_break_damage": pair_damage,
            "force_root_damage": interventions["force_root"]["nll"] - native_nll,
            "force_leaf_damage": interventions["force_leaf"]["nll"] - native_nll,
        }
        gates = {
            "P1_historical_root_exclusive_gain": native_nll <= 6.55,
            "P2_recursive_over_root": derived["recursive_gain_over_root"] >= 0.05,
            "P3_near_full_expand": derived["recursive_gap_to_full"] <= 0.10,
            "P4_source_causal": derived["source_shuffle_damage"] >= 0.20,
            "P5_root_and_detail_causal": derived["root_shuffle_damage"] >= 0.05 and max(detail_damage) >= 0.02,
            "P6_recursive_pairs_causal": sum(value >= 0.02 for value in pair_damage) >= 2,
            "P7_multiresolution_route": sum(value >= 0.05 for value in route_mass) >= 2 and (route_mass[-1] if route_mass else 1.0) <= 0.90,
            "P8_closure_finite_nonempty": recursive["closure"]["state_mse"] < 1e-10 and recursive["finite_gradients"] and recursive["test"]["nonempty"] > 0,
        }
    if gates and all(gates.values()):
        decision = "supported_full" if args.train_samples >= 20000 else "supported_pilot" if args.train_samples >= 5000 else "supported_smoke"
    else:
        decision = "partial" if gates and any(gates.values()) else "not_supported"
    summary = {
        "claim": "S2-LIFT-WMT-C01",
        "predict": "P-S2-LIFT-WMT-01",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {"direction": "en_to_zh", "rows": len(rows), "vocab": vocab},
        "models": results,
        "derived": derived,
        "gates": gates,
        "decision": decision,
        "boundary": "Real WMT S2 mechanism proof; not production translation, compression, semantic labels, world knowledge, or consciousness.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for result in results.values() for row in result["trace"]) + "\n", encoding="utf-8")
    (output / "examples.json").write_text(json.dumps({name: result["test"].get("examples", []) for name, result in results.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text("# S2 Lifting Pump WMT\n\n```json\n" + json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
