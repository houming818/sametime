#!/usr/bin/env python3
"""Real WMT private-protocol battle for multi-head lifting TreeHeaps."""
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
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
import s2_lifting_pump_wmt as prior
import s2_adaptive_lifting_wmt as adaptive


class HeadReader(nn.Module):
    def __init__(self, dim: int, hidden: int, depths: int):
        super().__init__()
        self.query = nn.Linear(hidden, dim, bias=False)
        self.stop = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        self.branch = nn.Linear(hidden, dim, bias=False)
        self.depth_embedding = nn.Embedding(depths + 1, dim)

    def forward(self, hidden: torch.Tensor, levels, masks):
        query = self.query(hidden)
        active = torch.ones((hidden.shape[0], 1), device=hidden.device)
        context = torch.zeros(
            (hidden.shape[0], levels[0].shape[-1]),
            device=hidden.device,
            dtype=levels[0].dtype,
        )
        masses = []
        for depth, (nodes, valid) in enumerate(zip(levels, masks)):
            active = active * valid.to(active.dtype)
            if depth == len(levels) - 1:
                stop_probability = torch.ones_like(active)
            else:
                depth_state = self.depth_embedding.weight[depth][None, None]
                q = query[:, None].expand_as(nodes)
                stop_probability = torch.sigmoid(
                    self.stop(torch.cat((q, nodes + depth_state), dim=-1)).squeeze(-1)
                )
            stopped = active * stop_probability
            context = context + (stopped[:, :, None] * nodes).sum(1)
            masses.append(stopped.sum(1).mean())
            if depth == len(levels) - 1:
                break
            expand = active * (1.0 - stop_probability)
            children = levels[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2,
            )
            scores = (
                self.branch(hidden)[:, None, None] * children
            ).sum(-1) / math.sqrt(nodes.shape[-1])
            probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
            active = (expand[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return context, torch.stack(masses)


class MultiHeadDecoder(nn.Module):
    def __init__(self, vocab: int, total_dim: int, head_dim: int, hidden: int, depths: int, heads: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, total_dim)
        self.readers = nn.ModuleList(
            HeadReader(head_dim, hidden, depths) for _ in range(heads)
        )
        self.cell = nn.GRUCell(2 * total_dim, hidden)
        self.output = nn.Linear(hidden + total_dim, vocab)
        self.hidden = hidden
        self.total_dim = total_dim

    def read(self, hidden, state_rows, head_ablate: int = -1):
        contexts, masses = [], []
        for index, (reader, (_, _, _, levels, masks)) in enumerate(
            zip(self.readers, state_rows)
        ):
            context, mass = reader(hidden, levels, masks)
            if index == head_ablate:
                context = torch.zeros_like(context)
            contexts.append(context)
            masses.append(mass)
        return torch.cat(contexts, dim=-1), torch.stack(masses)

    def teacher(self, state_rows, target: torch.Tensor, bos: int, head_ablate: int = -1):
        first = state_rows[0][3][0]
        hidden = first.new_zeros((first.shape[0], self.hidden))
        prev = torch.full(
            (first.shape[0],), bos, device=first.device, dtype=torch.long,
        )
        logits, route = [], []
        for step in range(target.shape[1]):
            context, mass = self.read(hidden, state_rows, head_ablate)
            hidden = self.cell(
                torch.cat((self.embedding(prev), context), dim=-1), hidden,
            )
            logits.append(self.output(torch.cat((hidden, context), dim=-1)))
            route.append(mass)
            prev = target[:, step]
        return torch.stack(logits, dim=1), torch.stack(route).mean(0)

    def greedy(self, state_rows, bos: int, eos: int, max_len: int, head_ablate: int = -1):
        first = state_rows[0][3][0]
        hidden = first.new_zeros((first.shape[0], self.hidden))
        prev = torch.full(
            (first.shape[0],), bos, device=first.device, dtype=torch.long,
        )
        done = torch.zeros(first.shape[0], device=first.device, dtype=torch.bool)
        output, route = [], []
        for _ in range(max_len):
            context, mass = self.read(hidden, state_rows, head_ablate)
            hidden = self.cell(
                torch.cat((self.embedding(prev), context), dim=-1), hidden,
            )
            prev = self.output(torch.cat((hidden, context), dim=-1)).argmax(-1)
            output.append(prev)
            route.append(mass)
            done |= prev.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(output, dim=1), torch.stack(route).mean(0)


class MultiHeadTreeHeap(prior.S2Model):
    def __init__(self, vocab: int, total_dim: int, hidden: int, heap_width: int, pad: int, heads: int):
        super().__init__()
        if total_dim % heads:
            raise ValueError("total dimension must be divisible by heads")
        self.head_count = heads
        self.head_dim = total_dim // heads
        self.encoder = nn.ModuleList(
            adaptive.AdaptiveLiftingEncoder(
                vocab, self.head_dim, heap_width, pad, True, False,
            )
            for _ in range(heads)
        )
        depths = self.encoder[0].depths
        self.decoder = MultiHeadDecoder(
            vocab, total_dim, self.head_dim, hidden, depths, heads,
        )
        self.depths = depths

    def states(self, src, length, intervention="native", pair_break_depth=-1):
        return [
            head.states(src, length, intervention, pair_break_depth)
            for head in self.encoder
        ]

    def teacher(
        self, src, length, target, bos, intervention="native",
        pair_break_depth=-1, head_ablate=-1, **kwargs,
    ):
        states = self.states(src, length, intervention, pair_break_depth)
        return self.decoder.teacher(states, target, bos, head_ablate)

    def greedy(
        self, src, length, bos, eos, max_len, intervention="native",
        head_ablate=-1, **kwargs,
    ):
        states = self.states(src, length, intervention)
        return self.decoder.greedy(states, bos, eos, max_len, head_ablate)


def make_model(variant: str, vocab: int, args, pad: int):
    if variant == "flat":
        return prior.FlatModel(vocab, args.dim, args.hidden)
    if variant.startswith("h"):
        heads = int(variant[1:])
        return MultiHeadTreeHeap(
            vocab, args.dim, args.hidden, args.heap_width, pad, heads,
        )
    raise ValueError(variant)


@torch.no_grad()
def evaluate(
    model, loader, args, pad, bos, eos, sp, generate=False,
    intervention="native", pair_break_depth=-1, head_ablate=-1,
):
    model.eval()
    loss_sum = tokens = exact = nonempty = batches = 0
    hypotheses, references, examples = [], [], []
    route_sum = None
    for src, length, target, _ in loader:
        src = src.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        logits, route = model.teacher(
            src, length, target, bos,
            intervention=intervention,
            pair_break_depth=pair_break_depth,
            head_ablate=head_ablate,
        )
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(valid.sum())
        if route is not None:
            row = route.detach().float().cpu()
            route_sum = row if route_sum is None else route_sum + row
        batches += 1
        if generate:
            predicted, _ = model.greedy(
                src, length, bos, eos, target.shape[1],
                intervention=intervention, head_ablate=head_ablate,
            )
            predicted, src_cpu, target_cpu = predicted.cpu(), src.cpu(), target.cpu()
            for index in range(src.shape[0]):
                hyp = base.clean(predicted[index].tolist(), eos, pad)
                ref = base.clean(target_cpu[index].tolist(), eos, pad)
                hypotheses.append(hyp)
                references.append(ref)
                exact += int(hyp == ref)
                nonempty += int(bool(hyp))
                if len(examples) < 12:
                    examples.append({
                        "en": sp.decode(base.clean(src_cpu[index].tolist(), eos, pad)),
                        "reference_zh": sp.decode(ref),
                        "hypothesis_zh": sp.decode(hyp),
                    })
    nll = loss_sum / max(1, tokens)
    result = {"nll": nll, "ppl": math.exp(min(20, nll)), "tokens": tokens}
    if route_sum is not None:
        result["route_depth_mass"] = (route_sum / max(1, batches)).tolist()
    if generate:
        result.update({
            "exact": exact / max(1, len(references)),
            "nonempty": nonempty / max(1, len(references)),
            "token_bleu4": base.bleu4(hypotheses, references),
            "examples": examples,
        })
    return result


@torch.no_grad()
def closure_audit(model, loader, device):
    if not isinstance(model, MultiHeadTreeHeap):
        return {}
    src, length, _, _ = next(iter(loader))
    src, length = src.to(device), length.to(device)
    rows = []
    for leaf, _, _, levels, _ in model.states(src, length):
        diff = levels[-1] - leaf
        rows.append({
            "state_mse": float(diff.square().mean()),
            "state_max_abs": float(diff.abs().max()),
        })
    return {"heads": rows, "max_state_mse": max(row["state_mse"] for row in rows)}


def train_one(variant, seed, loaders, args, vocab, pad, bos, eos, sp, output):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(variant, vocab, args, pad).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_nll, best = float("inf"), None
    trace = []
    gradients_finite = True
    head_gradient_seen = [False] * (model.head_count if isinstance(model, MultiHeadTreeHeap) else 0)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = steps = 0
        for src, length, target, _ in loaders[0]:
            src = src.to(args.device, non_blocking=True)
            length = length.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits, _ = model.teacher(src, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_finite = gradients_finite and prior.finite_gradients(model)
            if isinstance(model, MultiHeadTreeHeap):
                for index, head in enumerate(model.encoder):
                    magnitude = sum(
                        float(parameter.grad.detach().abs().sum())
                        for parameter in head.parameters()
                        if parameter.grad is not None
                    )
                    head_gradient_seen[index] |= magnitude > 1e-10
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        valid = evaluate(model, loaders[1], args, pad, bos, eos, sp)
        row = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_nll": total / max(1, steps),
            "valid_nll": valid["nll"],
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
    if best is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best)
    test = evaluate(model, loaders[2], args, pad, bos, eos, sp, generate=True)
    result = {
        "variant": variant,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
        "finite_gradients": gradients_finite,
        "head_gradient_seen": head_gradient_seen,
        "trace": trace,
        "test": test,
        "closure": closure_audit(model, loaders[2], args.device),
    }
    if variant == "h4":
        checkpoint = output / f"checkpoint_{variant}_seed{seed}.pt"
        torch.save({
            "variant": variant,
            "seed": seed,
            "state_dict": best,
            "config": vars(args),
        }, checkpoint)
        result["checkpoint"] = checkpoint.name
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, best if variant == "h4" else None


def audit_interventions(state, loaders, args, vocab, pad, bos, eos, sp):
    model = make_model("h4", vocab, args, pad).to(args.device)
    model.load_state_dict(state)
    normal = evaluate(model, loaders[2], args, pad, bos, eos, sp)
    rows = {
        "normal": normal,
        "source_shuffle": evaluate(
            model, loaders[2], args, pad, bos, eos, sp,
            intervention="source_shuffle",
        ),
        "root_shuffle": evaluate(
            model, loaders[2], args, pad, bos, eos, sp,
            intervention="root_shuffle",
        ),
        "detail_shuffle": [
            evaluate(
                model, loaders[2], args, pad, bos, eos, sp,
                intervention=f"detail_shuffle_{depth}",
            )
            for depth in range(model.depths)
        ],
        "pair_break": [
            evaluate(
                model, loaders[2], args, pad, bos, eos, sp,
                pair_break_depth=depth,
            )
            for depth in range(model.depths)
        ],
        "head_ablate": [
            evaluate(
                model, loaders[2], args, pad, bos, eos, sp,
                head_ablate=head,
            )
            for head in range(model.head_count)
        ],
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def cross_pair_audit(states, seeds, loaders, args, vocab, pad, bos, eos, sp):
    rows = []
    for encoder_seed in seeds:
        for decoder_seed in seeds:
            if encoder_seed == decoder_seed:
                continue
            encoder_state = states[encoder_seed]
            decoder_state = states[decoder_seed]
            mixed = {
                key: (encoder_state[key] if key.startswith("encoder.") else decoder_state[key])
                for key in encoder_state
            }
            model = make_model("h4", vocab, args, pad).to(args.device)
            model.load_state_dict(mixed)
            result = evaluate(model, loaders[2], args, pad, bos, eos, sp)
            rows.append({
                "encoder_seed": encoder_seed,
                "decoder_seed": decoder_seed,
                "nll": result["nll"],
            })
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    return rows


def summarize(results, interventions, cross_pairs):
    by_variant: Dict[str, List[dict]] = {}
    for row in results:
        by_variant.setdefault(row["variant"], []).append(row)
    aggregate = {}
    for variant, rows in by_variant.items():
        nll = [row["test"]["nll"] for row in rows]
        bleu = [row["test"]["token_bleu4"] for row in rows]
        aggregate[variant] = {
            "nll_mean": mean(nll),
            "nll_min": min(nll),
            "nll_max": max(nll),
            "bleu4_mean": mean(bleu),
            "parameters": rows[0]["parameters"],
            "seconds_mean": mean(row["seconds"] for row in rows),
        }
    normal = interventions["normal"]["nll"]
    damage = {
        "source_shuffle": interventions["source_shuffle"]["nll"] - normal,
        "root_shuffle": interventions["root_shuffle"]["nll"] - normal,
        "detail_shuffle": [row["nll"] - normal for row in interventions["detail_shuffle"]],
        "pair_break": [row["nll"] - normal for row in interventions["pair_break"]],
        "head_ablate": [row["nll"] - normal for row in interventions["head_ablate"]],
    }
    own_h4 = {row["seed"]: row["test"]["nll"] for row in by_variant["h4"]}
    for row in cross_pairs:
        row["damage_vs_decoder_own"] = row["nll"] - own_h4[row["decoder_seed"]]
    cross_damage = [row["damage_vs_decoder_own"] for row in cross_pairs]
    gates = {
        "P1_trainable_all_heads": all(
            row["finite_gradients"]
            and row["test"]["nonempty"] > 0.0
            and (not row["head_gradient_seen"] or all(row["head_gradient_seen"]))
            for row in results
        ),
        "P2_h4_beats_h1": aggregate["h1"]["nll_mean"] - aggregate["h4"]["nll_mean"] >= 0.02,
        "P2_all_heads_help": all(value >= 0.01 for value in damage["head_ablate"]),
        "P3_source_root_causal": damage["source_shuffle"] >= 0.05 and damage["root_shuffle"] >= 0.05,
        "P3_details_pairs_causal": (
            sum(value >= 0.02 for value in damage["detail_shuffle"]) >= 3
            and sum(value >= 0.02 for value in damage["pair_break"]) >= 3
        ),
        "P4_seed_private_or_shared": (
            median(cross_damage) >= 0.10
            or max(abs(value) for value in cross_damage) <= 0.02
        ),
        "P5_h4_beats_flat": aggregate["flat"]["nll_mean"] - aggregate["h4"]["nll_mean"] >= 0.02,
    }
    structural = [
        gates["P1_trainable_all_heads"], gates["P3_source_root_causal"],
        gates["P3_details_pairs_causal"],
    ]
    decision = "supported" if all(gates.values()) else "partial" if all(structural) else "not_supported"
    return aggregate, damage, gates, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_private_protocol_battle")
    parser.add_argument("--seeds", nargs="+", type=int, default=[71901, 71902, 71903])
    parser.add_argument("--variants", nargs="+", default=["flat", "h1", "h2", "h4"])
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=80000)
    parser.add_argument("--valid-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=3000)
    parser.add_argument("--max-scan", type=int, default=600000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    required = {"flat", "h1", "h4"}
    if not required.issubset(args.variants):
        raise ValueError(f"variants must include {sorted(required)}")
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    if any(args.dim % int(name[1:]) for name in args.variants if name.startswith("h")):
        raise ValueError("total dimension must be divisible by every head count")

    random.seed(71900)
    torch.manual_seed(71900)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    sampling_args = copy.copy(args)
    sampling_args.seed = 71900
    rows, sampling = adaptive.load_rows(sampling_args, sp)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    loaders = [
        DataLoader(
            base.ParallelDataset(split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=base.collate(pad),
            pin_memory=args.device.startswith("cuda"),
        )
        for split in splits
    ]
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results, h4_states = [], {}
    for seed in args.seeds:
        for variant in args.variants:
            row, state = train_one(
                variant, seed, loaders, args, vocab, pad, bos, eos, sp, output,
            )
            results.append(row)
            if state is not None:
                h4_states[seed] = state
    audit_seed = args.seeds[0]
    interventions = audit_interventions(
        h4_states[audit_seed], loaders, args, vocab, pad, bos, eos, sp,
    )
    cross_pairs = cross_pair_audit(
        h4_states, args.seeds, loaders, args, vocab, pad, bos, eos, sp,
    )
    aggregate, damage, gates, decision = summarize(
        results, interventions, cross_pairs,
    )
    summary = {
        "claim": "S3-PRIVATE-PROTOCOL-BATTLE-C01",
        "predict": "P-S3-PRIVATE-PROTOCOL-BATTLE-01",
        "status": decision,
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {"direction": "en_to_zh", "sampling": sampling, "vocab": vocab},
        "aggregate": aggregate,
        "intervention_damage_nll": damage,
        "cross_pairs": cross_pairs,
        "gates": gates,
        "boundary": (
            "Stage A tests private pairing, multi-head composition, and TreeHeap causality. "
            "It does not prove semantic heads or Transformer superiority."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "runs.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "interventions.json").write_text(
        json.dumps(interventions, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    trace = [row for result in results for row in result["trace"]]
    (output / "trace.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in trace) + "\n",
        encoding="utf-8",
    )
    readme = {
        "status": decision,
        "aggregate": aggregate,
        "intervention_damage_nll": damage,
        "gates": gates,
    }
    (output / "README.md").write_text(
        "# TreeHeap Private Protocol Battle\n\n```json\n"
        + json.dumps(readme, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
