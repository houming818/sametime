#!/usr/bin/env python3
"""C13 joint multi-round TreeHeap communication protocol smoke."""
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

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_c10_long_smoke as c10
import s3_stone1_c11_source_conditioned as c11
import s3_stone1_c12_hstate_unfold as c12
import s3_wmt_treeheap_seq2seq as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_stone1_c11_source_conditioned/checkpoint_latest.pt")
    p.add_argument("--block-dir", default="/home/nio/datasets/derived/stone1_c10/raw_32k_256")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_stone1_c13_emergent_protocol")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--valid-batches", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.20)
    p.add_argument("--dependence-weight", type=float, default=0.20)
    p.add_argument("--dependence-margin", type=float, default=0.10)
    p.add_argument("--target-depth", type=int, default=7)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--kernel-hidden", type=int, default=768)
    p.add_argument("--residual-scale", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=75033)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def path_code(nodes: int, depth: int, width: int, device, dtype) -> torch.Tensor:
    """Fixed root-to-node left/right code; no learned per-address table."""
    code = torch.zeros(nodes, width, device=device, dtype=dtype)
    if depth == 0:
        return code
    index = torch.arange(nodes, device=device)
    for bit in range(depth):
        shift = depth - bit - 1
        code[:, bit] = ((index >> shift) & 1).to(dtype) * 2.0 - 1.0
    return code


class SubheapRead(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, depths: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.path = nn.Linear(depths, dim, bias=False)
        self.q = nn.ModuleList(nn.Linear(dim, head_dim, bias=False) for _ in range(heads))
        self.k = nn.ModuleList(nn.Linear(dim, head_dim, bias=False) for _ in range(heads))
        self.v = nn.ModuleList(nn.Linear(dim, head_dim, bias=False) for _ in range(heads))
        self.out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, target: torch.Tensor, source: torch.Tensor,
                source_mask: torch.Tensor, depth: int, ablate_head: int = -1):
        target_path = self.path(path_code(
            target.shape[1], depth, self.path.in_features, target.device, target.dtype,
        ))[None]
        source_path = self.path(path_code(
            source.shape[1], depth, self.path.in_features, source.device, source.dtype,
        ))[None]
        target_state, source_state = target + target_path, source + source_path
        values, entropies = [], []
        for head in range(self.heads):
            query = self.q[head](target_state)
            key = self.k[head](source_state)
            value = self.v[head](source_state)
            score = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.head_dim)
            score = score.masked_fill(~source_mask[:, None, :], -1e4)
            weight = score.softmax(-1)
            context = torch.matmul(weight, value)
            if head == ablate_head:
                context = torch.zeros_like(context)
            values.append(context)
            entropy = -(weight.float() * weight.float().clamp_min(1e-9).log()).sum(-1)
            entropies.append(entropy.mean())
        return self.out(torch.cat(values, dim=-1)), torch.stack(entropies)


class EmergentProtocol(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, depths: int,
                 rounds: int, heads: int, head_dim: int, residual_scale: float):
        super().__init__()
        self.depths = depths
        self.rounds = rounds
        self.residual_scale = residual_scale
        self.read = SubheapRead(dim, heads, head_dim, depths)
        self.depth = nn.Embedding(depths, dim)
        self.round = nn.Embedding(rounds, dim)
        self.root_seed = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(5 * dim), nn.Linear(5 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, vocab),
        )

    @staticmethod
    def sibling_swap(nodes: torch.Tensor) -> torch.Tensor:
        if nodes.shape[1] < 2:
            return nodes
        result = nodes.clone()
        result[:, 0::2] = nodes[:, 1::2]
        result[:, 1::2] = nodes[:, 0::2]
        return result

    def target_masks(self, batch: int, device) -> list[torch.Tensor]:
        return [torch.ones(batch, 2 ** (self.depths - depth), dtype=torch.bool,
                           device=device) for depth in range(self.depths + 1)]

    def target_gates(self, batch: int, device, dtype) -> list[torch.Tensor]:
        return [torch.full((batch, 2 ** (self.depths - depth - 1)), 0.4,
                           device=device, dtype=dtype) for depth in range(self.depths)]

    def local_update(self, state, parent, context, depth: int, round_index: int):
        depth_state = self.depth.weight[depth][None, None].expand_as(state)
        round_state = self.round.weight[round_index][None, None].expand_as(state)
        feature = torch.cat((state, parent, context, parent - context,
                             parent * context + depth_state + round_state), dim=-1)
        return state + self.residual_scale * self.update(feature)

    def forward(self, encoder, source_root, source_levels, source_masks,
                intervention: str = "native", active_rounds: int | None = None):
        if intervention == "source_shuffle":
            source_root = source_root.roll(1, 0)
            source_levels = [row.roll(1, 0) for row in source_levels]
            source_masks = [row.roll(1, 0) for row in source_masks]
        elif intervention == "empty_source":
            source_root = torch.zeros_like(source_root)
            source_levels = [torch.zeros_like(row) for row in source_levels]
        elif intervention == "address_swap":
            source_levels = [self.sibling_swap(row) for row in source_levels]
        elif intervention not in {"native"} and not (
            intervention.startswith("head_zero_") or
            intervention.startswith("detail_zero_") or
            intervention == "last_round_zero"
        ):
            raise ValueError(intervention)

        batch, dim = source_root.shape
        root = source_root + self.root_seed(source_root)
        details = [torch.zeros(batch, 2 ** (self.depths - depth - 1), dim,
                               device=root.device, dtype=root.dtype)
                   for depth in range(self.depths)]
        masks = self.target_masks(batch, root.device)
        gates = self.target_gates(batch, root.device, root.dtype)
        entropy_rows = []
        rounds = self.rounds if active_rounds is None else active_rounds
        if intervention == "last_round_zero":
            rounds = max(0, rounds - 1)

        for round_index in range(rounds):
            levels, _ = encoder.unfold(root, details, masks, gates)
            contexts, entropy = [], []
            ablate_head = -1
            if intervention.startswith("head_zero_"):
                ablate_head = int(intervention.rsplit("_", 1)[1])
            for depth in range(self.depths):
                context, local_entropy = self.read(
                    levels[depth], source_levels[depth], source_masks[depth],
                    depth, ablate_head,
                )
                contexts.append(context)
                entropy.append(local_entropy)
            entropy_rows.append(torch.stack(entropy))
            root = self.local_update(root[:, None], levels[0], contexts[0],
                                     0, round_index)[:, 0]
            next_details = list(details)
            for detail_depth in range(self.depths):
                tree_depth = self.depths - detail_depth - 1
                next_details[detail_depth] = self.local_update(
                    details[detail_depth], levels[tree_depth], contexts[tree_depth],
                    tree_depth, round_index,
                )
            details = next_details

        if intervention.startswith("detail_zero_"):
            detail_depth = int(intervention.rsplit("_", 1)[1])
            details[detail_depth] = torch.zeros_like(details[detail_depth])
        levels, _ = encoder.unfold(root, details, masks, gates)
        entropy = torch.stack(entropy_rows) if entropy_rows else root.new_zeros(0)
        return self.output(levels[-1]), levels[-1], details, entropy


def load_model(args, vocab: int, pad: int):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("claim") != "S3-STONE1-SOURCE-CONDITIONED-C11":
        raise ValueError("C13 requires the C11 source-conditioned checkpoint")
    config = argparse.Namespace(**checkpoint["config"])
    model, _ = c10.make_model(config, vocab, pad)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    protocol = EmergentProtocol(
        vocab, config.dim, args.kernel_hidden, args.target_depth, args.rounds,
        args.heads, args.head_dim, args.residual_scale,
    )
    return model.encoder, protocol, config


def encode(encoder, source, length):
    state = encoder.states(source, length)
    return state[0], state[1], state[3], state[4]


def new_batch(stream, count, pad, device, rng):
    return c11.make_batch(stream, count, pad, device, rng, fixed_length=None)


@torch.no_grad()
def evaluate(protocol, encoder, stream, args, pad, rng, intervention="native"):
    protocol.eval(); encoder.eval()
    stream_state = copy.deepcopy(stream.rng.bit_generator.state)
    random_state = rng.getstate()
    total = tokens = 0.0
    variance = entropy_sum = 0.0
    for _ in range(args.valid_batches):
        source, length, target = new_batch(stream, args.batch, pad, args.device, rng)
        _, root, levels, masks = encode(encoder, source, length)
        logits, leaves, _, entropy = protocol(
            encoder, root, levels, masks, intervention=intervention,
        )
        total += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                       ignore_index=pad, reduction="sum"))
        tokens += int(target.ne(pad).sum())
        variance += float(leaves.float().var(dim=1).mean())
        entropy_sum += float(entropy.float().mean()) if entropy.numel() else 0.0
    stream.rng.bit_generator.state = stream_state
    rng.setstate(random_state)
    return {"nll": total / tokens, "tokens": int(tokens),
            "leaf_variance": variance / args.valid_batches,
            "attention_entropy": entropy_sum / args.valid_batches}


@torch.no_grad()
def generation(protocol, encoder, stream, args, sp, pad, eos, rng):
    protocol.eval(); encoder.eval()
    source, length, _ = new_batch(stream, 16, pad, args.device, rng)
    _, root, levels, masks = encode(encoder, source, length)
    logits, _, _, _ = protocol(encoder, root, levels, masks)
    outputs = logits.argmax(-1)
    rows, d2, d4, runs, eos_hits = [], [], [], [], 0
    for src, n, output in zip(source, length, outputs):
        raw = output.cpu().tolist()
        ids = base.clean(raw, eos, pad)
        eos_hits += int(eos in raw)
        d2.append(c12.distinct(ids, 2)); d4.append(c12.distinct(ids, 4))
        runs.append(c12.repeated_run(ids))
        rows.append({"source": sp.decode(src[:int(n)].cpu().tolist()),
                     "output": sp.decode(ids), "distinct2": d2[-1],
                     "distinct4": d4[-1], "max_repeat_run": runs[-1]})
    return {"distinct2": sum(d2) / len(d2), "distinct4": sum(d4) / len(d4),
            "max_repeat_run": max(runs), "eos_fraction": eos_hits / len(rows),
            "unique_output_fraction": len({row["output"] for row in rows}) / len(rows),
            "samples": rows[:8]}


@torch.no_grad()
def closure(encoder, stream, args, pad, rng) -> dict:
    encoder.eval()
    source, length, _ = new_batch(stream, args.batch, pad, args.device, rng)
    leaf, _, levels, _ = encode(encoder, source, length)
    error = (leaf - levels[-1]).float().abs()
    return {"max_abs": float(error.max()), "mse": float(error.square().mean())}


def parameter_delta(before: list[torch.Tensor], parameters) -> float:
    total = 0.0
    for old, new in zip(before, parameters):
        total += float((new.detach().cpu().float() - old).square().sum())
    return math.sqrt(total)


def main() -> None:
    args = parse_args()
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, pad, eos = sp.get_piece_size(), sp.get_piece_size(), sp.eos_id()
    encoder, protocol, config = load_model(args, pieces + 1, pad)
    encoder.to(args.device); protocol.to(args.device)
    encoder_before = [row.detach().cpu().float().clone() for row in encoder.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": protocol.parameters(), "lr": args.lr},
        {"params": encoder.parameters(), "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=1e-4)
    train = c11.AdjacentBlocks(Path(args.block_dir), "train", args.seed)
    valid = c11.AdjacentBlocks(Path(args.block_dir), "valid", args.seed + 9000)
    rng = random.Random(args.seed); valid_seed = args.seed + 8000
    initial = evaluate(protocol, encoder, valid, args, pad, random.Random(valid_seed))
    trace, started = [], time.time()
    for step in range(1, args.steps + 1):
        source, length, target = new_batch(train, args.batch, pad, args.device, rng)
        encoder.train(); protocol.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            _, root, levels, masks = encode(encoder, source, length)
            logits, _, _, _ = protocol(encoder, root, levels, masks)
            native = c11.token_nll(logits, target, pad)
            wrong_source, wrong_length = source.roll(1, 0), length.roll(1, 0)
            _, wrong_root, wrong_levels, wrong_masks = encode(
                encoder, wrong_source, wrong_length,
            )
            wrong_logits, _, _, _ = protocol(
                encoder, wrong_root, wrong_levels, wrong_masks,
            )
            wrong = c11.token_nll(wrong_logits, target, pad)
            dependence = F.relu(args.dependence_margin + native - wrong).mean()
            loss = native.mean() + args.dependence_weight * dependence
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(
            list(protocol.parameters()) + list(encoder.parameters()), 1.0,
        ))
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metric = evaluate(
                protocol, encoder, valid, args, pad, random.Random(valid_seed),
            )
            row = {"step": step, "train_nll": float(native.mean().detach()),
                   "dependence_loss": float(dependence.detach()), "grad_norm": grad,
                   "valid": metric, "elapsed_sec": time.time() - started}
            trace.append(row); print(json.dumps(row), flush=True)

    native = evaluate(protocol, encoder, valid, args, pad, random.Random(valid_seed))
    interventions = {}
    names = ["source_shuffle", "empty_source", "address_swap", "last_round_zero"]
    names += [f"head_zero_{head}" for head in range(args.heads)]
    names += [f"detail_zero_{depth}" for depth in range(args.target_depth)]
    for name in names:
        interventions[name] = evaluate(
            protocol, encoder, valid, args, pad, random.Random(valid_seed), name,
        )
    damage = {name: row["nll"] - native["nll"] for name, row in interventions.items()}
    generated = generation(protocol, encoder, valid, args, sp, pad, eos,
                           random.Random(valid_seed + 1))
    codec = closure(encoder, valid, args, pad, random.Random(valid_seed + 2))
    encoder_delta = parameter_delta(encoder_before, list(encoder.parameters()))
    gates = {
        "learned": math.isfinite(native["nll"]) and initial["nll"] - native["nll"] >= 0.20,
        "source_causal": damage["source_shuffle"] >= 0.05,
        "empty_causal": damage["empty_source"] >= 0.05,
        "address_causal": damage["address_swap"] >= 0.01,
        "two_heads_causal": sum(damage[f"head_zero_{h}"] >= 0.005 for h in range(args.heads)) >= 2,
        "last_round_causal": damage["last_round_zero"] >= 0.01,
        "two_details_causal": sum(damage[f"detail_zero_{d}"] >= 0.005 for d in range(args.target_depth)) >= 2,
        "generation_above_c12": generated["distinct2"] > 0.0079 and generated["distinct4"] > 0.0080 and generated["max_repeat_run"] < 128 and generated["unique_output_fraction"] > 0.25,
        "encoder_changed": encoder_delta > 0.0,
        "closure": codec["max_abs"] < 1e-5,
    }
    summary = {
        "claim": "S3-STONE1-EMERGENT-PROTOCOL-C13", "status": "smoke_complete",
        "host": socket.gethostname(), "config": vars(args),
        "parameters": {"encoder": sum(p.numel() for p in encoder.parameters()),
                       "protocol": sum(p.numel() for p in protocol.parameters())},
        "initial": initial, "final": native, "interventions": interventions,
        "damage": damage, "generation": generated, "closure": codec,
        "encoder_parameter_delta": encoder_delta, "gates": gates,
        "trace": trace, "elapsed_sec": time.time() - started,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"final_nll": native["nll"], "damage": damage,
                      "generation": generated, "closure": codec,
                      "encoder_parameter_delta": encoder_delta, "gates": gates},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
