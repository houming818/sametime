#!/usr/bin/env python3
"""Compare token-stream GRU decoding with algebraic target-H_state UNFOLD."""
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
import s3_stone1_decoder_depth_floor as c06
import s3_wmt_treeheap_seq2seq as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_stone1_c11_source_conditioned/checkpoint_latest.pt")
    p.add_argument("--block-dir", default="/home/nio/datasets/derived/stone1_c10/raw_32k_256")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_stone1_c12_hstate_unfold")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--valid-batches", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-depth", type=int, default=7)
    p.add_argument("--kernel-hidden", type=int, default=1024)
    p.add_argument("--seed", type=int, default=75032)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class HStateUnfoldDecoder(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, target_depth: int):
        super().__init__()
        self.target_depth = target_depth
        self.depth = nn.Embedding(target_depth, dim)
        self.root = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.features = nn.Sequential(
            nn.LayerNorm(5 * dim), nn.Linear(5 * dim, hidden), nn.GELU(),
        )
        self.detail = nn.Linear(hidden, dim)
        self.gate = nn.Linear(hidden, 1)
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

    def forward(self, source_root, source_levels, predictor, updater,
                intervention: str = "native"):
        if intervention in {"source_shuffle", "empty_source"}:
            if intervention == "source_shuffle":
                source_root = source_root.roll(1, 0)
                source_levels = [row.roll(1, 0) for row in source_levels]
            else:
                source_root = torch.zeros_like(source_root)
                source_levels = [torch.zeros_like(row) for row in source_levels]
        node = source_root[:, None] + self.root(source_root)[:, None]
        details = []
        gates = []
        for depth in range(self.target_depth):
            local = source_levels[min(depth, len(source_levels) - 1)]
            if intervention == "address_swap":
                local = self.sibling_swap(local)
            if local.shape[1] != node.shape[1]:
                raise ValueError(f"source/target node mismatch at depth {depth}")
            root = source_root[:, None].expand_as(node)
            depth_state = self.depth.weight[depth][None, None].expand_as(node)
            feature = torch.cat((node, local, root, node - local,
                                 node * local + depth_state), dim=-1)
            shared = self.features(feature)
            detail = self.detail(shared)
            if intervention == f"detail_zero_{depth}":
                detail = torch.zeros_like(detail)
            gate = torch.sigmoid(self.gate(shared))
            anchor = node - updater(detail)
            predicted = detail + predictor(anchor)
            left = gate * anchor + (1.0 - gate) * predicted
            right = gate * predicted + (1.0 - gate) * anchor
            expanded = torch.empty(
                node.shape[0], node.shape[1] * 2, node.shape[2],
                dtype=node.dtype, device=node.device,
            )
            expanded[:, 0::2], expanded[:, 1::2] = left, right
            details.append(detail)
            gates.append(gate.squeeze(-1))
            node = expanded
        return self.output(node), node, details, gates


def load_encoder(args, vocab: int, pad: int):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("claim") != "S3-STONE1-SOURCE-CONDITIONED-C11":
        raise ValueError("C12 requires the C11 source-conditioned checkpoint")
    config = argparse.Namespace(**checkpoint["config"])
    model, floor = c10.make_model(config, vocab, pad)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    return model.encoder, config, floor


def encode(encoder, source, length):
    with torch.no_grad():
        state = encoder.states(source, length)
    return state[1], state[3], state[4]


def distinct(tokens: list[int], n: int) -> float:
    grams = [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


def repeated_run(tokens: list[int], max_n: int = 8) -> int:
    best = 1
    for n in range(1, max_n + 1):
        run = 1
        for start in range(n, len(tokens) - n + 1, n):
            if tokens[start:start + n] == tokens[start - n:start]:
                run += 1
                best = max(best, run)
            else:
                run = 1
    return best


def new_batch(stream, count, pad, device, rng):
    return c11.make_batch(stream, count, pad, device, rng, fixed_length=None)


def gru_forward(decoder, levels, masks, target, bos):
    visible_levels, visible_masks = levels[:-1], masks[:-1]
    return decoder.teacher(visible_levels, visible_masks, target, bos, "depth_floor")[0]


@torch.no_grad()
def evaluate_unfold(decoder, encoder, stream, args, pad, bos, eos, rng,
                    intervention="native"):
    decoder.eval()
    total = tokens = 0.0
    variances = []
    detail_norms = None
    for _ in range(args.valid_batches):
        source, length, target = new_batch(stream, args.batch, pad, args.device, rng)
        root, levels, _ = encode(encoder, source, length)
        logits, leaves, details, _ = decoder(
            root, levels, encoder.predictor, encoder.update_kernel, intervention,
        )
        total += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                       ignore_index=pad, reduction="sum"))
        tokens += int(target.ne(pad).sum())
        variances.append(float(leaves.float().var(dim=1).mean()))
        norms = torch.tensor([float(row.float().norm(dim=-1).mean()) for row in details])
        detail_norms = norms if detail_norms is None else detail_norms + norms
    return {"nll": total / tokens, "tokens": int(tokens),
            "leaf_variance": sum(variances) / len(variances),
            "detail_norm_by_depth": (detail_norms / args.valid_batches).tolist()}


@torch.no_grad()
def evaluate_gru(decoder, encoder, stream, args, pad, bos, rng,
                 intervention="native"):
    decoder.eval()
    total = tokens = 0.0
    for _ in range(args.valid_batches):
        source, length, target = new_batch(stream, args.batch, pad, args.device, rng)
        if intervention == "source_shuffle":
            source, length = source.roll(1, 0), length.roll(1, 0)
        elif intervention == "empty_source":
            source = torch.full_like(source, pad)
            source[:, 0] = 2
            length = torch.ones_like(length)
        root, levels, masks = encode(encoder, source, length)
        logits = gru_forward(decoder, levels, masks, target, bos)
        total += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                       ignore_index=pad, reduction="sum"))
        tokens += int(target.ne(pad).sum())
    return {"nll": total / tokens, "tokens": int(tokens)}


@torch.no_grad()
def generation_metrics(arm, decoder, encoder, stream, args, sp, pad, bos, eos, rng):
    source, length, _ = new_batch(stream, 16, pad, args.device, rng)
    root, levels, masks = encode(encoder, source, length)
    if arm == "unfold":
        logits, _, _, _ = decoder(root, levels, encoder.predictor, encoder.update_kernel)
        outputs = logits.argmax(-1)
    else:
        outputs, _ = decoder.greedy(levels[:-1], masks[:-1], bos, eos, 128, "depth_floor")
    rows, d2, d4, runs, eos_hits = [], [], [], [], 0
    for src, n, output in zip(source, length, outputs):
        ids = base.clean(output.detach().cpu().tolist(), eos, pad)
        raw = output.detach().cpu().tolist()
        eos_hits += int(eos in raw)
        d2.append(distinct(ids, 2)); d4.append(distinct(ids, 4)); runs.append(repeated_run(ids))
        rows.append({"source": sp.decode(src[:int(n)].cpu().tolist()),
                     "output": sp.decode(ids), "distinct2": d2[-1],
                     "distinct4": d4[-1], "max_repeat_run": runs[-1]})
    return {"distinct2": sum(d2) / len(d2), "distinct4": sum(d4) / len(d4),
            "max_repeat_run": max(runs), "eos_fraction": eos_hits / len(rows),
            "unique_output_fraction": len({r["output"] for r in rows}) / len(rows),
            "samples": rows[:8]}


def train_arm(name, decoder, encoder, train, valid, args, pad, bos, eos, sp):
    decoder.to(args.device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed + (0 if name == "unfold" else 100))
    valid_seed = args.seed + 9000 + (0 if name == "unfold" else 100)
    trace = []
    started = time.time()
    initial = (evaluate_unfold if name == "unfold" else evaluate_gru)(
        decoder, encoder, valid, args, pad, bos, *( [eos] if name == "unfold" else []),
        random.Random(valid_seed),
    )
    for step in range(1, args.steps + 1):
        source, length, target = new_batch(train, args.batch, pad, args.device, rng)
        root, levels, masks = encode(encoder, source, length)
        decoder.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            if name == "unfold":
                logits, _, _, _ = decoder(root, levels, encoder.predictor, encoder.update_kernel)
            else:
                logits = gru_forward(decoder, levels, masks, target, bos)
            loss = base.ce(logits, target, pad)
        loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0); optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metric = (evaluate_unfold if name == "unfold" else evaluate_gru)(
                decoder, encoder, valid, args, pad, bos,
                *( [eos] if name == "unfold" else []), random.Random(valid_seed),
            )
            row = {"step": step, "train_nll": float(loss.detach()),
                   "valid": metric, "elapsed_sec": time.time() - started}
            trace.append(row); print(json.dumps({name: row}), flush=True)
    evaluator = evaluate_unfold if name == "unfold" else evaluate_gru
    call = lambda intervention: evaluator(
        decoder, encoder, valid, args, pad, bos,
        *( [eos] if name == "unfold" else []), random.Random(valid_seed), intervention,
    )
    final = call("native")
    interventions = {key: call(key) for key in ("source_shuffle", "empty_source")}
    if name == "unfold":
        interventions["address_swap"] = call("address_swap")
        for depth in range(args.target_depth):
            interventions[f"detail_zero_{depth}"] = call(f"detail_zero_{depth}")
    generated = generation_metrics(name, decoder, encoder, valid, args, sp, pad, bos, eos,
                                   random.Random(valid_seed + 1))
    return {"parameters": sum(p.numel() for p in decoder.parameters()),
            "initial": initial, "final": final, "interventions": interventions,
            "generation": generated, "trace": trace, "elapsed_sec": time.time() - started,
            "state_dict": copy.deepcopy(decoder.state_dict())}


def main() -> None:
    args = parse_args(); output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, pad, bos, eos = sp.get_piece_size(), sp.get_piece_size(), sp.bos_id(), sp.eos_id()
    encoder, config, floor = load_encoder(args, pieces + 1, pad); encoder.to(args.device)
    train_u = c11.AdjacentBlocks(Path(args.block_dir), "train", args.seed)
    valid_u = c11.AdjacentBlocks(Path(args.block_dir), "valid", args.seed + 9000)
    unfold = HStateUnfoldDecoder(pieces + 1, config.dim, args.kernel_hidden, args.target_depth)
    result_u = train_arm("unfold", unfold, encoder, train_u, valid_u, args, pad, bos, eos, sp)
    torch.manual_seed(args.seed + 1); torch.cuda.manual_seed_all(args.seed + 1)
    train_g = c11.AdjacentBlocks(Path(args.block_dir), "train", args.seed)
    valid_g = c11.AdjacentBlocks(Path(args.block_dir), "valid", args.seed + 9000)
    gru = c06.FloorPressureDecoder(pieces + 1, config.dim, config.hidden,
                                   encoder.depths, floor)
    result_g = train_arm("gru", gru, encoder, train_g, valid_g, args, pad, bos, eos, sp)
    for result in (result_u, result_g): result.pop("state_dict")
    native = result_u["final"]["nll"]
    damage = {k: v["nll"] - native for k, v in result_u["interventions"].items()}
    gates = {
        "finite_and_learned": math.isfinite(native) and result_u["initial"]["nll"] - native >= 0.05,
        "source_causal": damage["source_shuffle"] >= 0.05,
        "empty_source_worse": damage["empty_source"] >= 0.05,
        "address_causal": damage["address_swap"] >= 0.01,
        "detail_causal": max(damage[f"detail_zero_{d}"] for d in range(args.target_depth)) >= 0.01,
        "leaves_noncollapsed": result_u["final"]["leaf_variance"] >= 1e-4,
        "distinct2_gain": result_u["generation"]["distinct2"] - result_g["generation"]["distinct2"] >= 0.05,
        "distinct4_gain": result_u["generation"]["distinct4"] - result_g["generation"]["distinct4"] >= 0.05,
        "repeat_run_lower": result_u["generation"]["max_repeat_run"] < result_g["generation"]["max_repeat_run"],
    }
    summary = {"claim": "S3-STONE1-HSTATE-UNFOLD-C12", "status": "smoke_complete",
               "host": socket.gethostname(), "config": vars(args), "unfold": result_u,
               "gru": result_g, "unfold_damage": damage, "gates": gates}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "gates": gates,
                      "unfold_nll": native, "gru_nll": result_g["final"]["nll"],
                      "unfold_generation": result_u["generation"],
                      "gru_generation": result_g["generation"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
