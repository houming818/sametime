#!/usr/bin/env python3
"""Audit one frozen lifting checkpoint by progressively opening tree depth."""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as adaptive
import s3_wmt_treeheap_seq2seq as base


def read_with_cap(decoder, hidden, levels, masks, max_depth: int):
    """Read by recursive probability transport, forcing unresolved mass to stop at the cap."""
    query = decoder.query(hidden)
    active = torch.ones((hidden.shape[0], 1), device=hidden.device)
    context = torch.zeros((hidden.shape[0], levels[0].shape[-1]), device=hidden.device)
    masses: List[torch.Tensor] = []
    for depth, (nodes, valid) in enumerate(zip(levels, masks)):
        active = active * valid.to(active.dtype)
        forced_stop = depth >= max_depth or depth == len(levels) - 1
        if forced_stop:
            stop_probability = torch.ones_like(active)
        else:
            depth_state = decoder.depth_embedding.weight[depth][None, None]
            q = query[:, None].expand_as(nodes)
            stop_probability = torch.sigmoid(
                decoder.stop(torch.cat((q, nodes + depth_state), dim=-1)).squeeze(-1)
            )
        stopped = active * stop_probability
        context = context + (stopped[:, :, None] * nodes).sum(1)
        masses.append(stopped.sum(1))
        if forced_stop:
            break
        expand = active * (1.0 - stop_probability)
        children = levels[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2, nodes.shape[2])
        child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        scores = (
            decoder.branch(hidden)[:, None, None] * children
        ).sum(-1) / math.sqrt(nodes.shape[-1])
        probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
        active = (expand[:, :, None] * probability).reshape(nodes.shape[0], -1)
    return context, torch.stack(masses, dim=1)


def teacher_with_cap(model, src, length, target, bos: int, max_depth: int):
    _, _, _, levels, masks = model.states(src, length)
    decoder = model.decoder
    hidden = levels[0].new_zeros((levels[0].shape[0], decoder.hidden))
    previous = torch.full((levels[0].shape[0],), bos, device=src.device, dtype=torch.long)
    logits, route = [], []
    for step in range(target.shape[1]):
        context, mass = read_with_cap(decoder, hidden, levels, masks, max_depth)
        hidden = decoder.cell(torch.cat((decoder.embedding(previous), context), dim=-1), hidden)
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        route.append(mass)
        previous = target[:, step]
    return torch.stack(logits, dim=1), route


@torch.no_grad()
def evaluate_cap(model, loader, device, pad: int, bos: int, max_depth: int, max_batches: int):
    model.eval()
    loss_sum = token_count = 0
    route_sum = None
    route_rows = 0
    max_mass_error = 0.0
    batches = 0
    for src, length, target, _ in loader:
        if max_batches and batches >= max_batches:
            break
        src, length, target = src.to(device), length.to(device), target.to(device)
        logits, route = teacher_with_cap(model, src, length, target, bos, max_depth)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        token_count += int(target.ne(pad).sum())
        for mass in route:
            error = (mass.sum(1) - 1.0).abs()
            max_mass_error = max(max_mass_error, float(error.max()))
            depth_mass = mass.sum(0).double().cpu()
            if route_sum is None:
                route_sum = torch.zeros(model.encoder.depths + 1, dtype=torch.double)
            route_sum[: depth_mass.numel()] += depth_mass
            route_rows += mass.shape[0]
        batches += 1
    nll = loss_sum / max(1, token_count)
    return {
        "max_depth": max_depth,
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": token_count,
        "batches": batches,
        "route_depth_mass": (route_sum / max(1, route_rows)).tolist(),
        "max_route_mass_error": max_mass_error,
    }


@torch.no_grad()
def greedy_with_cap(model, src, length, bos: int, eos: int, max_len: int, max_depth: int):
    _, _, _, levels, masks = model.states(src, length)
    decoder = model.decoder
    hidden = levels[0].new_zeros((levels[0].shape[0], decoder.hidden))
    previous = torch.full((levels[0].shape[0],), bos, device=src.device, dtype=torch.long)
    done = torch.zeros(src.shape[0], device=src.device, dtype=torch.bool)
    output = []
    for _ in range(max_len):
        context, _ = read_with_cap(decoder, hidden, levels, masks, max_depth)
        hidden = decoder.cell(torch.cat((decoder.embedding(previous), context), dim=-1), hidden)
        previous = decoder.output(torch.cat((hidden, context), dim=-1)).argmax(-1)
        output.append(previous)
        done |= previous.eq(eos)
        if bool(done.all()):
            break
    return torch.stack(output, dim=1)


def decode(sp, ids: Sequence[int], eos: int, pad: int) -> str:
    clean = []
    for token in ids:
        if token in (eos, pad):
            break
        if token >= 0:
            clean.append(int(token))
    return sp.decode(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sp = spm.SentencePieceProcessor(model_file=config["spm_model"])
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    model = adaptive.make_model(
        checkpoint["name"], vocab, config["dim"], config["hidden"],
        config["heap_width"], pad,
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])

    load_args = SimpleNamespace(**config)
    rows, sampling = adaptive.load_rows(load_args, sp)
    offset = config["train_samples"] + config["valid_samples"]
    test_rows = rows[offset : offset + config["test_samples"]]
    loader = DataLoader(
        base.ParallelDataset(test_rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=base.collate(pad),
        pin_memory=args.device.startswith("cuda"),
    )

    depth_results = []
    for cap in range(model.encoder.depths + 1):
        row = evaluate_cap(model, loader, args.device, pad, bos, cap, args.max_batches)
        depth_results.append(row)
        print(json.dumps(row), flush=True)

    src, length, target, _ = next(iter(loader))
    src, length = src[: args.examples].to(args.device), length[: args.examples].to(args.device)
    selected_caps = sorted(set((0, model.encoder.depths // 2, model.encoder.depths)))
    generated = {
        cap: greedy_with_cap(model, src, length, bos, eos, config["max_len"] + 8, cap).cpu()
        for cap in selected_caps
    }
    examples = []
    for index in range(src.shape[0]):
        item = {
            "source": decode(sp, src[index].cpu().tolist(), eos, pad),
            "target": decode(sp, target[index].tolist(), eos, pad),
        }
        for cap in selected_caps:
            item[f"depth_{cap}"] = decode(sp, generated[cap][index].tolist(), eos, pad)
        examples.append(item)

    nlls = [row["nll"] for row in depth_results]
    improvements = [nlls[index - 1] - nlls[index] for index in range(1, len(nlls))]
    gates = {
        "P1_probability_mass_conserved": max(
            row["max_route_mass_error"] for row in depth_results
        ) < 1e-5,
        "P2_full_tree_beats_root_only": nlls[0] - nlls[-1] >= 0.05,
        "P3_depth_has_multiple_positive_steps": sum(value > 0.005 for value in improvements) >= 2,
        "P4_recursive_cap_obeyed": all(
            sum(row["route_depth_mass"][row["max_depth"] + 1 :]) < 1e-8
            for row in depth_results
        ),
    }
    decision = "supported" if all(gates.values()) else "partial" if any(gates.values()) else "not_supported"
    summary = {
        "claim": "S3-TREE-LIFT-RECURSIVE-C01",
        "predict": "P-S3-TREE-LIFT-DEPTH-01",
        "decision": "partial",
        "depth_subclaim_decision": decision,
        "claim_decision": "partial",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": checkpoint["name"],
        "config": config,
        "sampling": sampling,
        "audit": {"max_batches": args.max_batches, "test_rows": len(test_rows)},
        "depth_results": depth_results,
        "derived": {
            "root_to_full_nll_gain": nlls[0] - nlls[-1],
            "adjacent_nll_improvements": improvements,
            "best_depth": min(range(len(nlls)), key=nlls.__getitem__),
        },
        "gates": gates,
        "examples": examples,
        "interpretation": (
            "This frozen en-to-zh audit supports only probability conservation and the "
            "recursive-depth subclaim. Source, address, pairing, and outline-semantics "
            "predictions remain outside this run. It does not train independent models "
            "at different array lengths."
        ),
    }
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
