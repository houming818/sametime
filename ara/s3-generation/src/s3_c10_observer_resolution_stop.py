#!/usr/bin/env python3
"""Frozen C10 smoke for observer-resolution termination in recursive READ."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


def resolution_read(decoder, hidden, levels, masks, epsilon: float):
    query = decoder.query(hidden)
    active = torch.ones((hidden.shape[0], 1), device=hidden.device)
    context = torch.zeros(
        (hidden.shape[0], levels[0].shape[-1]), device=hidden.device,
    )
    route = []
    forced = []
    visited = []
    for depth, (nodes, valid) in enumerate(zip(levels, masks)):
        active = active * valid.to(active.dtype)
        visited.append((active > 0).sum(-1).to(active.dtype))
        last = depth == len(levels) - 1
        if last:
            forced_here = torch.zeros_like(active, dtype=torch.bool)
            stop_probability = torch.ones_like(active)
        else:
            forced_here = active.gt(0) & active.le(epsilon)
            depth_state = decoder.depth_embedding.weight[depth][None, None]
            q = query[:, None].expand_as(nodes)
            learned = torch.sigmoid(
                decoder.stop(torch.cat((q, nodes + depth_state), dim=-1)).squeeze(-1)
            )
            stop_probability = torch.where(
                forced_here, torch.ones_like(learned), learned,
            )
        stopped = active * stop_probability
        forced_mass = active * forced_here.to(active.dtype)
        context = context + (stopped[:, :, None] * nodes).sum(1)
        route.append(stopped.sum(1))
        forced.append(forced_mass.sum(1))
        if last:
            break
        expand = active * (1.0 - stop_probability)
        children = levels[depth + 1].reshape(
            nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
        )
        child_valid = masks[depth + 1].reshape(
            nodes.shape[0], nodes.shape[1], 2,
        )
        scores = (
            decoder.branch(hidden)[:, None, None] * children
        ).sum(-1) / math.sqrt(nodes.shape[-1])
        probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
        active = (expand[:, :, None] * probability).reshape(hidden.shape[0], -1)
    return (
        context,
        torch.stack(route, dim=1),
        torch.stack(forced, dim=1),
        torch.stack(visited, dim=1),
    )


def teacher(model, source, length, target, bos, epsilon: float):
    _, _, _, levels, masks = model.encoder.states(source, length)
    decoder = model.decoder
    hidden = levels[0].new_zeros((source.shape[0], decoder.hidden))
    previous = torch.full(
        (source.shape[0],), bos, device=source.device, dtype=torch.long,
    )
    logits = []
    route_sum = forced_sum = visited_sum = None
    for step in range(target.shape[1]):
        context, route, forced, visited = resolution_read(
            decoder, hidden, levels, masks, epsilon,
        )
        hidden = decoder.cell(
            torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
        )
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        route_sum = route if route_sum is None else route_sum + route
        forced_sum = forced if forced_sum is None else forced_sum + forced
        visited_sum = visited if visited_sum is None else visited_sum + visited
        previous = target[:, step]
    steps = max(1, target.shape[1])
    return (
        torch.stack(logits, dim=1),
        route_sum / steps,
        forced_sum / steps,
        visited_sum / steps,
    )


def align_depth(value, depths):
    if value.shape[1] < depths:
        value = F.pad(value, (depths - value.shape[1], 0))
    return value


@torch.no_grad()
def evaluate(model, rows, pad, bos, device, batch_size, epsilon: float):
    model.eval()
    depths = model.decoder.depths + 1
    loss_sum = tokens = examples = 0
    # Keep independent accumulators. Chained assignment would alias one Tensor
    # and make route, forced mass, and visited-node counts contaminate each other.
    route_sum = torch.zeros(depths, dtype=torch.float64)
    forced_sum = torch.zeros(depths, dtype=torch.float64)
    visited_sum = torch.zeros(depths, dtype=torch.float64)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        source, length, target = c10.collate_rows(batch, pad, device)
        logits, route, forced, visited = teacher(
            model, source, length, target, bos, epsilon,
        )
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
        weight = len(batch)
        route_sum += align_depth(route.detach().double().cpu(), depths).sum(0)
        forced_sum += align_depth(forced.detach().double().cpu(), depths).sum(0)
        visited_sum += align_depth(visited.detach().double().cpu(), depths).sum(0)
        examples += weight
    route = route_sum / max(1, examples)
    forced = forced_sum / max(1, examples)
    visited = visited_sum / max(1, examples)
    mean_depth = sum(depth * float(mass) for depth, mass in enumerate(route))
    return {
        "epsilon": epsilon,
        "nll": loss_sum / max(1, tokens),
        "tokens": tokens,
        "route_depth_mass": [float(value) for value in route],
        "forced_resolution_mass": [float(value) for value in forced],
        "total_forced_resolution_mass": float(forced.sum()),
        "mean_stop_depth": mean_depth,
        "visited_nodes_by_depth": [float(value) for value in visited],
        "mean_visited_nodes": float(visited.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epsilons", default="0,1e-6,1e-4,1e-3,3e-3,1e-2,2e-2,5e-2,1e-1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = argparse.Namespace(**checkpoint["config"])
    config.device, config.wmt_data, config.spm_model = args.device, args.wmt_data, args.spm_model
    config.task_train_rows, config.task_eval_rows = 1, args.rows
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    directions = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    model = c10.build_model(config, pieces + 3, pad).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _, _, rows = c10.collect_wmt_rows(config, sp, directions, eos)

    results = []
    for epsilon in (float(value) for value in args.epsilons.split(",")):
        row = evaluate(
            model, rows, pad, bos, args.device, args.batch_size, epsilon,
        )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    baseline = results[0]
    for row in results:
        row["delta_nll"] = row["nll"] - baseline["nll"]
        row["compute_reduction"] = 1.0 - (
            row["mean_visited_nodes"] / baseline["mean_visited_nodes"]
        )
    candidates = [
        row for row in results
        if row["epsilon"] <= 1e-2
        and row["delta_nll"] <= 0.02
        and row["compute_reduction"] >= 0.20
    ]
    report = {
        "diagnostic": "S3-C10-OBSERVER-RESOLUTION-STOP-D02",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "state_sha256": checkpoint.get("state_sha256"),
        "rows": len(rows),
        "results": results,
        "numerical_tail_gate_passed": bool(candidates),
        "small_epsilon_candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
