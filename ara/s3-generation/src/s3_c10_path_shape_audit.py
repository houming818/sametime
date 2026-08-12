#!/usr/bin/env python3
"""Audit whether C10 uses a distributed leaf index or a narrow path cursor."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


def leaf_distribution(decoder, hidden, levels, masks, uniform_depth: int = -1):
    active = torch.ones((hidden.shape[0], 1), device=hidden.device)
    depth_entropies = []
    for depth in range(len(levels) - 1):
        nodes = levels[depth]
        children = levels[depth + 1].reshape(
            nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
        )
        valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        if depth == uniform_depth:
            probability = valid.to(children.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1.0)
        else:
            scores = (
                decoder.branch(hidden)[:, None, None] * children
            ).sum(-1) / math.sqrt(nodes.shape[-1])
            probability = F.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
        local_entropy = -(probability.clamp_min(1e-30).log() * probability).sum(-1)
        parent_weight = active / active.sum(-1, keepdim=True).clamp_min(1e-30)
        depth_entropies.append((parent_weight * local_entropy).sum(-1))
        active = (active[:, :, None] * probability).reshape(hidden.shape[0], -1)
    active = active * masks[-1].to(active.dtype)
    active = active / active.sum(-1, keepdim=True).clamp_min(1e-30)
    return active, depth_entropies


def select_context(mode, step, probability, leaves, length):
    batch, width = probability.shape
    if mode == "all_leaf" or mode.startswith("uniform_depth_"):
        selected = probability
    elif mode.startswith("top"):
        k = min(int(mode[3:]), width)
        _, index = probability.topk(k, dim=-1)
        selected = torch.zeros_like(probability).scatter(1, index, probability.gather(1, index))
        selected = selected / selected.sum(-1, keepdim=True).clamp_min(1e-30)
    elif mode == "drop_top1":
        index = probability.argmax(-1, keepdim=True)
        selected = probability.scatter(1, index, 0.0)
        selected = selected / selected.sum(-1, keepdim=True).clamp_min(1e-30)
    elif mode == "uniform_leaf":
        valid = torch.arange(width, device=length.device)[None] < length[:, None]
        selected = valid.to(leaves.dtype)
        selected = selected / selected.sum(-1, keepdim=True).clamp_min(1.0)
    elif mode == "ordered_cursor":
        index = torch.minimum(
            torch.full_like(length, step), length.sub(1).clamp_min(0),
        )[:, None]
        selected = torch.zeros_like(probability).scatter(1, index, 1.0)
    else:
        raise ValueError(mode)
    return (selected[:, :, None] * leaves).sum(1)


def custom_teacher(model, source, length, target, bos, mode, collect=False):
    _, _, _, levels, masks = model.encoder.states(source, length)
    decoder = model.decoder
    hidden = levels[0].new_zeros((source.shape[0], decoder.hidden))
    prev = torch.full((source.shape[0],), bos, device=source.device, dtype=torch.long)
    logits = []
    observations = []
    uniform_depth = int(mode.rsplit("_", 1)[1]) if mode.startswith("uniform_depth_") else -1
    for step in range(target.shape[1]):
        probability, depth_entropies = leaf_distribution(
            decoder, hidden, levels, masks, uniform_depth,
        )
        context = select_context(mode, step, probability, levels[-1], length)
        hidden = decoder.cell(
            torch.cat((decoder.embedding(prev), context), dim=-1), hidden,
        )
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        if collect:
            observations.append({
                "probability": probability.detach().cpu(),
                "depth_entropies": [row.detach().cpu() for row in depth_entropies],
            })
        prev = target[:, step]
    return torch.stack(logits, dim=1), observations


@torch.no_grad()
def evaluate(model, rows, pad, bos, device, batch_size, mode, collect=False):
    model.eval()
    loss_sum = token_count = 0
    concentration = defaultdict(float)
    valid_steps = transitions = 0
    stationary = adjacent = monotonic = abs_jump = unique_sum = 0.0
    depth_entropy_sum = defaultdict(float)
    if mode == "native_model":
        for start in range(0, len(rows), batch_size):
            source, length, target = c10.collate_rows(rows[start:start + batch_size], pad, device)
            logits, _ = model.teacher(source, length, target, bos)
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                ignore_index=pad, reduction="sum",
            ))
            token_count += int(target.ne(pad).sum())
        return {"nll": loss_sum / max(1, token_count), "tokens": token_count}

    for start in range(0, len(rows), batch_size):
        source, length, target = c10.collate_rows(rows[start:start + batch_size], pad, device)
        logits, observations = custom_teacher(model, source, length, target, bos, mode, collect)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        valid = target.ne(pad)
        token_count += int(valid.sum())
        if not collect:
            continue
        paths = [[] for _ in range(source.shape[0])]
        for step, observation in enumerate(observations):
            probability = observation["probability"]
            step_valid = valid[:, step].cpu()
            if not bool(step_valid.any()):
                continue
            local = probability[step_valid]
            for k in (1, 2, 4, 8):
                concentration[f"top{k}_mass"] += float(
                    local.topk(min(k, local.shape[-1]), dim=-1).values.sum(-1).sum()
                )
            entropy = -(local.clamp_min(1e-30).log() * local).sum(-1)
            concentration["entropy"] += float(entropy.sum())
            concentration["effective_leaves"] += float(entropy.exp().sum())
            valid_steps += int(step_valid.sum())
            for depth, values in enumerate(observation["depth_entropies"]):
                depth_entropy_sum[depth] += float(values[step_valid].sum())
            argmax = probability.argmax(-1)
            for row in range(source.shape[0]):
                if bool(step_valid[row]):
                    paths[row].append(int(argmax[row]))
        for path in paths:
            if not path:
                continue
            unique_sum += len(set(path))
            for left, right in zip(path, path[1:]):
                transitions += 1
                stationary += left == right
                adjacent += right == left + 1
                monotonic += right >= left
                abs_jump += abs(right - left)

    result = {"nll": loss_sum / max(1, token_count), "tokens": token_count}
    if collect:
        result["leaf_distribution"] = {
            key: value / max(1, valid_steps) for key, value in concentration.items()
        }
        result["leaf_distribution"]["depth_branch_entropy"] = {
            str(depth): value / max(1, valid_steps)
            for depth, value in sorted(depth_entropy_sum.items())
        }
        result["address_trajectory"] = {
            "mean_unique_leaves_per_row": unique_sum / max(1, len(rows)),
            "stationary_rate": stationary / max(1, transitions),
            "adjacent_forward_rate": adjacent / max(1, transitions),
            "stationary_plus_adjacent_rate": (stationary + adjacent) / max(1, transitions),
            "monotonic_forward_rate": monotonic / max(1, transitions),
            "mean_absolute_jump": abs_jump / max(1, transitions),
            "transitions": transitions,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
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

    modes = ["native_model", "all_leaf", "top1", "top2", "top4", "top8", "drop_top1", "uniform_leaf", "ordered_cursor"]
    modes += [f"uniform_depth_{depth}" for depth in range(model.decoder.depths)]
    results = {}
    for mode in modes:
        result = evaluate(
            model, rows, pad, bos, args.device, args.batch_size,
            mode, collect=mode == "all_leaf",
        )
        results[mode] = result
        print(json.dumps({"mode": mode, **result}, ensure_ascii=False), flush=True)

    native = results["native_model"]["nll"]
    for result in results.values():
        result["delta_nll"] = result["nll"] - native
    shape = results["all_leaf"]["leaf_distribution"]
    trajectory = results["all_leaf"]["address_trajectory"]
    linked = {
        "top1_mass_ge_0p80": shape["top1_mass"] >= 0.80,
        "top1_damage_le_0p02": results["top1"]["delta_nll"] <= 0.02,
        "drop_top1_damage_ge_0p10": results["drop_top1"]["delta_nll"] >= 0.10,
        "sequential_rate_ge_0p70": trajectory["stationary_plus_adjacent_rate"] >= 0.70,
    }
    depth_damages = [
        results[f"uniform_depth_{depth}"]["delta_nll"]
        for depth in range(model.decoder.depths)
    ]
    distributed = {
        "top1_damage_gt_0p05": results["top1"]["delta_nll"] > 0.05,
        "top4_or_top8_recovers": min(
            results["top4"]["delta_nll"], results["top8"]["delta_nll"],
        ) < results["top1"]["delta_nll"] - 0.02,
        "at_least_two_causal_depths": sum(value >= 0.02 for value in depth_damages) >= 2,
    }
    report = {
        "diagnostic": "S3-C10-PATH-SHAPE-AUDIT-D01",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "state_sha256": checkpoint.get("state_sha256"),
        "rows": len(rows),
        "results": results,
        "linked_path_gates": linked,
        "distributed_leaf_gates": distributed,
        "classification": (
            "linked_path_shortcut" if all(linked.values()) else
            "distributed_leaf_index" if all(distributed.values()) else
            "mixed_or_degenerate_leaf_protocol"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
