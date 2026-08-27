#!/usr/bin/env python3
"""Frozen 3x3 probe of TreeHeap READ depth against target collapse length."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_multilevel_read_ablation_c12 as c12  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-RECURSIVE-DEPTH-LENGTH-COLLAPSE-D04"
DEPTHS = (5, 6, 7)
BLOCKS = (4, 2, 1)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pearson(left, right) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    lc, rc = [x - lm for x in left], [x - rm for x in right]
    numerator = sum(a * b for a, b in zip(lc, rc))
    denominator = math.sqrt(sum(x * x for x in lc) * sum(x * x for x in rc))
    if denominator == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / denominator


def empty_metric():
    return {
        "loss_sum": 0.0,
        "tokens": 0,
        "groups": 0,
        "block_hit": 0,
        "block_mass_sum": 0.0,
        "target_entropy_sum": 0.0,
        "adjacent_equal": 0,
        "adjacent_total": 0,
    }


@torch.no_grad()
def collapsed_teacher(decoder, tree, masks, target, bos, pad, depth, block_size):
    batch, target_steps = target.shape
    hidden = tree[0].new_zeros((batch, decoder.hidden))
    previous_embedding = decoder.embedding(
        torch.full((batch,), bos, dtype=torch.long, device=target.device)
    )
    metric = empty_metric()
    predicted_groups = []
    valid_groups = []
    previous_top1 = None
    previous_valid = None
    for start in range(0, target_steps, block_size):
        group = target[:, start:start + block_size]
        valid = group.ne(pad)
        group_valid = valid.any(-1)
        safe = group.masked_fill(~valid, 0)
        context, _ = d03.read_capped(decoder, hidden, tree, masks, depth)
        hidden = decoder.cell(
            torch.cat((previous_embedding, context), dim=-1), hidden,
        )
        logits = decoder.output(torch.cat((hidden, context), dim=-1)).float()
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        gathered = log_probs.gather(1, safe)
        metric["loss_sum"] += float((-(gathered * valid)).sum())
        metric["tokens"] += int(valid.sum())
        metric["groups"] += int(group_valid.sum())

        top1 = logits.argmax(-1)
        hit = ((top1[:, None] == safe) & valid).any(-1)
        metric["block_hit"] += int((hit & group_valid).sum())

        counts = torch.zeros_like(probabilities)
        counts.scatter_add_(1, safe, valid.to(probabilities.dtype))
        unique = counts.gt(0)
        block_mass = (probabilities * unique).sum(-1)
        metric["block_mass_sum"] += float(block_mass[group_valid].sum())
        weights = counts / counts.sum(-1, keepdim=True).clamp_min(1.0)
        target_entropy = -(weights.clamp_min(1e-12).log() * weights).sum(-1)
        metric["target_entropy_sum"] += float(target_entropy[group_valid].sum())

        if previous_top1 is not None:
            pair_valid = previous_valid & group_valid
            metric["adjacent_equal"] += int(
                (previous_top1.eq(top1) & pair_valid).sum()
            )
            metric["adjacent_total"] += int(pair_valid.sum())
        predicted_groups.append(top1)
        valid_groups.append(group_valid)
        previous_top1, previous_valid = top1, group_valid

        embedded = decoder.embedding(safe)
        group_embedding = (embedded * valid[:, :, None]).sum(1)
        group_embedding = group_embedding / valid.sum(-1, keepdim=True).clamp_min(1)
        previous_embedding = torch.where(
            group_valid[:, None], group_embedding, previous_embedding,
        )

    return metric, torch.stack(predicted_groups, dim=1), torch.stack(valid_groups, dim=1)


def merge_metric(target, source):
    for key in target:
        target[key] += source[key]


def finalize(metric, original_rows: int, depth: int, block_size: int):
    tokens = max(1, metric["tokens"])
    groups = max(1, metric["groups"])
    nll = metric["loss_sum"] / tokens
    return {
        "depth": depth,
        "block_size": block_size,
        "target_length_ratio": 1.0 / block_size,
        "nll_per_original_token": nll,
        "ppl": math.exp(min(20.0, nll)),
        "original_tokens": metric["tokens"],
        "output_groups": metric["groups"],
        "groups_per_row": metric["groups"] / max(1, original_rows),
        "realized_address_ratio": metric["groups"] / tokens,
        "top1_block_hit_rate": metric["block_hit"] / groups,
        "mean_block_probability_mass": metric["block_mass_sum"] / groups,
        "mean_target_block_entropy": metric["target_entropy_sum"] / groups,
        "adjacent_repetition_rate": metric["adjacent_equal"] / max(1, metric["adjacent_total"]),
    }


def decode_groups(sp, ids, valid, pieces, eos):
    selected = [
        int(token) for token, keep in zip(ids.tolist(), valid.tolist())
        if keep and 0 <= int(token) < pieces
    ]
    selected = c10.wmt.clean(selected, eos, pieces)
    return sp.decode(selected)


@torch.no_grad()
def probe_checkpoint(checkpoint_path, output, args, sp, pieces, pad, bos, eos, vocab):
    model, saved, config, state_hash, parent_hash = d03.load_model(
        checkpoint_path, args, sp, pad, vocab,
    )
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    test_rows = test_rows[:args.eval_rows]
    row_hash = c12.rows_sha256(test_rows)
    accumulators = {
        (depth, block): empty_metric()
        for depth in DEPTHS for block in BLOCKS
    }
    examples = []
    reference_loss_sum = 0.0
    reference_tokens = 0
    for start in range(0, len(test_rows), args.batch_size):
        rows = test_rows[start:start + args.batch_size]
        source, length, target = c10.collate_rows(rows, pad, args.device)
        levels, masks = d03.condition_states(model, source, length, "native")
        tree = model.decoder.convolve(levels, masks)
        reference_logits, _ = model.decoder.teacher(levels, masks, target, bos)
        reference_loss_sum += float(F.cross_entropy(
            reference_logits.reshape(-1, reference_logits.shape[-1]),
            target.reshape(-1), ignore_index=pad, reduction="sum",
        ))
        reference_tokens += int(target.ne(pad).sum())

        batch_outputs = {}
        for depth in DEPTHS:
            for block in BLOCKS:
                metric, predicted, group_valid = collapsed_teacher(
                    model.decoder, tree, masks, target, bos, pad, depth, block,
                )
                merge_metric(accumulators[(depth, block)], metric)
                batch_outputs[(depth, block)] = (predicted, group_valid)

        if start == 0:
            for row_index, row in enumerate(rows[:args.example_rows]):
                reference = c10.wmt.clean(target[row_index].tolist(), eos, pieces)
                example = {
                    "direction": row[2],
                    "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                    "reference": sp.decode(reference),
                    "outputs": {},
                }
                for depth in DEPTHS:
                    for block in BLOCKS:
                        predicted, group_valid = batch_outputs[(depth, block)]
                        example["outputs"][f"d{depth}_b{block}"] = decode_groups(
                            sp, predicted[row_index], group_valid[row_index], pieces, eos,
                        )
                examples.append(example)

    cells = {
        f"d{depth}_b{block}": finalize(
            accumulators[(depth, block)], len(test_rows), depth, block,
        )
        for depth in DEPTHS for block in BLOCKS
    }
    reference_nll = reference_loss_sum / max(1, reference_tokens)
    full_error = abs(cells["d7_b1"]["nll_per_original_token"] - reference_nll)
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "seed": int(saved["config"]["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_sha256": state_hash,
        "parent_state_sha256": parent_hash,
        "rows": len(test_rows),
        "test_row_sha256": row_hash,
        "cells": cells,
        "reference_full_nll": reference_nll,
        "d7_b1_reference_error": full_error,
        "finite": all(
            math.isfinite(float(value))
            for cell in cells.values()
            for value in cell.values()
            if isinstance(value, (int, float))
        ),
        "examples": examples,
    }
    write_json(output / "summary.json", summary)
    return summary


def aggregate(summaries, output, args):
    keys = [f"d{depth}_b{block}" for depth in DEPTHS for block in BLOCKS]
    row_hashes = {summary["test_row_sha256"] for summary in summaries}
    p0 = (
        len(row_hashes) == 1
        and all(summary["finite"] for summary in summaries)
        and all(summary["d7_b1_reference_error"] <= 1e-6 for summary in summaries)
        and all(
            cell["target_length_ratio"]
            <= cell["realized_address_ratio"]
            <= cell["target_length_ratio"]
            + summary["rows"] / max(1, cell["original_tokens"])
            + 1e-9
            for summary in summaries
            for cell in summary["cells"].values()
        )
    )
    p1_seed = {}
    p2_seed = {}
    vectors = {}
    for summary in summaries:
        cells = summary["cells"]
        seed = str(summary["seed"])
        p1_seed[seed] = (
            cells["d5_b4"]["nll_per_original_token"] < cells["d5_b1"]["nll_per_original_token"]
            and cells["d6_b2"]["nll_per_original_token"] < cells["d6_b1"]["nll_per_original_token"]
        )
        best = {
            str(depth): min(
                BLOCKS,
                key=lambda block: cells[f"d{depth}_b{block}"]["nll_per_original_token"],
            )
            for depth in DEPTHS
        }
        p2_seed[seed] = {
            "best_blocks": best,
            "pass": best == {"5": 4, "6": 2, "7": 1},
        }
        vectors[seed] = [cells[key]["nll_per_original_token"] for key in keys]
    correlations = []
    seeds = list(vectors)
    for left_index in range(len(seeds)):
        for right_index in range(left_index + 1, len(seeds)):
            left, right = seeds[left_index], seeds[right_index]
            correlations.append({
                "left": int(left),
                "right": int(right),
                "pearson": pearson(vectors[left], vectors[right]),
            })
    p1 = sum(p1_seed.values()) >= 2
    p2 = sum(row["pass"] for row in p2_seed.values()) >= 2
    p3 = sum(row["pearson"] >= 0.90 for row in correlations) >= 2
    result = {
        "claim": CLAIM,
        "mode": args.mode,
        "seeds": [summary["seed"] for summary in summaries],
        "test_row_sha256": next(iter(row_hashes)) if len(row_hashes) == 1 else sorted(row_hashes),
        "cell_order": keys,
        "nll_vectors": vectors,
        "p1_seed": p1_seed,
        "p2_seed": p2_seed,
        "seed_correlations": correlations,
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3},
        "decision": (
            "supported_exact_depth_length_alignment"
            if p0 and p1 and p2 and p3
            else "partial_or_not_supported_as_registered"
        ),
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--example-rows", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "smoke":
        args.eval_rows = args.eval_rows or 64
        args.batch_size = args.batch_size or 8
    else:
        args.eval_rows = args.eval_rows or 256
        args.batch_size = args.batch_size or 8
    random.seed(17301)
    torch.manual_seed(17301)
    torch.cuda.manual_seed_all(17301)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad, vocab = pieces, pieces + 3
    summaries = []
    for checkpoint_name in args.checkpoints:
        checkpoint = Path(checkpoint_name)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seed = int(saved["config"]["seed"])
        summary = probe_checkpoint(
            checkpoint, output / f"seed_{seed}", args,
            sp, pieces, pad, bos, eos, vocab,
        )
        summaries.append(summary)
        print(json.dumps({
            "event": "checkpoint_complete",
            "seed": seed,
            "d5_b4_nll": summary["cells"]["d5_b4"]["nll_per_original_token"],
            "d6_b2_nll": summary["cells"]["d6_b2"]["nll_per_original_token"],
            "d7_b1_nll": summary["cells"]["d7_b1"]["nll_per_original_token"],
        }), flush=True)
        del saved
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate(summaries, output, args)


if __name__ == "__main__":
    main()
