#!/usr/bin/env python3
"""Frozen cumulative-depth probe for the TreeHeap decoder probability field."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_multilevel_read_ablation_c12 as c12  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-RECURSIVE-DEPTH-PROBABILITY-EXPOSURE-D03"
CONDITIONS = ("native", "runtime_identity", "pair_break_depth_0", "source_shuffle")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def piece_label(sp, token_id: int, pieces: int) -> str:
    if 0 <= token_id < pieces:
        return sp.id_to_piece(token_id)
    return {
        pieces: "<pad>",
        pieces + 1: "<en2zh>",
        pieces + 2: "<zh2en>",
    }.get(token_id, f"<id:{token_id}>")


def self_test(output: Path) -> None:
    dim, hidden, depths, vocab = 8, 11, 4, 23

    class OldDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab, dim)
            self.query = torch.nn.Linear(hidden, dim)
            self.cell = torch.nn.GRUCell(2 * dim, hidden)
            self.output = torch.nn.Linear(hidden + dim, vocab)
            self.branch = torch.nn.Linear(hidden, dim)
            self.depth_embedding = torch.nn.Embedding(depths, dim)

    torch.manual_seed(17201)
    decoder = c12.c11.MultiLevelConvolutionDecoder(
        OldDecoder(), dim, hidden, depths, use_up=False,
    )
    state = torch.randn(3, hidden)
    tree = [torch.randn(3, 2 ** depth, dim) for depth in range(depths)]
    masks = [
        torch.ones(3, 2 ** depth, dtype=torch.bool)
        for depth in range(depths)
    ]
    expected, _ = decoder.read(state, tree, masks)
    actual, _ = read_capped(decoder, state, tree, masks, depths - 1)
    error = float((expected - actual).abs().max())
    result = {
        "claim": CLAIM,
        "self_test": "native_read_equals_full_depth_capped_read",
        "full_depth_max_abs_error": error,
        "pass": error <= 1e-7,
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    if not result["pass"]:
        raise RuntimeError(f"full-depth READ equivalence failed: {error}")


def pearson(left, right) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(centered_left, centered_right))
    left_norm = math.sqrt(sum(value * value for value in centered_left))
    right_norm = math.sqrt(sum(value * value for value in centered_right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def read_capped(decoder, hidden, tree, masks, max_depth: int):
    base_query = decoder.query(hidden)
    query = base_query
    frontier = masks[0].to(query.dtype)
    gain = torch.sigmoid(decoder.read_gain_logit)
    entropies = []
    last_depth = min(max_depth, len(tree) - 1)
    local = None
    for depth in range(last_depth + 1):
        nodes, valid = tree[depth], masks[depth]
        frontier = frontier * valid.to(frontier.dtype)
        frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
        local = (frontier[:, :, None] * nodes).sum(1)
        depth_state = decoder.depth_embedding.weight[depth][None].expand_as(local)
        query = query + gain * decoder.read_kernel(query, local, depth_state)
        entropy = -(
            frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()
        ).sum(-1)
        entropies.append(entropy)
        if depth == last_depth:
            break
        children = tree[depth + 1].reshape(
            nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
        )
        child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        branch_query = (
            decoder.branch(hidden) + gain * (query - base_query)
        )[:, None, None]
        scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
        scores = scores.masked_fill(~child_valid, -1e9)
        probability = F.softmax(scores, dim=-1)
        probability = probability * child_valid.to(probability.dtype)
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
        frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
    return local + (query - base_query), torch.stack(entropies, dim=-1)


def teacher_capped(decoder, tree, masks, target, bos: int, max_depth: int):
    hidden = tree[0].new_zeros((tree[0].shape[0], decoder.hidden))
    previous = torch.full(
        (tree[0].shape[0],), bos, dtype=torch.long, device=tree[0].device,
    )
    logits = []
    final_frontier_entropy = []
    for step in range(target.shape[1]):
        context, entropy = read_capped(decoder, hidden, tree, masks, max_depth)
        hidden = decoder.cell(
            torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
        )
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        final_frontier_entropy.append(entropy[:, -1])
        previous = target[:, step]
    return torch.stack(logits, dim=1), torch.stack(final_frontier_entropy, dim=1)


def condition_states(model, source, length, condition: str):
    previous_mode = model.encoder.runtime_mode
    intervention = "native"
    pair_break_depth = -1
    runtime_mode = None
    if condition == "runtime_identity":
        runtime_mode = "identity"
    elif condition == "pair_break_depth_0":
        pair_break_depth = 0
    elif condition == "source_shuffle":
        intervention = "source_shuffle"
    elif condition != "native":
        raise ValueError(condition)
    model.encoder.runtime_mode = runtime_mode
    try:
        _, _, _, levels, masks = model.states(
            source, length, intervention=intervention,
            pair_break_depth=pair_break_depth,
        )
    finally:
        model.encoder.runtime_mode = previous_mode
    return levels, masks


def empty_depth_metric():
    return {
        "loss_sum": 0.0,
        "tokens": 0,
        "entropy_sum": 0.0,
        "true_probability_sum": 0.0,
        "top1_correct": 0,
        "frontier_entropy_sum": 0.0,
        "js_from_previous_sum": 0.0,
        "improved_true_logp": 0,
        "top1_changed": 0,
        "transition_tokens": 0,
    }


def finalize_metric(metric, depth: int, width: int):
    tokens = max(1, metric["tokens"])
    transitions = max(1, metric["transition_tokens"])
    nll = metric["loss_sum"] / tokens
    return {
        "depth": depth,
        "width": width,
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": metric["tokens"],
        "mean_vocab_entropy": metric["entropy_sum"] / tokens,
        "mean_true_probability": metric["true_probability_sum"] / tokens,
        "top1_accuracy": metric["top1_correct"] / tokens,
        "mean_frontier_entropy": metric["frontier_entropy_sum"] / tokens,
        "js_from_previous": (
            metric["js_from_previous_sum"] / transitions
            if metric["transition_tokens"] else None
        ),
        "true_logp_improvement_fraction": (
            metric["improved_true_logp"] / transitions
            if metric["transition_tokens"] else None
        ),
        "top1_change_fraction": (
            metric["top1_changed"] / transitions
            if metric["transition_tokens"] else None
        ),
    }


def load_model(checkpoint_path: Path, args, sp, pad: int, vocab: int):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("arm") != "read":
        raise RuntimeError(f"D03 requires a read checkpoint: {checkpoint_path}")
    saved_hash = c10.state_sha256(saved["state_dict"])
    if saved.get("state_sha256") and saved_hash != saved["state_sha256"]:
        raise RuntimeError(f"checkpoint state hash mismatch: {checkpoint_path}")
    run_config = saved["config"]
    parent_path = Path(run_config["checkpoint"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_hash = c10.state_sha256(parent["state_dict"])
    if parent.get("state_sha256") and parent_hash != parent["state_sha256"]:
        raise RuntimeError(f"parent state hash mismatch: {parent_path}")
    config = SimpleNamespace(**parent["config"])
    config.device = args.device
    config.wmt_data = args.wmt_data
    config.task_train_rows = int(run_config["train_rows"])
    config.task_eval_rows = max(args.eval_rows, int(run_config["eval_rows"]))
    config.max_wmt_scan_lines = 3_000_000
    model = c12.build_arm(
        config, vocab, pad, parent["state_dict"], "read", int(run_config["seed"]),
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved, config, saved_hash, parent_hash


@torch.no_grad()
def probe_checkpoint(checkpoint_path: Path, output: Path, args, sp, pieces, pad, bos, eos, vocab):
    model, saved, config, state_hash, parent_hash = load_model(
        checkpoint_path, args, sp, pad, vocab,
    )
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    test_rows = test_rows[:args.eval_rows]
    row_hash = c12.rows_sha256(test_rows)
    depth_count = model.decoder.depth_embedding.num_embeddings
    conditions = [name for name in args.conditions if name in CONDITIONS]
    if not conditions or "native" not in conditions:
        raise RuntimeError("conditions must include native")
    accumulators = {
        condition: [empty_depth_metric() for _ in range(depth_count)]
        for condition in conditions
    }
    native_reference_loss_sum = 0.0
    native_reference_tokens = 0
    full_depth_max_abs_logit_error = 0.0
    examples = []
    for start in range(0, len(test_rows), args.batch_size):
        batch_rows = test_rows[start:start + args.batch_size]
        source, length, target = c10.collate_rows(batch_rows, pad, args.device)
        valid = target.ne(pad)
        safe_target = target.clamp_min(0)
        for condition in conditions:
            levels, masks = condition_states(model, source, length, condition)
            tree = model.decoder.convolve(levels, masks)
            reference_logits = None
            if condition == "native":
                reference_logits, _ = model.decoder.teacher(levels, masks, target, bos)
                native_reference_loss_sum += float(F.cross_entropy(
                    reference_logits.reshape(-1, reference_logits.shape[-1]),
                    target.reshape(-1), ignore_index=pad, reduction="sum",
                ))
                native_reference_tokens += int(valid.sum())
            previous_log_probs = None
            previous_top1 = None
            for depth in range(depth_count):
                logits, frontier_entropy = teacher_capped(
                    model.decoder, tree, masks, target, bos, depth,
                )
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probabilities = log_probs.exp()
                true_logp = log_probs.gather(-1, safe_target[:, :, None]).squeeze(-1)
                true_probability = true_logp.exp()
                entropy = -(probabilities * log_probs).sum(-1)
                top1 = logits.argmax(-1)
                if reference_logits is not None and depth == depth_count - 1:
                    full_depth_max_abs_logit_error = max(
                        full_depth_max_abs_logit_error,
                        float((logits - reference_logits).abs().max()),
                    )
                metric = accumulators[condition][depth]
                metric["loss_sum"] += float((-true_logp[valid]).sum())
                metric["tokens"] += int(valid.sum())
                metric["entropy_sum"] += float(entropy[valid].sum())
                metric["true_probability_sum"] += float(true_probability[valid].sum())
                metric["top1_correct"] += int((top1.eq(target) & valid).sum())
                metric["frontier_entropy_sum"] += float(frontier_entropy[valid].sum())
                if previous_log_probs is not None:
                    log_midpoint = torch.logaddexp(previous_log_probs, log_probs) - math.log(2.0)
                    previous_probabilities = previous_log_probs.exp()
                    js = 0.5 * (
                        (previous_probabilities * (previous_log_probs - log_midpoint)).sum(-1)
                        + (probabilities * (log_probs - log_midpoint)).sum(-1)
                    )
                    metric["js_from_previous_sum"] += float(js[valid].sum())
                    metric["improved_true_logp"] += int(
                        ((true_logp > previous_log_probs.gather(
                            -1, safe_target[:, :, None],
                        ).squeeze(-1) + 1e-8) & valid).sum()
                    )
                    metric["top1_changed"] += int((top1.ne(previous_top1) & valid).sum())
                    metric["transition_tokens"] += int(valid.sum())
                if condition == "native" and start == 0:
                    if depth == 0 and not examples:
                        for row_index, row in enumerate(batch_rows[:args.example_rows]):
                            reference_ids = c10.wmt.clean(target[row_index].tolist(), eos, pieces)
                            examples.append({
                                "direction": row[2],
                                "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                                "reference": sp.decode(reference_ids),
                                "depths": [],
                            })
                    for row_index, example in enumerate(examples):
                        predicted_ids = c10.wmt.clean(top1[row_index].tolist(), eos, pieces)
                        positions = []
                        for position in range(min(args.top_positions, target.shape[1])):
                            values, indices = probabilities[row_index, position].topk(args.top_k)
                            positions.append({
                                "position": position,
                                "target": piece_label(
                                    sp, int(target[row_index, position]), pieces,
                                ),
                                "top": [
                                    {
                                        "piece": piece_label(sp, int(index), pieces),
                                        "probability": float(value),
                                    }
                                    for value, index in zip(values, indices)
                                ],
                            })
                        example["depths"].append({
                            "depth": depth,
                            "width": int(levels[depth].shape[1]),
                            "teacher_forced_top1": sp.decode(predicted_ids),
                            "positions": positions,
                        })
                previous_log_probs = log_probs
                previous_top1 = top1

    curves = {}
    for condition, metrics in accumulators.items():
        curves[condition] = [
            finalize_metric(metric, depth, 2 ** depth)
            for depth, metric in enumerate(metrics)
        ]
    native_nll = [row["nll"] for row in curves["native"]]
    native_js = [row["js_from_previous"] for row in curves["native"][1:]]
    reference_full_depth_nll = (
        native_reference_loss_sum / max(1, native_reference_tokens)
    )
    full_depth_reference_error = abs(native_nll[-1] - reference_full_depth_nll)
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_sha256": state_hash,
        "parent_state_sha256": parent_hash,
        "seed": int(saved["config"]["seed"]),
        "rows": len(test_rows),
        "test_row_sha256": row_hash,
        "conditions": conditions,
        "depth_count": depth_count,
        "curves": curves,
        "diagnostics": {
            "native_nll_range": max(native_nll) - min(native_nll),
            "native_adjacent_js_over_1e_4": sum(
                value is not None and value > 1e-4 for value in native_js
            ),
            "native_nll_deltas": [
                native_nll[index] - native_nll[index - 1]
                for index in range(1, len(native_nll))
            ],
            "full_depth_nll": native_nll[-1],
            "reference_full_depth_nll": reference_full_depth_nll,
            "full_depth_reference_error": full_depth_reference_error,
            "full_depth_max_abs_logit_error": full_depth_max_abs_logit_error,
        },
        "examples": examples,
    }
    numeric_values = []
    for curve in curves.values():
        for row in curve:
            numeric_values.extend(value for value in row.values() if isinstance(value, (int, float)))
    summary["finite"] = all(finite_number(value) for value in numeric_values)
    write_json(output / "summary.json", summary)
    return summary


def aggregate(summaries, output: Path, args):
    row_hashes = {summary["test_row_sha256"] for summary in summaries}
    native_deltas = {
        str(summary["seed"]): summary["diagnostics"]["native_nll_deltas"]
        for summary in summaries
    }
    seed_pairs = []
    for left_index in range(len(summaries)):
        for right_index in range(left_index + 1, len(summaries)):
            left, right = summaries[left_index], summaries[right_index]
            seed_pairs.append({
                "left": left["seed"],
                "right": right["seed"],
                "native_delta_pearson": pearson(
                    left["diagnostics"]["native_nll_deltas"],
                    right["diagnostics"]["native_nll_deltas"],
                ),
            })
    exposure_seed_passes = 0
    structural_seed_passes = 0
    structural_offsets = {}
    for summary in summaries:
        diagnostics = summary["diagnostics"]
        exposure = (
            diagnostics["native_nll_range"] >= 0.05
            and diagnostics["native_adjacent_js_over_1e_4"] >= 2
        )
        exposure_seed_passes += int(exposure)
        native_curve = summary["curves"]["native"]
        native_delta = diagnostics["native_nll_deltas"]
        offsets = {}
        for condition in ("runtime_identity", "pair_break_depth_0"):
            if condition not in summary["curves"]:
                continue
            condition_nll = [row["nll"] for row in summary["curves"][condition]]
            condition_delta = [
                condition_nll[index] - condition_nll[index - 1]
                for index in range(1, len(condition_nll))
            ]
            offsets[condition] = sum(
                abs(a - b) for a, b in zip(native_delta, condition_delta)
            ) / max(1, len(native_delta))
        source_damage = 0.0
        if "source_shuffle" in summary["curves"]:
            source_damage = (
                summary["curves"]["source_shuffle"][-1]["nll"]
                - native_curve[-1]["nll"]
            )
        structural = bool(offsets) and max(offsets.values()) >= 0.005 and source_damage > 0.0
        structural_seed_passes += int(structural)
        structural_offsets[str(summary["seed"])] = {
            "mean_abs_delta_offsets": offsets,
            "source_shuffle_full_depth_damage": source_damage,
            "pass": structural,
        }
    p0 = (
        len(row_hashes) == 1
        and all(summary["finite"] for summary in summaries)
        and all(
            summary["diagnostics"]["full_depth_reference_error"] <= 1e-6
            and summary["diagnostics"]["full_depth_max_abs_logit_error"] <= 1e-6
            for summary in summaries
        )
    )
    p1 = exposure_seed_passes >= 2
    p2 = sum(row["native_delta_pearson"] >= 0.50 for row in seed_pairs) >= 2
    p3 = structural_seed_passes >= 2
    aggregate_summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "seed_summaries": [str(path) for path in args.checkpoints],
        "seeds": [summary["seed"] for summary in summaries],
        "test_row_sha256": next(iter(row_hashes)) if len(row_hashes) == 1 else sorted(row_hashes),
        "native_nll_deltas": native_deltas,
        "seed_pair_correlations": seed_pairs,
        "structural_offsets": structural_offsets,
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3},
        "decision": (
            "supported_frozen_recursive_exposure"
            if p0 and p1 and p2 and p3
            else "not_supported_as_registered"
        ),
    }
    write_json(output / "summary.json", aggregate_summary)
    print(json.dumps(aggregate_summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", default=[])
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--example-rows", type=int, default=4)
    parser.add_argument("--top-positions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(output)
        return
    if not args.checkpoints:
        parser.error("at least one --checkpoint is required unless --self-test is used")
    if args.mode == "smoke":
        args.eval_rows = args.eval_rows or 32
        args.batch_size = args.batch_size or 4
    else:
        args.eval_rows = args.eval_rows or 256
        args.batch_size = args.batch_size or 8
    random.seed(17201)
    torch.manual_seed(17201)
    torch.cuda.manual_seed_all(17201)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    vocab = pieces + 3
    summaries = []
    for checkpoint_name in args.checkpoints:
        checkpoint = Path(checkpoint_name)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seed = int(saved["config"]["seed"])
        seed_output = output / f"seed_{seed}"
        summary = probe_checkpoint(
            checkpoint, seed_output, args, sp, pieces, pad, bos, eos, vocab,
        )
        summaries.append(summary)
        print(json.dumps({
            "event": "checkpoint_complete",
            "seed": seed,
            "native_nll_range": summary["diagnostics"]["native_nll_range"],
            "full_depth_nll": summary["diagnostics"]["full_depth_nll"],
        }), flush=True)
        del saved
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate(summaries, output, args)


if __name__ == "__main__":
    main()
