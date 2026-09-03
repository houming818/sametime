#!/usr/bin/env python3
"""Measure CE/predictive gradient scale and alignment for a trained D12 probe."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

import s3_predictive_state_sketch_d12 as d12
import s3_pretrain_task_posterior_pipeline as c10
import s3_recursive_depth_probability_exposure as d03
import s3_structural_protocol_full_pipeline_d10 as d10
import s3_structural_slot_ownership_d09_scale as d09


def gradient_vector(loss, values, retain_graph=True):
    gradients = torch.autograd.grad(
        loss, values, retain_graph=retain_graph, allow_unused=True,
    )
    return [
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(values, gradients)
    ]


def compare_gradients(left, right):
    left_square = sum(float(value.detach().double().square().sum()) for value in left)
    right_square = sum(float(value.detach().double().square().sum()) for value in right)
    dot = sum(float((a.detach().double() * b.detach().double()).sum()) for a, b in zip(left, right))
    left_norm = math.sqrt(left_square)
    right_norm = math.sqrt(right_square)
    return {
        "ce_norm": left_norm,
        "predictive_norm": right_norm,
        "predictive_to_ce_ratio": right_norm / max(left_norm, 1e-30),
        "cosine": dot / max(left_norm * right_norm, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--d12-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parallel-data", default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv")
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11201)
    parser.add_argument("--ownership-seed", type=int, default=11202)
    parser.add_argument("--sketch-seed", type=int, default=11203)
    parser.add_argument("--sketch-width", type=int, default=128)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--start-line", type=int, default=8012)
    parser.add_argument("--eval-rows", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.wmt_data = args.eval_wmt_data
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    args.pad, args.vocab = pieces, pieces + 3
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    source_cpu, _, config, source_hash, _ = d03.load_model(
        Path(args.source_checkpoint), args, sp, args.pad, args.vocab,
    )
    model, warm = d12.make_model(source_cpu, config, args, Path(args.warm_start))
    checkpoint = torch.load(args.d12_checkpoint, map_location="cpu", weights_only=False)
    d09.load_trainable(model, checkpoint)
    predictor = d12.LevelPredictor(
        config.dim, int(math.log2(args.max_slots)) + 1, args.sketch_width,
    ).to(args.device)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    sketcher = d12.FixedOutcomeSketch(
        args.vocab, args.sketch_width, args.max_target, args.sketch_seed,
    ).to(args.device)
    if sketcher.digest() != checkpoint["run"]["sketch_sha256"]:
        raise RuntimeError("fixed sketch hash mismatch")

    _, _, excluded = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, 32,
    )
    iterator = d10.iter_parallel_batches(
        Path(args.parallel_data), sp, direction_ids, eos, args.batch_size,
        args.start_line, args.start_line + args.batches * args.batch_size * 2, excluded,
    )
    model.train()
    predictor.train()
    groups = {
        "compressor": [parameter for parameter in model.compressor.parameters() if parameter.requires_grad],
        "up_kernel": [parameter for parameter in model.reconstructor.up_kernel.parameters() if parameter.requires_grad],
        "protocol_gain": [model.protocol_gain_logit],
    }
    rows = []
    for batch_index, (_, batch, _) in enumerate(iterator):
        if batch_index >= args.batches:
            break
        source, lengths, target = c10.collate_rows(batch, args.pad, args.device)
        depth = d12.DEPTHS[batch_index % len(d12.DEPTHS)]
        payload = d12.forward_batch(
            model, predictor, sketcher, source, lengths, target,
            bos, args.pad, depth, False,
        )
        tokens = int(target.ne(args.pad).sum())
        ce = F.cross_entropy(
            payload["logits"].reshape(-1, payload["logits"].shape[-1]),
            target.reshape(-1), ignore_index=args.pad, reduction="sum",
        ) / max(1, tokens)
        predictive = d12.sketch_loss(payload["predictions"], payload["target_state"])
        row = {
            "batch": batch_index, "depth": depth,
            "ce": float(ce.detach()), "predictive_loss": float(predictive.detach()),
        }
        tree_ce = gradient_vector(ce, payload["tree"])
        tree_predictive = gradient_vector(predictive, payload["tree"])
        row["tree_all"] = compare_gradients(tree_ce, tree_predictive)
        row["tree_by_level"] = [
            compare_gradients([left], [right])
            for left, right in zip(tree_ce, tree_predictive)
        ]
        row["parameter_groups"] = {}
        for name, parameters in groups.items():
            ce_gradient = gradient_vector(ce, parameters)
            predictive_gradient = gradient_vector(predictive, parameters)
            row["parameter_groups"][name] = compare_gradients(
                ce_gradient, predictive_gradient,
            )
        rows.append(row)

    if len(rows) != args.batches:
        raise RuntimeError(f"expected {args.batches} batches, got {len(rows)}")
    ratios = [row["tree_all"]["predictive_to_ce_ratio"] for row in rows]
    cosines = [row["tree_all"]["cosine"] for row in rows]
    median_ratio = sorted(ratios)[len(ratios) // 2]
    result = {
        "claim": d12.CLAIM,
        "audit": "post-smoke gradient calibration",
        "source_sha256": source_hash,
        "warm_start_sha256": warm["trainable_state_sha256"],
        "checkpoint_model_sha256": checkpoint["trainable_state_sha256"],
        "checkpoint_predictor_sha256": checkpoint["predictor_state_sha256"],
        "sketch_sha256": sketcher.digest(),
        "batches": rows,
        "aggregate": {
            "tree_ratio_min": min(ratios),
            "tree_ratio_median": median_ratio,
            "tree_ratio_max": max(ratios),
            "tree_cosine_mean": sum(cosines) / len(cosines),
            "suggested_loss_weights": {
                str(target): target / max(median_ratio, 1e-30)
                for target in (0.01, 0.05, 0.20)
            },
        },
        "interpretation": (
            "Suggested weights target auxiliary/CE TreeHeap gradient-norm ratios; "
            "they are calibration observations, not authorized training settings."
        ),
    }
    d12.write_json(Path(args.output), result)
    print(json.dumps(result["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
