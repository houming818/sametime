#!/usr/bin/env python3
"""Frozen-checkpoint audit for C10's recursive TreeHeap read path."""
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


@torch.no_grad()
def evaluate(
    model,
    rows,
    pad: int,
    bos: int,
    device: str,
    batch_size: int,
    route_mode: str = "native",
    pair_break_depth: int = -1,
):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    route_sum = None
    examples = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        source, length, target = c10.collate_rows(batch, pad, device)
        logits, route = model.teacher(
            source,
            length,
            target,
            bos,
            route_mode=route_mode,
            pair_break_depth=pair_break_depth,
        )
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="sum",
        ))
        token_count += int(target.ne(pad).sum())
        weight = len(batch)
        route = route.detach().double().cpu()
        full_depths = model.decoder.depths + 1
        if route.numel() < full_depths:
            # Dynamic TreeHeaps have different root-to-leaf depths. Align the
            # terminal leaf bucket so leaf-collapse remains comparable.
            route = F.pad(route, (full_depths - route.numel(), 0))
        weighted_route = route * weight
        route_sum = weighted_route if route_sum is None else route_sum + weighted_route
        examples += weight
    nll = loss_sum / max(1, token_count)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": token_count,
        "rows": examples,
        "route": [float(value) for value in (route_sum / max(1, examples))],
    }


def main() -> None:
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
    config.device = args.device
    config.wmt_data = args.wmt_data
    config.spm_model = args.spm_model
    config.task_train_rows = 1
    config.task_eval_rows = args.rows

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces = sp.get_piece_size()
    eos, bos = sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3

    model = c10.build_model(config, vocab, pad).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)

    results = {}
    for route_mode in ("native", "force_leaf", "force_root"):
        results[route_mode] = evaluate(
            model, test_rows, pad, bos, args.device, args.batch_size,
            route_mode=route_mode,
        )

    model.encoder.runtime_mode = "identity"
    results["runtime_identity"] = evaluate(
        model, test_rows, pad, bos, args.device, args.batch_size,
    )
    model.encoder.runtime_mode = None

    results["pair_break_depth_0"] = evaluate(
        model, test_rows, pad, bos, args.device, args.batch_size,
        pair_break_depth=0,
    )

    branch_weight = model.decoder.branch.weight.detach().clone()
    with torch.no_grad():
        model.decoder.branch.weight.zero_()
    results["uniform_branch"] = evaluate(
        model, test_rows, pad, bos, args.device, args.batch_size,
    )
    with torch.no_grad():
        model.decoder.branch.weight.copy_(branch_weight)

    native = results["native"]["nll"]
    report = {
        "experiment": "S3-C10-READ-PATH-AUDIT",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_state_sha256": checkpoint.get("state_sha256"),
        "rows": len(test_rows),
        "results": results,
        "deltas_from_native": {
            name: row["nll"] - native for name, row in results.items()
        },
        "interpretation_gate": {
            "native_matches_force_leaf": abs(
                results["force_leaf"]["nll"] - native
            ) < 1e-5,
            "force_root_damage": results["force_root"]["nll"] - native,
            "identity_damage": results["runtime_identity"]["nll"] - native,
            "pair_break_depth_0_damage": (
                results["pair_break_depth_0"]["nll"] - native
            ),
            "uniform_branch_damage": results["uniform_branch"]["nll"] - native,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
