#!/usr/bin/env python3
"""Frozen held-out test audit for C11 multi-level TreeHeap convolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


def rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row[4]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes(str(row[2]), "utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--c10-baseline-checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-rows", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source", type=int, default=253)
    parser.add_argument("--max-target", type=int, default=253)
    args = parser.parse_args()

    started = time.time()
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    trained_payload = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    baseline_payload = torch.load(args.c10_baseline_checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**source_payload["config"])
    config.device = args.device
    config.wmt_data = args.wmt_data
    # The split and direction are deterministic per source line. Only one train
    # row is needed here; collecting 2x test rows preserves the formal filter.
    config.task_train_rows = 1
    config.task_eval_rows = max(args.test_rows * 2, args.test_rows + 128)
    config.max_wmt_scan_lines = 3_000_000

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3

    base = c10.build_model(config, vocab, pad)
    base.load_state_dict(source_payload["state_dict"], strict=True)
    model = c11.HStateConvolutionModel(base, config.dim, config.hidden).to(args.device)
    model.load_state_dict(trained_payload["state_dict"], strict=True)

    collected = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    test_rows = c11.clean_rows(collected[2], args.max_source, args.max_target)[:args.test_rows]
    if len(test_rows) != args.test_rows:
        raise RuntimeError(f"expected {args.test_rows} test rows, got {len(test_rows)}")

    def evaluate(*, mode="native", ablate_depth=-1, intervention="native", pair_break_depth=-1, runtime_mode=None):
        return c11.evaluate(
            model, test_rows, pad, bos, args.device, args.batch_size,
            mode, ablate_depth, intervention, pair_break_depth, runtime_mode,
        )

    native = evaluate()
    interventions = {
        "native": native,
        "bypass_up": evaluate(mode="bypass_up"),
        "leaf_only": evaluate(mode="leaf_only"),
        "source_shuffle": evaluate(intervention="source_shuffle"),
        "runtime_identity": evaluate(runtime_mode="identity"),
        "pair_break_depth_0": evaluate(pair_break_depth=0),
        "ablate_depth": {},
    }
    depth_count = model.decoder.depth_embedding.num_embeddings
    for depth in range(depth_count):
        interventions["ablate_depth"][str(depth)] = evaluate(ablate_depth=depth)

    deltas = {
        name: result["nll"] - native["nll"]
        for name, result in interventions.items()
        if name not in ("native", "ablate_depth")
    }
    depth_deltas = [
        interventions["ablate_depth"][str(depth)]["nll"] - native["nll"]
        for depth in range(depth_count)
    ]
    generation = c10.task_generation_metrics(
        model, test_rows,
        SimpleNamespace(device=args.device, max_generation=min(96, args.max_target + 16)),
        sp, pad, bos, eos, pieces, limit=min(32, len(test_rows)),
    )
    baseline = c10.build_model(config, vocab, pad).to(args.device)
    baseline.load_state_dict(baseline_payload["state_dict"], strict=True)
    baseline_test = c10.evaluate_nll(
        baseline, test_rows, pad, bos, args.device, args.batch_size,
    )
    baseline_generation = c10.task_generation_metrics(
        baseline, test_rows,
        SimpleNamespace(device=args.device, max_generation=min(96, args.max_target + 16)),
        sp, pad, bos, eos, pieces, limit=min(32, len(test_rows)),
    )
    threshold = baseline_test["nll"] + 0.10
    summary = {
        "claim": c11.CLAIM,
        "audit": "frozen_independent_test_split",
        "host": socket.gethostname(),
        "config": vars(args),
        "source_state_sha256": source_payload.get("state_sha256"),
        "trained_state_sha256": trained_payload.get("state_sha256"),
        "c10_baseline_state_sha256": baseline_payload.get("state_sha256"),
        "test_rows": len(test_rows),
        "test_rows_sha256": rows_sha256(test_rows),
        "native": native,
        "c10_baseline_same_rows": baseline_test,
        "delta_nll_vs_c10_same_rows": native["nll"] - baseline_test["nll"],
        "interventions": interventions,
        "intervention_deltas": deltas,
        "depth_ablation_deltas": depth_deltas,
        "generation": generation,
        "c10_baseline_generation_same_rows": baseline_generation,
        "p2": {
            "nll_within_c10_plus_0_10": native["nll"] <= threshold,
            "two_nonleaf_depths_helpful_0_01": sum(value >= 0.01 for value in depth_deltas[:-1]) >= 2,
            "bypass_up_helpful_0_02": deltas["bypass_up"] >= 0.02,
            "source_shuffle_causal": deltas["source_shuffle"] > 0.0,
            "runtime_identity_causal": deltas["runtime_identity"] > 0.0,
            "pair_break_causal": deltas["pair_break_depth_0"] > 0.0,
        },
        "seconds": time.time() - started,
    }
    summary["p2_pass"] = all(summary["p2"].values())
    write_json(Path(args.evidence_dir) / "summary.json", summary)
    print(json.dumps({
        "event": "complete",
        "test_nll": native["nll"],
        "c10_baseline_test_nll": baseline_test["nll"],
        "delta_nll_vs_c10_same_rows": native["nll"] - baseline_test["nll"],
        "intervention_deltas": deltas,
        "depth_ablation_deltas": depth_deltas,
        "token_bleu4": generation["token_bleu4"],
        "p2": summary["p2"],
        "p2_pass": summary["p2_pass"],
    }), flush=True)


if __name__ == "__main__":
    main()
