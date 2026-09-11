#!/usr/bin/env python3
"""Read-only expanded evaluation for F13 learned grouped directions."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import torch

import s3_filter_guided_theta_calibration_f04 as f04
import s3_pretrain_task_posterior_pipeline as c10
import s3_structural_protocol_full_pipeline_d10 as d10
import s3_treeheap_learned_polytope_router_f13 as f13
import s3_wmt_treeheap_seq2seq as wmt_metrics
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13-R1"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_theta(path: Path, extra_dim: int, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    theta = f13.LearnedGroupedTheta(
        extra_dim,
        int(config["rank"]),
        int(config["groups"]),
        scalarized=payload["arm"] == "scalarized",
    ).to(device)
    theta.load_state_dict(payload["state_dict"])
    theta.eval()
    return theta, payload


def route_context(theta):
    return nullcontext() if theta is None else f04.routed_fold(theta)


@torch.inference_mode()
def collect_generation(model, theta, rows, args, sp, pad, bos, eos, pieces):
    per_depth = {}
    with route_context(theta):
        for depth in DEPTHS:
            hypotheses, references, examples = [], [], []
            adjacent_equal = adjacent_total = 0
            for row in rows:
                source, lengths, target = c10.collate_rows([row], pad, args.device)
                generated, _, _, _, _ = model.greedy(
                    source, lengths, bos, eos, args.max_generation, depth,
                )
                hypothesis = d10.wmt.clean(generated[0].tolist(), eos, pieces)
                reference = d10.wmt.clean(target[0].tolist(), eos, pieces)
                hypotheses.append(hypothesis)
                references.append(reference)
                adjacent_equal += sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
                adjacent_total += max(0, len(hypothesis) - 1)
                if len(examples) < 12:
                    examples.append({
                        "kind": row[2],
                        "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                        "reference": sp.decode(reference),
                        "generation": sp.decode(hypothesis),
                    })
            per_depth[str(depth)] = {
                "hypotheses": hypotheses,
                "references": references,
                "bleu4": wmt_metrics.bleu4(hypotheses, references),
                "nonempty_rate": sum(bool(row) for row in hypotheses) / len(hypotheses),
                "repetition_rate": adjacent_equal / max(1, adjacent_total),
                "examples": examples,
            }
    medians = sorted(row["bleu4"] for row in per_depth.values())
    return {
        "per_depth": per_depth,
        "bleu4_median": medians[len(medians) // 2],
        "nonempty_min": min(row["nonempty_rate"] for row in per_depth.values()),
        "repetition_max": max(row["repetition_rate"] for row in per_depth.values()),
    }


def block_summaries(arms: dict, rows: int, block_size: int) -> list[dict]:
    blocks = []
    for start in range(0, rows, block_size):
        end = start + block_size
        block = {"block": start // block_size, "start": start, "end": end, "arms": {}}
        for arm, report in arms.items():
            by_depth = {}
            for depth in map(str, DEPTHS):
                row = report["generation"]["per_depth"][depth]
                by_depth[depth] = wmt_metrics.bleu4(
                    row["hypotheses"][start:end], row["references"][start:end],
                )
            ordered = sorted(by_depth.values())
            block["arms"][arm] = {
                "per_depth_bleu4": by_depth,
                "bleu4_median": ordered[len(ordered) // 2],
            }
        block["grouped_minus_scalar_bleu4_median"] = (
            block["arms"]["grouped"]["bleu4_median"]
            - block["arms"]["scalarized"]["bleu4_median"]
        )
        blocks.append(block)
    return blocks


def evaluate_nll(model, theta, rows, args, pad, bos):
    with route_context(theta):
        return d10.valid_summary(model, rows, args, pad, bos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scalar-theta", required=True)
    parser.add_argument("--grouped-theta", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-pairs", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--max-generation", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    scalar_path = Path(args.scalar_theta)
    grouped_path = Path(args.grouped_theta)
    input_hashes = {
        "checkpoint": sha256(checkpoint),
        "scalar_theta": sha256(scalar_path),
        "grouped_theta": sha256(grouped_path),
    }

    payload, base_args, sp, model, pieces, eos, bos, source_hash = load_runtime(
        checkpoint, args.device,
    )
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    scalar, scalar_payload = load_theta(scalar_path, model.extra_dim, args.device)
    grouped, grouped_payload = load_theta(grouped_path, model.extra_dim, args.device)
    base_args.device = args.device
    base_args.eval_batch = args.eval_batch
    base_args.max_generation = args.max_generation
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    valid_rows, test_rows, pairs = d10.collect_wmt_eval(
        Path(args.wmt_data), sp, direction_ids, eos, args.eval_pairs,
    )
    if len(test_rows) % args.block_size:
        raise ValueError("test row count must be divisible by block size")

    arms = {}
    for name, theta in (("native", None), ("scalarized", scalar), ("grouped", grouped)):
        arms[name] = {
            "valid": evaluate_nll(model, theta, valid_rows, base_args, pieces, bos),
            "test": evaluate_nll(model, theta, test_rows, base_args, pieces, bos),
            "generation": collect_generation(
                model, theta, test_rows, base_args, sp, pieces, bos, eos, pieces,
            ),
        }
        print(json.dumps({
            "event": "arm_complete",
            "arm": name,
            "test_nll": arms[name]["test"]["mean_nll"],
            "bleu4_median": arms[name]["generation"]["bleu4_median"],
        }, ensure_ascii=False), flush=True)

    blocks = block_summaries(arms, len(test_rows), args.block_size)
    grouped_wins = sum(row["grouped_minus_scalar_bleu4_median"] > 0 for row in blocks)
    output_hashes = {
        "checkpoint": sha256(checkpoint),
        "scalar_theta": sha256(scalar_path),
        "grouped_theta": sha256(grouped_path),
    }
    scalar_generation = arms["scalarized"]["generation"]
    grouped_generation = arms["grouped"]["generation"]
    finite = all(
        math.isfinite(value)
        for arm in arms.values()
        for value in (arm["valid"]["mean_nll"], arm["test"]["mean_nll"], arm["generation"]["bleu4_median"])
    )
    gates = {
        "O0_reload_and_finite": finite,
        "O1_frozen_and_hash_stable": f04.frozen_exact(model, expected_model) and input_hashes == output_hashes,
        "P1_full_bleu_advantage": grouped_generation["bleu4_median"] > scalar_generation["bleu4_median"],
        "P2_block_majority": grouped_wins >= 5,
        "P3_generation_health": (
            grouped_generation["nonempty_min"] >= scalar_generation["nonempty_min"] - 0.02
            and grouped_generation["repetition_max"] <= scalar_generation["repetition_max"] + 0.02
        ),
    }
    summary = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "config": vars(args),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "source_sha256": source_hash,
        "scalar_arm": scalar_payload["arm"],
        "grouped_arm": grouped_payload["arm"],
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "pairs": len(pairs),
        "arms": arms,
        "blocks": blocks,
        "grouped_block_wins": grouped_wins,
        "gates": gates,
        "seconds": time.time() - started,
        "claim_boundary": "read-only F13 expanded evaluation; no retraining or FOLD promotion",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "contract.json", {
        "claim": CLAIM,
        "input_hashes": input_hashes,
        "eval_pairs": args.eval_pairs,
        "test_rows": len(test_rows),
        "block_size": args.block_size,
        "blocks": len(blocks),
    })
    (output / "README.md").write_text(
        "# F13-R1 expanded evaluation\n\n"
        f"- Gates: {json.dumps(gates, ensure_ascii=False)}\n"
        f"- Grouped block wins: {grouped_wins}/{len(blocks)}\n"
        f"- Runtime seconds: {summary['seconds']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "grouped_block_wins": grouped_wins,
        "full_bleu_delta": grouped_generation["bleu4_median"] - scalar_generation["bleu4_median"],
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)
    if not gates["O0_reload_and_finite"] or not gates["O1_frozen_and_hash_stable"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
