#!/usr/bin/env python3
"""Frozen CPU audit of what the full-corpus decoder reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_full_corpus_repair_seq2seq as full


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def swap_siblings(source: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    result = source.clone()
    for row, raw_length in enumerate(length.tolist()):
        width = int(raw_length)
        paired = width - width % 2
        if paired:
            result[row, :paired] = source[row, :paired].reshape(-1, 2).flip(1).reshape(-1)
    return result


def swap_halves(source: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    result = source.clone()
    for row, raw_length in enumerate(length.tolist()):
        width = int(raw_length)
        split = width // 2
        if split:
            result[row, :width] = torch.cat((source[row, split:width], source[row, :split]))
    return result


def encode_levels(model, source: torch.Tensor, length: torch.Tensor):
    leaf, root, details, fold_masks = model.encoder.fold(source, length)
    levels, level_masks = model.encoder.unfold(root, details, fold_masks)
    return levels, level_masks


def memory_at(model, levels, masks, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return levels[depth] + model.resolution.weight[depth], masks[depth]


def nll_sum(logits: torch.Tensor, target: torch.Tensor, pad: int) -> Tuple[float, int]:
    loss = F.cross_entropy(
        logits.flatten(0, 1), target.flatten(), ignore_index=pad, reduction="sum"
    )
    return float(loss), int(target.ne(pad).sum())


@torch.no_grad()
def audit_batch(
    model, source, length, target, bos: int, pad: int,
    depths: Sequence[int], structural_depths: Sequence[int],
):
    native_levels, native_masks = encode_levels(model, source, length)
    sibling_levels, sibling_masks = encode_levels(model, swap_siblings(source, length), length)
    half_levels, half_masks = encode_levels(model, swap_halves(source, length), length)
    rows: Dict[int, Dict[str, Tuple[float, int]]] = {}
    reverse_error: Dict[int, float] = {}

    for depth in depths:
        native, valid = memory_at(model, native_levels, native_masks, depth)
        conditions = {
            "native": (native, valid),
            "zero": (torch.zeros_like(native), valid),
        }
        if depth in structural_depths:
            sibling, sibling_valid = memory_at(model, sibling_levels, sibling_masks, depth)
            half, half_valid = memory_at(model, half_levels, half_masks, depth)
            conditions.update({
                "sample_swap": (native.roll(1, 0), valid.roll(1, 0)),
                "source_sibling_swap": (sibling, sibling_valid),
                "source_half_swap": (half, half_valid),
            })
        native_logits = model.decoder.teacher(native, valid, target, bos)
        rows[depth] = {"native": nll_sum(native_logits, target, pad)}
        for name, (memory, mask) in conditions.items():
            if name == "native":
                continue
            logits = model.decoder.teacher(memory, mask, target, bos)
            rows[depth][name] = nll_sum(logits, target, pad)

        reverse_logits = model.decoder.teacher(native.flip(1), valid.flip(1), target, bos)
        rows[depth]["node_reverse"] = nll_sum(reverse_logits, target, pad)
        reverse_error[depth] = float((reverse_logits - native_logits).abs().max())
    return rows, reverse_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default="/home/nio/datasets/treeheap_checkpoints/checkpoint_annealed.pt",
    )
    parser.add_argument("--data-root", default="/home/nio/datasets")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_full_decoder_causal_audit",
    )
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=72051)
    args = parser.parse_args()

    started = time.time()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = full.repair_probe.make_model(base_checkpoint, "cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    config = dict(checkpoint.get("config", {}))
    config.update(
        data_root=args.data_root,
        spm_model=args.spm_model,
        device="cpu",
        num_workers=0,
        eval_batch=args.eval_batch,
        batch=args.eval_batch,
        seed=args.seed,
    )
    run = SimpleNamespace(**config)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    run.pad = sp.get_piece_size()
    run.bos = sp.bos_id()
    run.eos = sp.eos_id()
    stream = iter(full.make_loader(run, "valid", args.seed + 9000, args.eval_batch))

    all_depths = list(range(model.depths + 1))
    tested_depths = sorted(set((0, model.depths // 2, model.depths)))
    total: Dict[int, Dict[str, dict]] = {}
    reverse_max: Dict[int, float] = {}
    example_sources: List[str] = []
    for batch_number in range(1, args.eval_batches + 1):
        source, length, target, _ = next(stream)
        if not example_sources:
            for row in range(min(4, source.shape[0])):
                ids = full.base.clean(source[row].tolist(), run.eos, run.pad)
                example_sources.append(sp.decode(ids))
        # Native and zero are measured at every resolution. The remaining
        # interventions are needed only at root, middle, and leaf.
        rows, maxima = audit_batch(
            model, source, length, target, run.bos, run.pad,
            all_depths, tested_depths,
        )
        for depth, conditions in rows.items():
            depth_total = total.setdefault(depth, {})
            for name, (loss, tokens) in conditions.items():
                item = depth_total.setdefault(name, {"loss": 0.0, "tokens": 0})
                item["loss"] += loss
                item["tokens"] += tokens
        for depth, value in maxima.items():
            reverse_max[depth] = max(reverse_max.get(depth, 0.0), value)
        print(json.dumps({"batch": batch_number, "elapsed_sec": time.time() - started}), flush=True)

    metrics = {}
    for depth in all_depths:
        conditions = total[depth]
        native = conditions["native"]["loss"] / conditions["native"]["tokens"]
        row = {"native_nll": native, "native_ppl": math.exp(min(20.0, native))}
        for name, item in conditions.items():
            nll = item["loss"] / item["tokens"]
            if name != "native":
                row[name] = {"nll": nll, "delta_nll": nll - native}
        row["node_reverse_max_abs_logit"] = reverse_max[depth]
        metrics[str(depth)] = row

    leaf = metrics[str(model.depths)]
    structural_deltas = []
    for depth in tested_depths[:-1]:
        for name in ("source_sibling_swap", "source_half_swap"):
            structural_deltas.append(metrics[str(depth)][name]["delta_nll"] - leaf[name]["delta_nll"])
    gates = {
        "P1_state_causality": max(
            leaf["zero"]["delta_nll"], leaf["sample_swap"]["delta_nll"]
        ) >= 0.20,
        "P2_permutation_invariance": all(
            metrics[str(depth)]["node_reverse_max_abs_logit"] <= 1e-5
            and abs(metrics[str(depth)]["node_reverse"]["delta_nll"]) <= 1e-6
            for depth in all_depths
        ),
        "P3_fold_structure_sensitivity": max(structural_deltas) >= 0.02,
    }
    conclusion = (
        "content_and_fold_causal" if gates["P1_state_causality"] and gates["P3_fold_structure_sensitivity"]
        else "content_only_or_flat" if gates["P1_state_causality"]
        else "decoder_prefix_dominant"
    )
    summary = {
        "claim_id": "S3-FULL-DECODER-CAUSAL-C01",
        "status": "completed_single_checkpoint_cpu_audit",
        "conclusion": conclusion,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "git_head": git_head(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_model_digest": full.repair_probe.model_digest(model),
        "config": vars(args),
        "model_depths": model.depths,
        "tested_structural_depths": tested_depths,
        "metrics_root_to_leaf": metrics,
        "gates": gates,
        "examples": example_sources,
        "elapsed_sec": time.time() - started,
        "boundary": (
            "Frozen teacher-forced mechanism audit only; no quality, superiority, "
            "semantic, world-model, or consciousness claim."
        ),
    }
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# S3 full decoder causal audit\n\n"
        f"Claim: `S3-FULL-DECODER-CAUSAL-C01`\n\n"
        f"Conclusion: **{conclusion}**\n\n"
        f"Gates: `{json.dumps(gates, ensure_ascii=False)}`\n\n"
        "See `summary.json` for per-depth intervention metrics.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
