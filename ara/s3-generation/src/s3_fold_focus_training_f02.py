#!/usr/bin/env python3
"""Train only depth-conditioned FOLD focus on a frozen TreeHeap checkpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402
from treeheap_epoch_translate_cli import DEPTHS, load_runtime  # noqa: E402
from treeheap_fold_matrix_decode_probe import (  # noqa: E402
    adjacent_repetition,
    character_f2,
    load_specimens,
    scaled_fold,
)


CLAIM = "S3-FOLD-FOCUS-TRAINING-F02"
NATIVE_SCALE = math.sqrt(0.5)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@contextmanager
def tensor_fold_scale(scale: torch.Tensor):
    original = d07.fold_protocol
    d07.fold_protocol = lambda slots, mask: scaled_fold(slots, mask, scale)
    try:
        yield
    finally:
        d07.fold_protocol = original


class FocusModel(nn.Module):
    """Frozen TreeHeap plus one bounded focus scalar for each exposed depth."""

    def __init__(self, base: nn.Module, initial_scale: float = NATIVE_SCALE):
        super().__init__()
        if not math.isclose(initial_scale, NATIVE_SCALE, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("F02 requires the native scale as its exact origin")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.focus_coordinates = nn.Parameter(torch.zeros(len(DEPTHS)))
        self.base.eval()

    @property
    def compressor(self):
        return self.base.compressor

    def scales(self) -> torch.Tensor:
        coordinate = torch.tanh(self.focus_coordinates)
        displacement = torch.where(
            coordinate >= 0.0,
            (1.0 - NATIVE_SCALE) * coordinate,
            NATIVE_SCALE * coordinate,
        )
        return NATIVE_SCALE + displacement

    def scale(self, depth: int) -> torch.Tensor:
        return self.scales()[DEPTHS.index(depth)]

    def train(self, mode: bool = True):
        self.training = mode
        self.base.eval()
        return self

    def teacher(self, source, lengths, target, bos: int, depth: int, intervention="native"):
        with tensor_fold_scale(self.scale(depth)):
            return self.base.teacher(source, lengths, target, bos, depth, intervention)

    @torch.no_grad()
    def greedy(self, source, lengths, bos: int, eos: int, max_len: int, depth: int):
        with tensor_fold_scale(self.scale(depth)):
            return self.base.greedy(source, lengths, bos, eos, max_len, depth)


def base_checkpoint_state(payload: dict, base: nn.Module) -> dict[str, torch.Tensor]:
    current = base.state_dict()
    return {
        name: current[name].detach().cpu()
        for name in payload["trainable_state_dict"]
    }


@torch.no_grad()
def direct_decode_summary(model, specimens, args, sp, pieces: int, eos: int, bos: int):
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    rows = []
    for specimen in specimens:
        raw = sp.encode(specimen["source"].strip(), out_type=int)
        source_ids = [direction_ids[specimen["direction"]], *raw[:32], eos]
        source = torch.tensor([source_ids], dtype=torch.long, device=args.device)
        lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)
        reference = specimen.get("reference", "")
        reference_pieces = len(sp.encode(reference, out_type=int)) if reference else 0
        for depth in DEPTHS:
            generated, route, budgets, _, _ = model.greedy(
                source, lengths, bos, eos, args.max_generation, depth
            )
            raw_output = generated[0].tolist()
            eos_reached = eos in raw_output
            clean = d10.wmt.clean(raw_output, eos, pieces)
            output = sp.decode(clean)
            length_ratio = len(clean) / max(1, reference_pieces)
            repetition = adjacent_repetition(clean)
            formed = (
                bool(clean)
                and eos_reached
                and repetition <= 0.2
                and 0.2 <= length_ratio <= 2.5
            )
            rows.append({
                "specimen_id": specimen["id"],
                "category": specimen["category"],
                "direction": specimen["direction"],
                "depth": depth,
                "scale": float(model.scale(depth).detach().cpu()) if isinstance(model, FocusModel) else NATIVE_SCALE,
                "source": specimen["source"],
                "reference": reference,
                "generation": output,
                "output_pieces": len(clean),
                "reference_pieces": reference_pieces,
                "length_ratio": length_ratio,
                "eos_reached": eos_reached,
                "adjacent_repetition_rate": repetition,
                "character_f2_diagnostic": character_f2(output, reference),
                "formed": formed,
                "budget": int(budgets[0]),
                "route_depth_mass": [float(value) for value in route.detach().cpu()],
            })
    per_depth = {}
    for depth in DEPTHS:
        group = [row for row in rows if row["depth"] == depth]
        per_depth[str(depth)] = {
            "scale": group[0]["scale"],
            "formed_rate": sum(row["formed"] for row in group) / len(group),
            "eos_rate": sum(row["eos_reached"] for row in group) / len(group),
            "mean_output_pieces": sum(row["output_pieces"] for row in group) / len(group),
            "mean_character_f2_diagnostic": sum(row["character_f2_diagnostic"] for row in group) / len(group),
            "mean_repetition": sum(row["adjacent_repetition_rate"] for row in group) / len(group),
        }
    return {
        "count": len(rows),
        "per_depth": per_depth,
        "formed_rate": sum(row["formed"] for row in rows) / len(rows),
        "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows),
        "mean_character_f2_diagnostic": sum(row["character_f2_diagnostic"] for row in rows) / len(rows),
        "rows": rows,
    }


def save_focus_checkpoint(path: Path, model, optimizer, step: int, cursor: int, run: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "claim": CLAIM,
        "step": step,
        "cursor": cursor,
        "focus_coordinates": model.focus_coordinates.detach().cpu(),
        "focus_scales": [float(value) for value in model.scales().detach().cpu()],
        "optimizer_state_dict": optimizer.state_dict(),
        "run": run,
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--specimens", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--parallel-data", default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv")
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11401)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--generation-examples", type=int, default=32)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--max-lines", type=int, default=500000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--wake-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        args.max_steps = min(args.max_steps, 30)
        args.max_lines = min(args.max_lines, 4096)
        args.eval_rows = min(args.eval_rows, 32)
        args.generation_examples = min(args.generation_examples, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.wake_every = min(args.wake_every, 10)
        args.log_every = min(args.log_every, 5)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = load_runtime(Path(args.checkpoint), args.device)
    payload, base_args, sp, base, pieces, eos, bos, source_hash = runtime
    args.spm_model = base_args.spm_model
    args.pad, args.vocab = pieces, pieces + 3
    args.wmt_data = args.eval_wmt_data
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    specimens = load_specimens(args.specimens, None)
    if args.mode == "smoke":
        specimens = specimens[:2]

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    valid_rows, test_rows, excluded = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows
    )
    expected_base_hash = payload["trainable_state_sha256"]
    actual_base_hash = c10.state_sha256(base_checkpoint_state(payload, base))
    if actual_base_hash != expected_base_hash:
        raise RuntimeError("loaded base state does not match checkpoint hash")

    native_valid = d10.valid_summary(base, valid_rows, args, args.pad, bos)
    native_direct = direct_decode_summary(base, specimens, args, sp, pieces, eos, bos)
    model = FocusModel(base).to(args.device)
    focus_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    focus_direct = direct_decode_summary(model, specimens, args, sp, pieces, eos, bos)
    step0_nll_delta = abs(native_valid["mean_nll"] - focus_valid["mean_nll"])
    step0_text_exact = [row["generation"] for row in native_direct["rows"]] == [
        row["generation"] for row in focus_direct["rows"]
    ]

    optimizer = torch.optim.AdamW([model.focus_coordinates], lr=args.lr, weight_decay=0.0)
    latest = output / "focus_checkpoint_latest.pt"
    start_step = start_cursor = 0
    if args.resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("claim") != CLAIM:
            raise RuntimeError("resume claim mismatch")
        model.focus_coordinates.data.copy_(saved["focus_coordinates"].to(args.device))
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step, start_cursor = int(saved["step"]), int(saved["cursor"])

    run = {
        "claim": CLAIM,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_state_sha256": expected_base_hash,
        "source_sha256": source_hash,
        "specimens": str(Path(args.specimens)),
        "native_scale": NATIVE_SCALE,
        "trainable_parameters": 3,
        "frozen_parameters": sum(parameter.numel() for parameter in base.parameters()),
        "config": vars(args),
        "step0": {
            "native_valid": native_valid,
            "focus_valid": focus_valid,
            "nll_delta": step0_nll_delta,
            "direct_text_exact": step0_text_exact,
            "direct": focus_direct,
        },
    }
    if not (output / "contract.json").is_file():
        write_json(output / "contract.json", run)

    started = time.time()
    step, cursor = start_step, start_cursor
    finite = True
    nonzero_gradient_depths = set()
    iterator = d10.iter_parallel_batches(
        Path(args.parallel_data), sp, direction_ids, eos, args.batch_size,
        start_cursor, args.max_lines, excluded,
    )
    for cursor, batch, _ in iterator:
        if step >= args.max_steps:
            break
        step += 1
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        model.train()
        source, lengths, target = c10.collate_rows(batch, args.pad, args.device)
        logits, _, _, _, _ = model.teacher(source, lengths, target, bos, depth)
        tokens = int(target.ne(args.pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=args.pad, reduction="sum",
        ) / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = model.focus_coordinates.grad.detach().clone()
        finite = finite and bool(torch.isfinite(loss)) and bool(torch.isfinite(gradient).all())
        for index, value in enumerate(gradient):
            if abs(float(value)) > 1e-12:
                nonzero_gradient_depths.add(DEPTHS[index])
        if not finite:
            raise RuntimeError(f"non-finite focus state at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([model.focus_coordinates], 1.0))
        optimizer.step()
        if step == start_step + 1 or step % args.log_every == 0:
            event = {
                "event": "train", "step": step, "cursor": cursor, "depth": depth,
                "loss": float(loss.detach()), "grad_norm": grad_norm,
                "focus_gradient": [float(value) for value in gradient.cpu()],
                "focus_scales": [float(value) for value in model.scales().detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "trace.jsonl", event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if step % args.wake_every == 0 or step >= args.max_steps:
            valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
            direct = direct_decode_summary(model, specimens, args, sp, pieces, eos, bos)
            wake = {
                "event": "wake", "step": step, "cursor": cursor,
                "focus_scales": [float(value) for value in model.scales().detach().cpu()],
                "valid": valid, "direct": direct,
                "nonzero_gradient_depths": sorted(nonzero_gradient_depths),
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "wakes.jsonl", wake)
            write_json(output / "wake_latest.json", wake)
            save_focus_checkpoint(latest, model, optimizer, step, cursor, run)
            print(json.dumps({
                "event": "wake", "step": step, "cursor": cursor,
                "focus_scales": wake["focus_scales"],
                "mean_nll": valid["mean_nll"],
                "formed_rate": direct["formed_rate"],
                "character_f2": direct["mean_character_f2_diagnostic"],
            }, ensure_ascii=False), flush=True)
        if step >= args.max_steps:
            break

    if step < args.max_steps:
        raise RuntimeError(f"training data ended at step {step} before {args.max_steps}")
    final_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    final_generation = d10.generation_summary(
        model, test_rows, args, sp, args.pad, bos, eos, pieces
    )
    final_direct = direct_decode_summary(model, specimens, args, sp, pieces, eos, bos)
    final_scales = [float(value) for value in model.scales().detach().cpu()]
    base_hash_after = c10.state_sha256(base_checkpoint_state(payload, model.base))
    reference_reload_nll = d10.valid_summary(
        model, valid_rows[: min(32, len(valid_rows))], args, args.pad, bos
    )["mean_nll"]
    save_focus_checkpoint(latest, model, optimizer, step, cursor, run)

    del model, base, runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_runtime = load_runtime(Path(args.checkpoint), args.device)
    reload_base = reload_runtime[3]
    reload_model = FocusModel(reload_base).to(args.device)
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    reload_model.focus_coordinates.data.copy_(saved["focus_coordinates"].to(args.device))
    reload_nll = d10.valid_summary(
        reload_model, valid_rows[: min(32, len(valid_rows))], args, args.pad, bos
    )["mean_nll"]
    reload_delta = abs(reference_reload_nll - reload_nll)

    gates = {
        "step0_function": step0_nll_delta <= 1e-9 and step0_text_exact,
        "finite": finite,
        "all_depths_received_gradient": nonzero_gradient_depths == set(DEPTHS),
        "base_frozen": base_hash_after == expected_base_hash,
        "step_complete": step == args.max_steps,
        "scale_in_open_unit_interval": all(0.0 < value < 1.0 for value in final_scales),
        "reload": reload_delta <= 1e-9,
    }
    summary = {
        "claim": CLAIM, "mode": args.mode, "host": socket.gethostname(),
        "step": step, "cursor": cursor, "focus_scales": final_scales,
        "initial_valid": focus_valid, "final_valid": final_valid,
        "nll_improvement": focus_valid["mean_nll"] - final_valid["mean_nll"],
        "initial_direct": focus_direct, "final_direct": final_direct,
        "final_generation": final_generation,
        "base_state_sha256_before": expected_base_hash,
        "base_state_sha256_after": base_hash_after,
        "reload_nll_delta": reload_delta,
        "gates": gates, "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete", "step": step, "focus_scales": final_scales,
        "nll_improvement": summary["nll_improvement"], "gates": gates,
    }, ensure_ascii=False), flush=True)
    if not all(gates.values()):
        raise SystemExit(5)


if __name__ == "__main__":
    main()
