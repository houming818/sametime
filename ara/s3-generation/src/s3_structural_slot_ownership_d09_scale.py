#!/usr/bin/env python3
"""Scale the supported D08R1 adjacent-subheap protocol with bounded wake gates."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_structural_slot_ownership_d08 as d08  # noqa: E402


CLAIM = "S3-STRUCTURAL-SLOT-OWNERSHIP-D09-SCALE"
DEPTHS = d07.DEPTHS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def trainable_hash(model) -> str:
    return c10.state_sha256(d08.trainable_state(model))


def load_trainable(model, payload) -> None:
    incompatible = model.load_state_dict(payload["trainable_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected warm-start keys: {incompatible.unexpected_keys}")
    if trainable_hash(model) != payload["trainable_state_sha256"]:
        raise RuntimeError("warm-start trainable hash mismatch")


def save_checkpoint(
    path: Path, model, optimizer, step: int, payload: dict, progress: dict | None = None,
) -> str:
    state = d08.trainable_state(model)
    state_hash = c10.state_sha256(state)
    torch.save({
        "claim": CLAIM,
        "step": step,
        "trainable_state_dict": state,
        "trainable_state_sha256": state_hash,
        "optimizer_state_dict": optimizer.state_dict(),
        "run": payload,
        "progress": progress or {},
    }, path)
    return state_hash


def generation_summary(model, rows, args, sp, pad, bos, eos, pieces) -> dict:
    per_depth = {
        str(depth): d07.generation_metrics(
            model, rows, args, sp, pad, bos, eos, pieces, depth,
        )
        for depth in DEPTHS
    }
    bleu = sorted(row["token_bleu4"] for row in per_depth.values())
    return {
        "per_depth": per_depth,
        "bleu4_median": bleu[len(bleu) // 2],
        "nonempty_min": min(row["nonempty_rate"] for row in per_depth.values()),
        "repetition_max": max(row["adjacent_repetition_rate"] for row in per_depth.values()),
    }


def valid_summary(model, rows, args, pad, bos) -> dict:
    per_depth = {
        str(depth): d08.evaluate(
            model, rows, pad, bos, args.device, args.batch_size, depth,
        )
        for depth in DEPTHS
    }
    return {
        "per_depth": per_depth,
        "mean_nll": sum(row["nll"] for row in per_depth.values()) / len(per_depth),
    }


def causal_summary(model, rows, args, pad, bos) -> dict:
    output = {}
    for depth in DEPTHS:
        native = d08.evaluate(model, rows, pad, bos, args.device, args.batch_size, depth)
        shuffle = d08.evaluate(
            model, rows, pad, bos, args.device, args.batch_size, depth, "shuffle",
        )
        zero = d08.evaluate(model, rows, pad, bos, args.device, args.batch_size, depth, "zero")
        output[str(depth)] = {
            "native": native,
            "shuffle": shuffle,
            "zero": zero,
            "shuffle_delta": shuffle["nll"] - native["nll"],
            "zero_delta": zero["nll"] - native["nll"],
        }
    return output


def make_model(frozen_cpu, config, args):
    frozen = copy.deepcopy(frozen_cpu).to(args.device)
    model = d08.OwnershipPressureProtocolModel(
        frozen, config.dim, config.hidden, args.max_slots,
        ownership_mode="subheap", ownership_seed=args.ownership_seed,
    ).to(args.device)
    return model


def self_test(output: Path) -> None:
    valid = torch.ones(1, 16, dtype=torch.bool)
    owner = d08.ownership_mask(valid, torch.tensor([8]), 32, "subheap", 1)
    assert owner is not None
    assert bool(owner[0, :8].any(0).all())
    assert not bool((owner[0, :8].sum(0) > 1).any())
    write_json(output / "self_test.json", {"claim": CLAIM, "passed": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--warm-start")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10901)
    parser.add_argument("--ownership-seed", type=int, default=10811 + 91)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=200000)
    parser.add_argument("--eval-rows", type=int, default=1000)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wake-every", type=int, default=2500)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(output)
        return
    if not args.source_checkpoint or not args.warm_start:
        parser.error("--source-checkpoint and --warm-start are required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad, vocab = pieces, pieces + 3
    frozen_cpu, _, config, source_hash, parent_hash = d03.load_model(
        Path(args.source_checkpoint), args, sp, pad, vocab,
    )
    train_rows, valid_rows, test_rows = d07.collect_rows(config, sp, pieces, eos, args)
    row_hashes = {
        "train": d07.rows_sha256(train_rows),
        "valid": d07.rows_sha256(valid_rows),
        "test": d07.rows_sha256(test_rows),
    }
    model = make_model(frozen_cpu, config, args)
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    if warm.get("claim") != "S3-STRUCTURAL-SLOT-OWNERSHIP-D08R1" or warm.get("arm") != "subheap":
        raise RuntimeError("warm-start contract mismatch")
    load_trainable(model, warm)
    warm_hash = trainable_hash(model)
    language_before = c10.state_sha256(d07.language_backbone_state(model))
    source_before = c10.state_sha256(model.frozen_source.state_dict())

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 1)
    depth_rng = random.Random(args.seed + 2)
    depth_schedule = []
    while len(depth_schedule) < args.steps:
        block = list(DEPTHS)
        depth_rng.shuffle(block)
        depth_schedule.extend(block)

    latest_path = output / "checkpoint_latest.pt"
    best_path = output / "checkpoint_best.pt"
    start_step = 0
    resume_progress = {}
    if args.resume and latest_path.exists():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if latest.get("claim") != CLAIM:
            raise RuntimeError("resume claim mismatch")
        if latest["run"]["row_sha256"] != row_hashes:
            raise RuntimeError("resume data hash mismatch")
        load_trainable(model, latest)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        start_step = int(latest["step"])
        resume_progress = latest.get("progress", {})

    if start_step:
        initial_valid = latest["run"]["initial_valid"]
        initial_generation = latest["run"]["initial_generation"]
        saved_best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_model = make_model(frozen_cpu, config, args)
        load_trainable(best_model, saved_best)
        best_nll = valid_summary(best_model, valid_rows, args, pad, bos)["mean_nll"]
        best_step = int(saved_best["step"])
        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        initial_valid = valid_summary(model, valid_rows, args, pad, bos)
        initial_generation = generation_summary(
            model, test_rows[:128], args, sp, pad, bos, eos, pieces,
        )
        best_nll = initial_valid["mean_nll"]
        best_step = 0
    stale_wakes = int(resume_progress.get("stale_wakes", 0))
    stopped_early = False
    run_contract = {
        "source_checkpoint": args.source_checkpoint,
        "warm_start": args.warm_start,
        "source_sha256": source_hash,
        "parent_sha256": parent_hash,
        "warm_trainable_sha256": warm_hash,
        "row_sha256": row_hashes,
        "initial_valid": initial_valid,
        "initial_generation": initial_generation,
        "config": vars(args),
    }
    if not best_path.exists() or start_step == 0:
        save_checkpoint(
            best_path, model, optimizer, start_step, run_contract,
            {"best_step": best_step, "best_nll": best_nll, "stale_wakes": stale_wakes},
        )
    trace_path = output / "trace.jsonl"
    if start_step == 0 and trace_path.exists():
        trace_path.unlink()
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
        batch = schedule[step - 1]
        depth = depth_schedule[step - 1]
        model.train()
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        logits, route, budgets, slots, entropy = model.teacher(source, lengths, target, bos, depth)
        token_count = int(target.ne(pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / max(1, token_count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not d07.finite_trainable_gradients(model):
            raise RuntimeError(f"non-finite gradient at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()

        if step == 1 or step % 500 == 0:
            row = {
                "event": "train", "step": step, "depth": depth,
                "train_nll": float(loss.detach()), "grad_norm": grad_norm,
                "mean_budget": float(budgets.float().mean()),
                "slot_variance": float(slots.detach().var()),
                "route": [float(value) for value in route.detach().cpu()],
                "slot_route": model.compressor.last_route_statistics,
                "entropy": [float(value) for value in entropy.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        if step % args.wake_every == 0 or step == args.steps:
            valid = valid_summary(model, valid_rows, args, pad, bos)
            generation = generation_summary(
                model, test_rows[:128], args, sp, pad, bos, eos, pieces,
            )
            improved = valid["mean_nll"] <= best_nll - args.min_delta
            if improved:
                best_nll = valid["mean_nll"]
                best_step = step
                stale_wakes = 0
                best_hash = save_checkpoint(
                    best_path, model, optimizer, step, run_contract,
                    {"best_step": best_step, "best_nll": best_nll, "stale_wakes": stale_wakes},
                )
            else:
                stale_wakes += 1
                best_hash = c10.state_sha256(
                    torch.load(best_path, map_location="cpu", weights_only=False)["trainable_state_dict"]
                )
            latest_hash = save_checkpoint(
                latest_path, model, optimizer, step, run_contract,
                {"best_step": best_step, "best_nll": best_nll, "stale_wakes": stale_wakes},
            )
            wake = {
                "event": "wake", "step": step, "valid": valid,
                "generation": generation, "improved": improved,
                "best_step": best_step, "best_mean_nll": best_nll,
                "stale_wakes": stale_wakes,
                "latest_trainable_sha256": latest_hash,
                "best_trainable_sha256": best_hash,
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "wakes.jsonl", wake)
            write_json(output / "wake_latest.json", wake)
            print(json.dumps(wake, ensure_ascii=False), flush=True)
            if step >= 10000 and stale_wakes >= args.patience:
                stopped_early = True
                break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trainable(model, best)
    final_valid = valid_summary(model, valid_rows, args, pad, bos)
    final_causal = causal_summary(model, test_rows, args, pad, bos)
    final_generation = generation_summary(
        model, test_rows, args, sp, pad, bos, eos, pieces,
    )
    source_after = c10.state_sha256(model.frozen_source.state_dict())
    language_after = c10.state_sha256(d07.language_backbone_state(model))

    reloaded = make_model(frozen_cpu, config, args)
    load_trainable(reloaded, best)
    reload_eval = valid_summary(reloaded, valid_rows[:128], args, pad, bos)
    reference_eval = valid_summary(model, valid_rows[:128], args, pad, bos)
    reload_delta = abs(reload_eval["mean_nll"] - reference_eval["mean_nll"])
    reload_hash_match = trainable_hash(reloaded) == trainable_hash(model) == best["trainable_state_sha256"]

    causal_passes = sum(
        row["shuffle_delta"] >= 0.10 and row["zero_delta"] >= 0.10
        for row in final_causal.values()
    )
    structure_pass = all(
        abs(row["native"]["route"]["owner_leaf_coverage"] - 1.0) <= 1e-6
        and row["native"]["route"]["argmax_coverage"] >= 0.999
        and row["native"]["route"]["route_pair_overlap"] < 0.05
        and row["native"]["slot_variance"] > 1e-4
        and row["native"]["between_slot_variance"] > 1e-8
        for row in final_causal.values()
    )
    gates = {
        "S0_contract": source_before == source_after == source_hash and language_before == language_after,
        "S1_scale_learning": initial_valid["mean_nll"] - final_valid["mean_nll"] >= 0.10,
        "S2_input_causality": causal_passes >= 2,
        "S3_structure_numeric": structure_pass,
        "S4_generation_trend": (
            final_generation["bleu4_median"] >= 5.0
            and final_generation["bleu4_median"] - initial_generation["bleu4_median"] >= 0.5
            and final_generation["nonempty_min"] == 1.0
            and final_generation["repetition_max"] <= 0.10
        ),
        "S5_reload": reload_hash_match and reload_delta < 1e-9,
    }
    decision = "scale_rung_supported" if all(gates.values()) else "scale_rung_not_supported"
    summary = {
        "claim": CLAIM, "decision": decision, "host": socket.gethostname(),
        "gates": gates, "best_step": best_step, "stopped_early": stopped_early,
        "rows": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "row_sha256": row_hashes, "initial_valid": initial_valid,
        "initial_generation": initial_generation, "best_valid": final_valid,
        "best_causal": final_causal, "best_generation": final_generation,
        "reload": {"hash_match": reload_hash_match, "mean_nll_delta": reload_delta},
        "hashes": {
            "source_before": source_before, "source_after": source_after,
            "language_before": language_before, "language_after": language_after,
            "warm_trainable": warm_hash, "best_trainable": best["trainable_state_sha256"],
        },
        "config": vars(args), "seconds": time.time() - started,
        "contracts": {
            "target_enters_compressor": False,
            "target_length_enters_budget": False,
            "source_frozen": True,
            "language_backbone_frozen": True,
            "ownership": "adjacent_recursive_subheap",
            "transformer_or_self_attention": False,
            "flat_length_route_table": False,
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "decision": decision, "gates": gates,
                      "best_step": best_step, "bleu4_median": final_generation["bleu4_median"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
