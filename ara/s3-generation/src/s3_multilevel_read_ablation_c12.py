#!/usr/bin/env python3
"""Strict matched-stream ablation of C10, multi-level READ, and READ + K_up."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-MULTILEVEL-READ-ABLATION-C12"
ARMS = ("c10", "read", "read_up")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(int(row[4]).to_bytes(8, "big"))
        digest.update(row[2].encode("ascii"))
        c10.update_stream_digest(digest, [row])
    return digest.hexdigest()


def selected_parameter_sha256(model, prefixes) -> str:
    state = {
        name: value
        for name, value in model.state_dict().items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    return c10.state_sha256(state)


def build_arm(config, vocab: int, pad: int, parent_state, arm: str, seed: int):
    # Rebuild the parent for every arm so no optimizer or runtime state leaks.
    base = c10.build_model(config, vocab, pad)
    base.load_state_dict(parent_state, strict=True)
    if arm == "c10":
        return base
    # Identical seed and complete module construction make READ initialization
    # bit-identical between the two structural arms.
    random.seed(seed + 12012)
    torch.manual_seed(seed + 12012)
    torch.cuda.manual_seed_all(seed + 12012)
    return c11.HStateConvolutionModel(
        base, config.dim, config.hidden, use_up=arm == "read_up",
    ).to(config.device)


def teacher(model, arm, source, length, target, bos, *, mode="native", ablate_depth=-1,
            intervention="native", pair_break_depth=-1, runtime_mode=None):
    previous = model.encoder.runtime_mode
    model.encoder.runtime_mode = runtime_mode
    try:
        if arm == "c10":
            route_mode = "force_leaf" if mode == "leaf_only" else "native"
            return model.teacher(
                source, length, target, bos,
                intervention=intervention, route_mode=route_mode,
                pair_break_depth=pair_break_depth,
            )
        return model.teacher(
            source, length, target, bos, mode=mode,
            ablate_depth=ablate_depth, intervention=intervention,
            pair_break_depth=pair_break_depth,
        )
    finally:
        model.encoder.runtime_mode = previous


@torch.no_grad()
def evaluate(model, arm, rows, pad, bos, device, batch_size, **kwargs):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    for start in range(0, len(rows), batch_size):
        source, length, target = c10.collate_rows(rows[start:start + batch_size], pad, device)
        logits, _ = teacher(model, arm, source, length, target, bos, **kwargs)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
    nll = loss_sum / max(1, tokens)
    return {"nll": nll, "ppl": math.exp(min(20.0, nll)), "tokens": tokens}


@torch.no_grad()
def generation_metrics(model, rows, args, sp, pad, bos, eos, pieces, limit=32):
    model.eval()
    hypotheses = []
    references = []
    examples = []
    adjacent_equal = 0
    adjacent_total = 0
    max_token_shares = []
    for row in rows[:limit]:
        source, length, target = c10.collate_rows([row], pad, args.device)
        generated, route = model.greedy(
            source, length, bos, eos, min(args.max_generation, target.shape[1] + 16),
        )
        hypothesis = c10.wmt.clean(generated[0].tolist(), eos, pieces)
        reference = c10.wmt.clean(target[0].tolist(), eos, pieces)
        hypotheses.append(hypothesis)
        references.append(reference)
        adjacent_equal += sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
        adjacent_total += max(0, len(hypothesis) - 1)
        counts = Counter(hypothesis)
        max_token_shares.append(max(counts.values()) / len(hypothesis) if hypothesis else 0.0)
        if len(examples) < 8:
            examples.append({
                "direction": row[2],
                "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                "reference": sp.decode(reference),
                "generation": sp.decode(hypothesis),
                "route": [float(value) for value in route.detach().cpu()],
            })
    return {
        "token_bleu4": c10.wmt_metrics.bleu4(hypotheses, references),
        "nonempty": sum(bool(row) for row in hypotheses) / max(1, len(hypotheses)),
        "adjacent_repetition_rate": adjacent_equal / max(1, adjacent_total),
        "mean_max_token_share": sum(max_token_shares) / max(1, len(max_token_shares)),
        "examples": examples,
    }


def atomic_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--max-generation", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        args.steps = args.steps or 60
        args.batch_size = args.batch_size or 8
        args.train_rows = args.train_rows or 512
        args.eval_rows = args.eval_rows or 64
        args.log_every = args.log_every or 20
    else:
        args.steps = args.steps or 25000
        args.batch_size = args.batch_size or 16
        args.train_rows = args.train_rows or 200000
        args.eval_rows = args.eval_rows or 1000
        args.log_every = args.log_every or 500

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"
    if trace_path.exists() and not args.resume:
        trace_path.unlink()

    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**parent["config"])
    config.device = args.device
    config.wmt_data = args.wmt_data
    config.task_train_rows = args.train_rows
    config.task_eval_rows = args.eval_rows
    config.max_wmt_scan_lines = 3_000_000
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    model = build_arm(config, vocab, pad, parent["state_dict"], args.arm, args.seed)
    parent_sha = c10.state_sha256(parent["state_dict"])
    if parent.get("state_sha256") and parent_sha != parent["state_sha256"]:
        raise RuntimeError("parent checkpoint state hash mismatch")

    train_rows, valid_rows, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    train_rows = train_rows[:args.train_rows]
    valid_rows = valid_rows[:args.eval_rows]
    test_rows = test_rows[:args.eval_rows]
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 21)
    stream_hash = c10.stream_sha256(schedule)
    row_hashes = {
        "train": rows_sha256(train_rows),
        "valid": rows_sha256(valid_rows),
        "test": rows_sha256(test_rows),
    }
    read_init_sha = None
    if args.arm != "c10":
        read_init_sha = selected_parameter_sha256(
            model, ("decoder.read_kernel.", "decoder.read_gain_logit"),
        )

    initial_valid = evaluate(model, args.arm, valid_rows, pad, bos, args.device, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_nll = initial_valid["nll"]
    best_state = c10.cpu_state(model)
    progress_path = output / "checkpoint_progress.pt"
    start_step = 0
    train_loss_sum = 0.0
    train_tokens = 0
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress["stream_sha256"] != stream_hash or progress["arm"] != args.arm:
            raise RuntimeError("resume contract mismatch")
        model.load_state_dict(progress["state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(args.device)
        start_step = int(progress["step"])
        best_nll = float(progress["best_nll"])
        best_state = progress["best_state"]
        train_loss_sum = float(progress["train_loss_sum"])
        train_tokens = int(progress["train_tokens"])
        print(json.dumps({"event": "resume", "arm": args.arm, "step": start_step}), flush=True)

    started = time.time()
    for step, batch in enumerate(schedule, 1):
        if step <= start_step:
            continue
        model.train()
        source, length, target = c10.collate_rows(batch, pad, args.device)
        logits, route = teacher(model, args.arm, source, length, target, bos)
        tokens = int(target.ne(pad).sum())
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        )
        loss = loss_sum / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        train_loss_sum += float(loss_sum.detach())
        train_tokens += tokens
        if step == 1 or step == args.steps or step % args.log_every == 0:
            valid = evaluate(model, args.arm, valid_rows, pad, bos, args.device, args.batch_size)
            row = {
                "arm": args.arm, "step": step,
                "train_nll": train_loss_sum / max(1, train_tokens),
                "valid_nll": valid["nll"], "grad_norm": grad_norm,
                "route": [float(value) for value in route.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
            print(json.dumps(row), flush=True)
            if valid["nll"] < best_nll:
                best_nll = valid["nll"]
                best_state = c10.cpu_state(model)
            if args.mode == "formal":
                atomic_save(progress_path, {
                    "claim": CLAIM, "arm": args.arm, "step": step,
                    "stream_sha256": stream_hash,
                    "state_dict": c10.cpu_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_nll": best_nll, "best_state": best_state,
                    "train_loss_sum": train_loss_sum, "train_tokens": train_tokens,
                })

    model.load_state_dict(best_state, strict=True)
    final_valid = evaluate(model, args.arm, valid_rows, pad, bos, args.device, args.batch_size)
    final_test = evaluate(model, args.arm, test_rows, pad, bos, args.device, args.batch_size)
    interventions = {
        "native": final_test,
        "source_shuffle": evaluate(
            model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
            intervention="source_shuffle",
        ),
        "runtime_identity": evaluate(
            model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
            runtime_mode="identity",
        ),
        "pair_break_depth_0": evaluate(
            model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
            pair_break_depth=0,
        ),
    }
    if args.arm != "c10":
        interventions["leaf_only"] = evaluate(
            model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
            mode="leaf_only",
        )
        interventions["bypass_up"] = evaluate(
            model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
            mode="bypass_up",
        )
        interventions["ablate_depth"] = {}
        for depth in range(model.decoder.depth_embedding.num_embeddings):
            interventions["ablate_depth"][str(depth)] = evaluate(
                model, args.arm, test_rows, pad, bos, args.device, args.batch_size,
                ablate_depth=depth,
            )

    native_nll = final_test["nll"]
    deltas = {
        name: result["nll"] - native_nll
        for name, result in interventions.items()
        if name not in ("native", "ablate_depth")
    }
    depth_deltas = []
    if "ablate_depth" in interventions:
        depth_deltas = [
            interventions["ablate_depth"][str(depth)]["nll"] - native_nll
            for depth in range(len(interventions["ablate_depth"]))
        ]
    generation = generation_metrics(
        model, test_rows, args, sp, pad, bos, eos, pieces,
        limit=min(32, len(test_rows)),
    )
    final_state_sha = c10.state_sha256(best_state)
    summary = {
        "claim": CLAIM, "arm": args.arm, "mode": args.mode,
        "host": socket.gethostname(), "config": vars(args),
        "parent_state_sha256": parent_sha,
        "initial_read_state_sha256": read_init_sha,
        "final_state_sha256": final_state_sha,
        "rows": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "row_sha256": row_hashes, "stream_sha256": stream_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "initial_valid": initial_valid, "best_valid": final_valid, "test": final_test,
        "train_nll": train_loss_sum / max(1, train_tokens), "train_tokens": train_tokens,
        "interventions": interventions, "intervention_deltas": deltas,
        "depth_ablation_deltas": depth_deltas,
        "generation": generation, "seconds": time.time() - started,
        "finite": math.isfinite(final_test["nll"]),
    }
    write_json(output / "summary.json", summary)
    if args.mode == "formal":
        atomic_save(output / "checkpoint_best.pt", {
            "claim": CLAIM, "arm": args.arm, "state_dict": best_state,
            "state_sha256": final_state_sha, "config": vars(args),
            "parent_state_sha256": parent_sha, "stream_sha256": stream_hash,
        })
    print(json.dumps({
        "event": "complete", "arm": args.arm,
        "stream_sha256": stream_hash, "row_sha256": row_hashes,
        "initial_read_state_sha256": read_init_sha,
        "test_nll": final_test["nll"], "token_bleu4": generation["token_bleu4"],
        "adjacent_repetition_rate": generation["adjacent_repetition_rate"],
        "intervention_deltas": deltas, "depth_ablation_deltas": depth_deltas,
        "evidence": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
