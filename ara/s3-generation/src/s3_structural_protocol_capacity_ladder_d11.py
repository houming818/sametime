#!/usr/bin/env python3
"""Matched residual-TreeHeap capacity rung initialized from completed D10."""
from __future__ import annotations

import argparse
import copy
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
import s3_hstate_multilevel_convolution as hstate  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402
import s3_structural_slot_ownership_d08 as d08  # noqa: E402
import s3_structural_slot_ownership_d09_scale as d09  # noqa: E402


CLAIM = "S3-STRUCTURAL-PROTOCOL-CAPACITY-LADDER-D11"
DEPTHS = d07.DEPTHS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class DecoderSeed(nn.Module):
    """Independent decoder modules used by one added TreeHeap channel."""

    def __init__(self, vocab: int, dim: int, depths: int, pad: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=pad)
        self.query = nn.Linear(dim, dim, bias=False)
        self.cell = nn.GRUCell(2 * dim, dim)
        self.output = nn.Linear(2 * dim, vocab)
        self.branch = nn.Linear(dim, dim, bias=False)
        self.depth_embedding = nn.Embedding(depths, dim)


class ResidualTreeHeapChannel(nn.Module):
    """A complete additional TreeHeap protocol path, not a flat logit adapter."""

    def __init__(
        self, source_dim: int, dim: int, vocab: int, pad: int,
        max_slots: int, source_depths: int, ownership_seed: int,
    ):
        super().__init__()
        self.source_projection = nn.Linear(source_dim, dim, bias=False)
        self.compressor = d08.OwnershipCompressor(
            dim, max_slots, source_depths,
            ownership_mode="subheap", ownership_seed=ownership_seed,
        )
        seed = DecoderSeed(vocab, dim, int(math.log2(max_slots)) + 1, pad)
        self.reconstructor = hstate.MultiLevelConvolutionDecoder(
            seed, dim, dim, int(math.log2(max_slots)) + 1, use_up=True,
        )
        self.protocol_gain_logit = nn.Parameter(torch.tensor(-4.0))

    def protocol(self, source_tree, masks, depth: int, budgets, intervention: str):
        projected = [self.source_projection(nodes) for nodes in source_tree]
        slots, slot_mask, entropy = self.compressor(projected, masks, depth, budgets)
        slots = slots * torch.sigmoid(self.protocol_gain_logit)
        if intervention == "shuffle":
            if slots.shape[0] > 1:
                slots = slots.roll(1, dims=0)
                slot_mask = slot_mask.roll(1, dims=0)
        elif intervention == "zero":
            slots = torch.zeros_like(slots)
        elif intervention != "native":
            raise ValueError(intervention)
        tree, tree_masks = d07.fold_protocol(slots, slot_mask)
        return tree, tree_masks, slots, entropy


class CapacityExpandedTreeHeap(nn.Module):
    """D10 plus an exactly-zero-at-initialization residual TreeHeap channel."""

    def __init__(
        self, base, source_dim: int, vocab: int, pad: int,
        extra_dim: int, max_slots: int, ownership_seed: int,
    ):
        super().__init__()
        self.frozen_source = base.frozen_source
        self.compressor = base.compressor
        self.reconstructor = base.reconstructor
        self.max_slots = max_slots
        self.extra_dim = extra_dim
        self.extra = None
        if extra_dim > 0:
            source_depths = self.frozen_source.decoder.depth_embedding.num_embeddings
            self.extra = ResidualTreeHeapChannel(
                source_dim, extra_dim, vocab, pad, max_slots,
                source_depths, ownership_seed + 1000,
            )
            # tanh(0) makes the expanded model exactly equal to D10 at step zero.
            self.extra_logit_gain = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("extra_logit_gain", torch.tensor(0.0))
        self.last_extra_route_statistics = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_source.eval()
        return self

    def protocol(self, source, lengths, depth: int, intervention: str = "native"):
        with torch.no_grad():
            levels, masks = d03.condition_states(self.frozen_source, source, lengths, "native")
            source_tree = self.frozen_source.decoder.convolve(levels, masks)
        budgets = d06_depth_budgets(lengths, depth, self.max_slots)
        slots, slot_mask, entropy = self.compressor(source_tree, masks, depth, budgets)
        slots = slots * torch.sigmoid(self.reconstructor_gain())
        if intervention == "shuffle":
            if slots.shape[0] > 1:
                slots = slots.roll(1, dims=0)
                slot_mask = slot_mask.roll(1, dims=0)
        elif intervention == "zero":
            slots = torch.zeros_like(slots)
        elif intervention != "native":
            raise ValueError(intervention)
        base_tree, base_masks = d07.fold_protocol(slots, slot_mask)
        extra_payload = None
        if self.extra is not None:
            extra_payload = self.extra.protocol(
                source_tree, masks, depth, budgets, intervention,
            )
            self.last_extra_route_statistics = dict(self.extra.compressor.last_route_statistics)
        return base_tree, base_masks, budgets, slots, entropy, extra_payload

    def reconstructor_gain(self):
        # D10 stores this gain on the protocol model, not on its decoder. It is
        # copied onto this wrapper during construction.
        return self.protocol_gain_logit

    def teacher(self, source, lengths, target, bos: int, depth: int, intervention="native"):
        base_tree, base_masks, budgets, slots, entropy, extra = self.protocol(
            source, lengths, depth, intervention,
        )
        base_logits, route = self.reconstructor.teacher(base_tree, base_masks, target, bos)
        if extra is not None:
            extra_tree, extra_masks, _, _ = extra
            extra_logits, extra_route = self.extra.reconstructor.teacher(
                extra_tree, extra_masks, target, bos,
            )
            base_logits = base_logits + torch.tanh(self.extra_logit_gain) * extra_logits
            route = (route + extra_route) * 0.5
        return base_logits, route, budgets, slots, entropy

    @torch.no_grad()
    def greedy(self, source, lengths, bos: int, eos: int, max_len: int, depth: int):
        base_tree, base_masks, budgets, slots, entropy, extra = self.protocol(
            source, lengths, depth, "native",
        )
        base_tree = self.reconstructor.convolve(base_tree, base_masks)
        base_hidden = base_tree[0].new_zeros((base_tree[0].shape[0], self.reconstructor.hidden))
        extra_tree = extra_masks = extra_hidden = None
        if extra is not None:
            extra_tree, extra_masks, _, _ = extra
            extra_tree = self.extra.reconstructor.convolve(extra_tree, extra_masks)
            extra_hidden = extra_tree[0].new_zeros(
                (extra_tree[0].shape[0], self.extra.reconstructor.hidden)
            )
        previous = torch.full(
            (base_tree[0].shape[0],), bos, dtype=torch.long, device=base_tree[0].device,
        )
        done = torch.zeros_like(previous, dtype=torch.bool)
        outputs, routes = [], []
        for _ in range(max_len):
            context, base_route = self.reconstructor.read(
                base_hidden, base_tree, base_masks,
            )
            base_hidden = self.reconstructor.cell(
                torch.cat((self.reconstructor.embedding(previous), context), dim=-1),
                base_hidden,
            )
            logits = self.reconstructor.output(torch.cat((base_hidden, context), dim=-1))
            route = base_route
            if extra_tree is not None:
                extra_context, extra_route = self.extra.reconstructor.read(
                    extra_hidden, extra_tree, extra_masks,
                )
                extra_hidden = self.extra.reconstructor.cell(
                    torch.cat((self.extra.reconstructor.embedding(previous), extra_context), dim=-1),
                    extra_hidden,
                )
                extra_logits = self.extra.reconstructor.output(
                    torch.cat((extra_hidden, extra_context), dim=-1)
                )
                logits = logits + torch.tanh(self.extra_logit_gain) * extra_logits
                route = (route + extra_route) * 0.5
            previous = logits.argmax(-1)
            outputs.append(previous)
            routes.append(route)
            done |= previous.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(outputs, dim=1), torch.stack(routes).mean(0), budgets, slots, entropy


def d06_depth_budgets(lengths, depth: int, max_slots: int):
    # Keep this call isolated so the capacity runner cannot silently alter the
    # D10 depth-pressure contract.
    import s3_recursive_depth_length_pressure as d06
    return d06.depth_budgets(lengths, depth, 2, max_slots)


def trainable_state(model) -> dict[str, torch.Tensor]:
    return d08.trainable_state(model)


def trainable_hash(model) -> str:
    return c10.state_sha256(trainable_state(model))


def parameter_summary(model) -> dict:
    groups = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        row = groups.setdefault(group, {"total": 0, "trainable": 0})
        row["total"] += parameter.numel()
        if parameter.requires_grad:
            row["trainable"] += parameter.numel()
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "groups": groups,
    }


def make_model(source_cpu, config, args, warm_path: Path, checkpoint: dict | None = None):
    base, warm = d10.make_model(source_cpu, config, args, warm_path)
    model = CapacityExpandedTreeHeap(
        base, config.dim, args.vocab, args.pad, args.extra_dim,
        args.max_slots, args.ownership_seed,
    ).to(args.device)
    model.protocol_gain_logit = base.protocol_gain_logit
    if checkpoint is not None:
        incompatible = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
        if trainable_hash(model) != checkpoint["trainable_state_sha256"]:
            raise RuntimeError("D11 trainable-state reload hash mismatch")
    return model, warm


def save_checkpoint(path: Path, model, optimizer, step: int, cursor: int, run: dict) -> str:
    state = trainable_state(model)
    digest = c10.state_sha256(state)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "claim": CLAIM, "arm": run["arm"], "step": step, "cursor": cursor,
        "trainable_state_dict": state, "trainable_state_sha256": digest,
        "optimizer_state_dict": optimizer.state_dict(), "run": run,
    }, temporary)
    os.replace(temporary, path)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=("treeheap-63m", "treeheap-106m"), required=True)
    parser.add_argument("--extra-dim", type=int, required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--parallel-data", default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv")
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11101)
    parser.add_argument("--ownership-seed", type=int, default=11102)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--generation-examples", type=int, default=64)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--max-lines", type=int, default=d10.PARALLEL_ROWS)
    parser.add_argument("--wake-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()
    if (args.arm == "treeheap-63m") != (args.extra_dim == 0):
        raise ValueError("treeheap-63m requires extra_dim=0; treeheap-106m requires extra_dim>0")
    if args.mode == "smoke":
        args.max_steps = min(args.max_steps, 50)
        args.max_lines = min(args.max_lines, 4096)
        args.eval_rows = min(args.eval_rows, 32)
        args.generation_examples = min(args.generation_examples, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.wake_every = min(args.wake_every, 25)
        args.log_every = min(args.log_every, 10)

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
    model, warm = make_model(source_cpu, config, args, Path(args.warm_start))
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    valid_rows, test_rows, excluded = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )
    parameters = parameter_summary(model)
    initial_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    initial_generation = d10.generation_summary(
        model, test_rows, args, sp, args.pad, bos, eos, pieces,
    )
    initial_hash = trainable_hash(model)
    run = {
        "claim": CLAIM, "mode": args.mode, "arm": args.arm,
        "extra_dim": args.extra_dim, "seed": args.seed,
        "source_sha256": source_hash,
        "warm_start_sha256": warm["trainable_state_sha256"],
        "parallel_sha256": d10.PARALLEL_SHA256,
        "parameters": parameters, "config": vars(args),
        "initial_valid": initial_valid, "initial_generation": initial_generation,
        "initial_trainable_sha256": initial_hash,
    }
    write_json(output / "contract.json", run)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )
    latest = output / "checkpoint_latest.pt"
    trace = output / "trace.jsonl"
    started = time.time()
    step = cursor = processed_tokens = 0
    best_nll = initial_valid["mean_nll"]
    best_step = 0
    best_path = output / "checkpoint_best.pt"
    save_checkpoint(best_path, model, optimizer, step, cursor, run)
    stage_counts = {}
    iterator = d10.iter_parallel_batches(
        Path(args.parallel_data), sp, direction_ids, eos, args.batch_size,
        0, args.max_lines, excluded,
    )
    for cursor, batch, stage_counts in iterator:
        step += 1
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        model.train()
        source, lengths, target = c10.collate_rows(batch, args.pad, args.device)
        logits, route, _, slots, entropy = model.teacher(
            source, lengths, target, bos, depth,
        )
        tokens = int(target.ne(args.pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=args.pad, reduction="sum",
        ) / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not d07.finite_trainable_gradients(model):
            raise RuntimeError(f"non-finite gradient at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        ))
        optimizer.step()
        processed_tokens += tokens
        if step == 1 or step % args.log_every == 0:
            event = {
                "event": "train", "step": step, "cursor": cursor,
                "depth": depth, "loss": float(loss.detach()),
                "grad_norm": grad_norm, "tokens": processed_tokens,
                "slot_variance": float(slots.detach().var()),
                "extra_logit_gain": float(torch.tanh(model.extra_logit_gain).detach()),
                "base_route": model.compressor.last_route_statistics,
                "extra_route": model.last_extra_route_statistics,
                "entropy": [float(x) for x in entropy.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if step % args.wake_every == 0 or step >= args.max_steps:
            valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
            generation = d10.generation_summary(
                model, test_rows, args, sp, args.pad, bos, eos, pieces,
            )
            if valid["mean_nll"] < best_nll:
                best_nll, best_step = valid["mean_nll"], step
                save_checkpoint(best_path, model, optimizer, step, cursor, run)
            digest = save_checkpoint(latest, model, optimizer, step, cursor, run)
            wake = {
                "event": "wake", "step": step, "cursor": cursor,
                "valid": valid, "generation": generation,
                "best_nll": best_nll, "best_step": best_step,
                "checkpoint_sha256": digest,
                "extra_logit_gain": float(torch.tanh(model.extra_logit_gain).detach()),
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "wakes.jsonl", wake)
            write_json(output / "wake_latest.json", wake)
            print(json.dumps(wake, ensure_ascii=False), flush=True)
        if step >= args.max_steps:
            break

    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model, _ = make_model(source_cpu, config, args, Path(args.warm_start), best)
    final_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    final_generation = d10.generation_summary(
        model, test_rows, args, sp, args.pad, bos, eos, pieces,
    )
    causal = d10.causal_summary(model, test_rows, args, args.pad, bos)
    source_after = c10.state_sha256(model.frozen_source.state_dict())
    reload_model, _ = make_model(source_cpu, config, args, Path(args.warm_start), best)
    reload_valid = d10.valid_summary(
        reload_model, valid_rows[:32], args, args.pad, bos,
    )["mean_nll"]
    reference_valid = d10.valid_summary(
        model, valid_rows[:32], args, args.pad, bos,
    )["mean_nll"]
    causal_depths = sum(
        row["shuffle_delta"] >= 0.10 and row["zero_delta"] >= 0.10
        for row in causal.values()
    )
    structure = all(
        row["native"]["route"]["owner_leaf_coverage"] == 1.0
        and row["native"]["route"]["argmax_coverage"] >= 0.999
        and row["native"]["route"]["route_pair_overlap"] < 0.05
        for row in causal.values()
    )
    gates = {
        "steps_complete": step == args.max_steps,
        "finite": math.isfinite(final_valid["mean_nll"]),
        "input_causality": causal_depths >= 2,
        "structure": structure,
        "nonempty": final_generation["nonempty_min"] == 1.0,
        "repetition": final_generation["repetition_max"] <= 0.10,
        "source_frozen": source_hash == source_after,
        "trainable_updated": initial_hash != latest_payload["trainable_state_sha256"],
        "reload": abs(reference_valid - reload_valid) < 1e-9,
    }
    summary = {
        "claim": CLAIM, "mode": args.mode, "arm": args.arm,
        "host": socket.gethostname(), "parameters": parameters,
        "step": step, "cursor": cursor, "processed_tokens": processed_tokens,
        "initial_valid": initial_valid, "best_valid": final_valid,
        "initial_generation": initial_generation,
        "best_generation": final_generation, "causal": causal,
        "extra_logit_gain": float(torch.tanh(model.extra_logit_gain).detach()),
        "gates": gates, "passed": all(gates.values()),
        "reload_nll_delta": abs(reference_valid - reload_valid),
        "stage_counts": stage_counts, "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "arm": args.arm, "gates": gates}, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
