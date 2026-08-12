#!/usr/bin/env python3
"""C11 multi-level TreeHeap convolution over a complete H_state."""
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
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_treeheap_butterfly_bilingual_full as wmt  # noqa: E402


CLAIM = "S3-HSTATE-MULTILEVEL-CONV-C11"


def json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def state_sha256(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SharedUpKernel(nn.Module):
    """One parent-left-right kernel shared by every TreeHeap depth."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, parent, left, right):
        return torch.tanh(self.net(torch.cat((parent, left, right), dim=-1)))


class SharedReadKernel(nn.Module):
    """One residual query kernel shared by every root-to-leaf depth."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, query, local, depth):
        return torch.tanh(self.net(torch.cat((query, local, depth), dim=-1)))


class MultiLevelConvolutionDecoder(nn.Module):
    def __init__(self, old_decoder: nn.Module, dim: int, hidden: int, depths: int):
        super().__init__()
        self.embedding = old_decoder.embedding
        self.query = old_decoder.query
        self.cell = old_decoder.cell
        self.output = old_decoder.output
        self.up_kernel = SharedUpKernel(dim)
        self.read_kernel = SharedReadKernel(dim)
        self.branch = old_decoder.branch
        self.depth_embedding = old_decoder.depth_embedding
        # Start as a small residual correction to the inherited C10 readout.
        # The formal test must measure learned structure, not recovery from a
        # randomly destroyed checkpoint interface.
        self.up_gain_logit = nn.Parameter(torch.tensor(-4.0))
        self.read_gain_logit = nn.Parameter(torch.tensor(-4.0))
        self.hidden = hidden
        self.depths = depths

    def convolve(self, levels, masks, bypass_up: bool = False):
        tree = list(levels)
        if bypass_up:
            return tree
        gain = torch.sigmoid(self.up_gain_logit)
        for depth in range(len(tree) - 2, -1, -1):
            children = tree[depth + 1].reshape(
                tree[depth].shape[0], tree[depth].shape[1], 2, tree[depth].shape[2],
            )
            left, right = children[:, :, 0], children[:, :, 1]
            update = self.up_kernel(tree[depth], left, right)
            tree[depth] = tree[depth] + gain * update
            tree[depth] = tree[depth] * masks[depth][:, :, None]
        return tree

    def read(self, hidden, tree, masks, mode: str = "native", ablate_depth: int = -1):
        base_query = self.query(hidden)
        query = base_query
        frontier = masks[0].to(query.dtype)
        entropies = []
        gain = torch.sigmoid(self.read_gain_logit)
        last_depth = len(tree) - 1
        for depth, (nodes, valid) in enumerate(zip(tree, masks)):
            frontier = frontier * valid.to(frontier.dtype)
            frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
            local = (frontier[:, :, None] * nodes).sum(1)
            if depth != ablate_depth and mode != "leaf_only":
                depth_state = self.depth_embedding.weight[depth][None].expand_as(local)
                query = query + gain * self.read_kernel(query, local, depth_state)
            entropies.append(
                -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1).mean()
            )
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            # Retain the C10 branch carrier and let accumulated multi-level
            # evidence perturb it gradually.
            branch_query = (self.branch(hidden) + gain * (query - base_query))[:, None, None]
            scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
            scores = scores.masked_fill(~child_valid, -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * child_valid.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
        # Preserve the inherited leaf-resolution carrier and add the complete
        # root-to-leaf convolution as a trainable residual protocol.
        return local + (query - base_query), torch.stack(entropies)

    def teacher(self, levels, masks, target, bos: int, mode: str = "native", ablate_depth: int = -1):
        hidden = levels[0].new_zeros((levels[0].shape[0], self.hidden))
        previous = torch.full((levels[0].shape[0],), bos, dtype=torch.long, device=levels[0].device)
        logits, diagnostics = [], []
        tree = self.convolve(levels, masks, bypass_up=mode == "bypass_up")
        for step in range(target.shape[1]):
            context, entropy = self.read(hidden, tree, masks, mode, ablate_depth)
            hidden = self.cell(torch.cat((self.embedding(previous), context), dim=-1), hidden)
            logits.append(self.output(torch.cat((hidden, context), dim=-1)))
            diagnostics.append(entropy)
            previous = target[:, step]
        return torch.stack(logits, dim=1), torch.stack(diagnostics).mean(0)

    def greedy(self, levels, masks, bos: int, eos: int, max_len: int):
        hidden = levels[0].new_zeros((levels[0].shape[0], self.hidden))
        previous = torch.full((levels[0].shape[0],), bos, dtype=torch.long, device=levels[0].device)
        done = torch.zeros(levels[0].shape[0], dtype=torch.bool, device=levels[0].device)
        output, diagnostics = [], []
        tree = self.convolve(levels, masks)
        for _ in range(max_len):
            context, entropy = self.read(hidden, tree, masks)
            hidden = self.cell(torch.cat((self.embedding(previous), context), dim=-1), hidden)
            previous = self.output(torch.cat((hidden, context), dim=-1)).argmax(-1)
            output.append(previous)
            diagnostics.append(entropy)
            done |= previous.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(output, dim=1), torch.stack(diagnostics).mean(0)


class HStateConvolutionModel(nn.Module):
    def __init__(self, base_model, dim: int, hidden: int):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = MultiLevelConvolutionDecoder(
            base_model.decoder, dim, hidden, self.encoder.depths,
        )

    def states(self, source, length, intervention: str = "native", pair_break_depth: int = -1):
        return self.encoder.states(source, length, intervention, pair_break_depth)

    def teacher(
        self, source, length, target, bos, mode: str = "native", ablate_depth: int = -1,
        intervention: str = "native", pair_break_depth: int = -1,
    ):
        _, _, _, levels, masks = self.states(source, length, intervention, pair_break_depth)
        return self.decoder.teacher(levels, masks, target, bos, mode, ablate_depth)

    def greedy(self, source, length, bos, eos, max_len):
        _, _, _, levels, masks = self.states(source, length)
        return self.decoder.greedy(levels, masks, bos, eos, max_len)


@torch.no_grad()
def evaluate(
    model, rows, pad, bos, device, batch_size, mode="native", ablate_depth=-1,
    intervention="native", pair_break_depth=-1, runtime_mode=None,
):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    previous_mode = model.encoder.runtime_mode
    model.encoder.runtime_mode = runtime_mode
    try:
        for start in range(0, len(rows), batch_size):
            source, length, target = c10.collate_rows(rows[start:start + batch_size], pad, device)
            logits, _ = model.teacher(
                source, length, target, bos, mode, ablate_depth,
                intervention, pair_break_depth,
            )
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                ignore_index=pad, reduction="sum",
            ))
            tokens += int(target.ne(pad).sum())
    finally:
        model.encoder.runtime_mode = previous_mode
    nll = loss_sum / max(1, tokens)
    return {"nll": nll, "ppl": math.exp(min(20.0, nll)), "tokens": tokens}


def gradient_probe(model, batch, pad, bos, device):
    model.train()
    source, length, target = c10.collate_rows(batch, pad, device)
    _, _, _, levels, masks = model.states(source, length)
    for level in levels:
        level.retain_grad()
    logits, entropy = model.decoder.teacher(levels, masks, target, bos)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    level_grad_norms = [float(level.grad.norm()) if level.grad is not None else 0.0 for level in levels]
    parameter_groups = {
        "encoder_embedding": [model.encoder.embedding.weight],
        "fold_predictor": list(model.encoder.predictor.parameters()),
        "fold_update": list(model.encoder.update_kernel.parameters()),
        "butterfly_forward": list(model.encoder.communication.forward_kernel.parameters()),
        "butterfly_backward": list(model.encoder.communication.backward_kernel.parameters()),
        "up_kernel": list(model.decoder.up_kernel.parameters()),
        "read_kernel": list(model.decoder.read_kernel.parameters()),
        "branch_kernel": list(model.decoder.branch.parameters()),
    }
    group_norms = {}
    for name, parameters in parameter_groups.items():
        squares = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
        group_norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return {
        "loss": float(loss.detach()),
        "level_widths": [level.shape[1] for level in levels],
        "level_grad_norms": level_grad_norms,
        "parameter_grad_norms": group_norms,
        "frontier_entropy": [float(value) for value in entropy.detach().cpu()],
        "finite": bool(torch.isfinite(loss)) and all(math.isfinite(value) for value in level_grad_norms),
    }


def clean_rows(rows, max_source: int, max_target: int):
    return [row for row in rows if len(row[0]) <= max_source and len(row[1]) <= max_target]


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--max-source", type=int, default=0)
    parser.add_argument("--max-target", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        args.steps = args.steps or 40
        args.batch_size = args.batch_size or 8
        args.train_rows = args.train_rows or 512
        args.eval_rows = args.eval_rows or 64
        args.max_source = args.max_source or 64
        args.max_target = args.max_target or 64
        args.log_every = args.log_every or 10
    else:
        args.steps = args.steps or 25000
        args.batch_size = args.batch_size or 16
        args.train_rows = args.train_rows or 200000
        args.eval_rows = args.eval_rows or 1000
        args.max_source = args.max_source or 253
        args.max_target = args.max_target or 253
        args.log_every = args.log_every or 1000

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"
    if trace_path.exists() and not args.resume:
        trace_path.unlink()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.wmt_data = args.wmt_data
    config.task_train_rows = max(args.train_rows * 2, args.train_rows + 1024)
    config.task_eval_rows = max(args.eval_rows * 2, args.eval_rows + 128)
    config.max_wmt_scan_lines = 3_000_000
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    base_model = c10.build_model(config, vocab, pad)
    base_model.load_state_dict(payload["state_dict"], strict=True)
    model = HStateConvolutionModel(base_model, config.dim, config.hidden).to(args.device)

    collected = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    train_rows = clean_rows(collected[0], args.max_source, args.max_target)[:args.train_rows]
    valid_rows = clean_rows(collected[1], args.max_source, args.max_target)[:args.eval_rows]
    if len(train_rows) < args.batch_size or not valid_rows:
        raise RuntimeError(f"insufficient rows train={len(train_rows)} valid={len(valid_rows)}")
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 21)
    stream_hash = c10.stream_sha256(schedule)

    probe_before = gradient_probe(model, schedule[0], pad, bos, args.device)
    initial = evaluate(model, valid_rows, pad, bos, args.device, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_nll = initial["nll"]
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    progress_path = output / "checkpoint_progress.pt"
    start_step = 0
    train_loss_sum = 0.0
    train_tokens = 0
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("stream_sha256") != stream_hash:
            raise RuntimeError("resume stream hash mismatch")
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
        print(json.dumps({"event": "resume", "step": start_step}), flush=True)
    started = time.time()
    for step, batch in enumerate(schedule, 1):
        if step <= start_step:
            continue
        model.train()
        source, length, target = c10.collate_rows(batch, pad, args.device)
        logits, entropy = model.teacher(source, length, target, bos)
        count = int(target.ne(pad).sum())
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        )
        loss = loss_sum / max(1, count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        train_loss_sum += float(loss_sum.detach())
        train_tokens += count
        if step == 1 or step == args.steps or step % args.log_every == 0:
            valid = evaluate(model, valid_rows, pad, bos, args.device, args.batch_size)
            row = {
                "step": step,
                "train_nll": train_loss_sum / max(1, train_tokens),
                "valid_nll": valid["nll"],
                "grad_norm": grad_norm,
                "frontier_entropy": [float(value) for value in entropy.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
            print(json.dumps(row), flush=True)
            if valid["nll"] < best_nll:
                best_nll = valid["nll"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if args.mode == "formal":
                temporary = progress_path.with_suffix(".pt.tmp")
                torch.save({
                    "claim": CLAIM,
                    "step": step,
                    "stream_sha256": stream_hash,
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_nll": best_nll,
                    "best_state": best_state,
                    "train_loss_sum": train_loss_sum,
                    "train_tokens": train_tokens,
                }, temporary)
                os.replace(temporary, progress_path)

    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_rows, pad, bos, args.device, args.batch_size)
    probe_after = gradient_probe(model, schedule[-1], pad, bos, args.device)
    interventions = {
        "native": final,
        "bypass_up": evaluate(model, valid_rows, pad, bos, args.device, args.batch_size, "bypass_up"),
        "leaf_only": evaluate(model, valid_rows, pad, bos, args.device, args.batch_size, "leaf_only"),
        "source_shuffle": evaluate(
            model, valid_rows, pad, bos, args.device, args.batch_size,
            intervention="source_shuffle",
        ),
        "runtime_identity": evaluate(
            model, valid_rows, pad, bos, args.device, args.batch_size,
            runtime_mode="identity",
        ),
        "pair_break_depth_0": evaluate(
            model, valid_rows, pad, bos, args.device, args.batch_size,
            pair_break_depth=0,
        ),
        "ablate_depth": {},
    }
    for depth in range(len(probe_after["level_widths"])):
        interventions["ablate_depth"][str(depth)] = evaluate(
            model, valid_rows, pad, bos, args.device, args.batch_size, "native", depth,
        )
    native_nll = final["nll"]
    nonleaf_deltas = [
        interventions["ablate_depth"][str(depth)]["nll"] - native_nll
        for depth in range(len(probe_after["level_widths"]) - 1)
    ]
    p0 = {
        "tree_widths_double": probe_after["level_widths"] == [1 << depth for depth in range(len(probe_after["level_widths"]))],
        "no_stop_parameter": not any("stop" in name for name, _ in model.decoder.named_parameters()),
        "all_level_gradients_nonzero": all(value > 0.0 for value in probe_after["level_grad_norms"]),
        "all_required_parameter_gradients_nonzero": all(value > 0.0 for value in probe_after["parameter_grad_norms"].values()),
        "finite": probe_after["finite"] and math.isfinite(final["nll"]),
    }
    p1 = {
        "valid_nll_improved": final["nll"] < initial["nll"],
        "bottom_up_causal": abs(interventions["bypass_up"]["nll"] - native_nll) > 1e-4,
        "two_nonleaf_depths_helpful": sum(value > 1e-4 for value in nonleaf_deltas) >= 2,
    }
    generation = c10.task_generation_metrics(
        model, valid_rows, SimpleNamespace(
            device=args.device, max_generation=min(96, args.max_target + 16),
        ),
        sp, pad, bos, eos, pieces, limit=min(32, len(valid_rows)),
    )
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "host": socket.gethostname(),
        "source_checkpoint": args.checkpoint,
        "source_state_sha256": payload.get("state_sha256"),
        "trained_state_sha256": state_sha256(best_state),
        "config": vars(args),
        "rows": {"train": len(train_rows), "valid": len(valid_rows)},
        "stream_sha256": stream_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial": initial,
        "final": final,
        "train_tokens": train_tokens,
        "probe_before": probe_before,
        "probe_after": probe_after,
        "interventions": interventions,
        "intervention_deltas": {
            name: result["nll"] - native_nll
            for name, result in interventions.items()
            if name not in ("native", "ablate_depth")
        },
        "nonleaf_ablation_deltas": nonleaf_deltas,
        "generation": generation,
        "p0": p0,
        "p0_pass": all(p0.values()),
        "p1": p1,
        "p1_pass": all(p1.values()),
        "seconds": time.time() - started,
        "not_proved": [
            "human-readable hierarchy", "private-protocol uniqueness",
            "formal WMT improvement when mode=smoke", "product readiness",
        ],
    }
    json_write(output / "summary.json", summary)
    if args.mode == "formal":
        checkpoint_path = output / "checkpoint_best.pt"
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({
            "claim": CLAIM, "state_dict": best_state,
            "state_sha256": summary["trained_state_sha256"], "config": vars(args),
        }, temporary)
        os.replace(temporary, checkpoint_path)
    print(json.dumps({
        "event": "complete", "p0_pass": summary["p0_pass"], "p1_pass": summary["p1_pass"],
        "initial_nll": initial["nll"], "final_nll": final["nll"],
        "level_grad_norms": probe_after["level_grad_norms"],
        "nonleaf_ablation_deltas": nonleaf_deltas,
        "evidence": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
