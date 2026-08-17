#!/usr/bin/env python3
"""Matched training for native and reference-normalized FOLD arms."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


ARMS = ("learned", "ref_zero", "ref_embedding_mean")
EPSILON = 1e-8


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ReferenceEncoder(nn.Module):
    def __init__(self, inner: nn.Module, origin: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("origin", origin.detach().clone())
        self.depths = inner.depths
        self.heap_width = inner.heap_width
        self.runtime_mode = None

    def fold(self, source, length, pair_break_depth=-1):
        leaf, leaf_mask = self.inner.raw_leaf(source, length)
        mode = self.runtime_mode or self.inner.communication_mode
        leaf = self.inner.communication(leaf, leaf_mask, mode)
        node, node_mask = leaf, leaf_mask
        details, scales, masks = [], [], [leaf_mask]
        active_depths = int(math.log2(leaf.shape[1]))
        for depth in range(active_depths):
            left, right = node[:, 0::2], node[:, 1::2]
            lm, rm = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == pair_break_depth:
                right, rm = right.roll(1, dims=0), rm.roll(1, dims=0)
            paired = lm & rm
            only_left, only_right = lm & ~rm, rm & ~lm
            a, b = left - self.origin, right - self.origin
            scale = torch.sqrt(
                a.square().sum(-1, keepdim=True)
                + b.square().sum(-1, keepdim=True)
                + EPSILON
            )
            direction = (a + b) / (math.sqrt(2.0) * scale)
            detail = (b - a) / (math.sqrt(2.0) * scale)
            parent = self.origin + direction
            parent = torch.where(only_left[:, :, None], left, parent)
            parent = torch.where(only_right[:, :, None], right, parent)
            detail = torch.where(paired[:, :, None], detail, torch.zeros_like(detail))
            scale = torch.where(paired[:, :, None], scale, torch.ones_like(scale))
            node_mask = lm | rm
            node = parent * node_mask[:, :, None]
            details.append(detail * node_mask[:, :, None])
            scales.append(scale * node_mask[:, :, None])
            masks.append(node_mask)
        return leaf, node[:, 0], details, scales, masks

    def unfold(self, root, details, scales, masks):
        level = root[:, None]
        levels, level_masks = [level], [masks[-1]]
        for depth in range(len(details) - 1, -1, -1):
            detail, scale = details[depth], scales[depth]
            lm, rm = masks[depth][:, 0::2], masks[depth][:, 1::2]
            paired = lm & rm
            only_left, only_right = lm & ~rm, rm & ~lm
            direction = level - self.origin
            a = scale * (direction - detail) / math.sqrt(2.0)
            b = scale * (direction + detail) / math.sqrt(2.0)
            left, right = self.origin + a, self.origin + b
            left = torch.where(only_left[:, :, None], level, left)
            right = torch.where(only_right[:, :, None], level, right)
            left = torch.where((paired | only_left)[:, :, None], left, torch.zeros_like(left))
            right = torch.where((paired | only_right)[:, :, None], right, torch.zeros_like(right))
            expanded = torch.empty(
                left.shape[0], left.shape[1] * 2, left.shape[2],
                dtype=left.dtype, device=left.device,
            )
            expanded[:, 0::2], expanded[:, 1::2] = left, right
            level = expanded
            levels.append(level)
            level_masks.append(masks[depth])
        return levels, level_masks

    def states(self, source, length, intervention="native", pair_break_depth=-1):
        leaf, root, details, scales, masks = self.fold(source, length, pair_break_depth)
        if intervention == "source_shuffle":
            root = root.roll(1, dims=0)
            details = [value.roll(1, dims=0) for value in details]
            scales = [value.roll(1, dims=0) for value in scales]
            masks = [value.roll(1, dims=0) for value in masks]
        elif intervention != "native":
            raise ValueError(intervention)
        levels, level_masks = self.unfold(root, details, scales, masks)
        return leaf, root, details, levels, level_masks


class ReferenceModel(nn.Module):
    def __init__(self, base, origin):
        super().__init__()
        self.encoder = ReferenceEncoder(base.encoder, origin)
        self.decoder = base.decoder

    def states(self, source, length, intervention="native", pair_break_depth=-1):
        return self.encoder.states(source, length, intervention, pair_break_depth)

    def teacher(self, source, length, target, bos, intervention="native", pair_break_depth=-1):
        _, _, _, levels, masks = self.states(source, length, intervention, pair_break_depth)
        return self.decoder.teacher(levels, masks, target, bos, "native")

    def greedy(self, source, length, bos, eos, max_len):
        _, _, _, levels, masks = self.states(source, length)
        return self.decoder.greedy(levels, masks, bos, eos, max_len, "native")


def build_model(config, vocab, pad, arm, seed, pieces):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = c10.build_model(config, vocab, pad)
    common_hash = c10.state_sha256(c10.cpu_state(base))
    if arm == "learned":
        return base, common_hash
    if arm == "ref_zero":
        origin = torch.zeros(config.dim, device=config.device)
    elif arm == "ref_embedding_mean":
        origin = base.encoder.embedding.weight[:pieces].mean(0).detach()
    else:
        raise ValueError(arm)
    return ReferenceModel(base, origin).to(config.device), common_hash


@torch.no_grad()
def evaluate(model, rows, pad, bos, device, batch_size, **kwargs):
    model.eval()
    loss_sum, tokens = 0.0, 0
    for start in range(0, len(rows), batch_size):
        source, length, target = c10.collate_rows(rows[start:start + batch_size], pad, device)
        logits, _ = model.teacher(source, length, target, bos, **kwargs)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
    return loss_sum / max(1, tokens)


@torch.no_grad()
def generation(model, rows, pad, bos, eos, pieces, device, sp, limit=256):
    by_direction = {
        "en2zh": {"hypotheses": [], "references": []},
        "zh2en": {"hypotheses": [], "references": []},
    }
    examples = []
    adjacent_equal, adjacent_total = 0, 0
    for row in rows[:limit]:
        source, length, target = c10.collate_rows([row], pad, device)
        generated, _ = model.greedy(source, length, bos, eos, min(96, target.shape[1] + 16))
        hypothesis = c10.wmt.clean(generated[0].tolist(), eos, pieces)
        reference = c10.wmt.clean(target[0].tolist(), eos, pieces)
        direction = row[2]
        hypothesis_text, reference_text = sp.decode(hypothesis), sp.decode(reference)
        by_direction[direction]["hypotheses"].append(hypothesis_text)
        by_direction[direction]["references"].append(reference_text)
        adjacent_equal += sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
        adjacent_total += max(0, len(hypothesis) - 1)
        if len(examples) < 24:
            examples.append({
                "direction": direction,
                "source": row[3][1] if direction == "en2zh" else row[3][0],
                "reference": reference_text,
                "generation": hypothesis_text,
            })

    standard = {}
    try:
        import sacrebleu
        for direction, values in by_direction.items():
            hypotheses, references = values["hypotheses"], values["references"]
            tokenizer = "zh" if direction == "en2zh" else "13a"
            standard[direction] = {
                "sentences": len(hypotheses),
                "sacrebleu": float(sacrebleu.corpus_bleu(
                    hypotheses, [references], tokenize=tokenizer,
                ).score),
                "chrf2": float(sacrebleu.corpus_chrf(
                    hypotheses, [references], word_order=2,
                ).score),
            }
    except (ImportError, RuntimeError, ValueError) as error:
        standard["error"] = f"{type(error).__name__}: {error}"
    return {
        "evaluated_sentences": min(limit, len(rows)),
        "standard": standard,
        "adjacent_repetition_rate": adjacent_equal / max(1, adjacent_total),
        "nonempty_rate": sum(
            bool(text) for values in by_direction.values() for text in values["hypotheses"]
        ) / max(1, min(limit, len(rows))),
        "examples": examples,
    }


def group_grad_norm(model):
    groups = {"encoder_embedding": 0.0, "communication": 0.0, "decoder": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().square().sum())
        if "embedding" in name and "decoder" not in name:
            groups["encoder_embedding"] += value
        elif "communication" in name:
            groups["communication"] += value
        elif "decoder" in name:
            groups["decoder"] += value
    return {name: math.sqrt(value) for name, value in groups.items()}


def closure(model, rows, pad, device):
    if not isinstance(model, ReferenceModel):
        return None
    source, length, _ = c10.collate_rows(rows[:16], pad, device)
    leaf, root, details, scales, masks = model.encoder.fold(source, length)
    levels, _ = model.encoder.unfold(root, details, scales, masks)
    difference = levels[-1] - leaf
    return {
        "mse": float(difference.square().mean().detach()),
        "max_abs": float(difference.abs().max().detach()),
        "root_rms": float(root.square().mean().sqrt().detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--config-checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--eval-wmt-data", default=None)
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13201)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--generation-rows", type=int, default=256)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.wmt_data = args.wmt_data
    config.task_train_rows = args.train_rows
    config.task_eval_rows = args.eval_rows
    config.max_wmt_scan_lines = 1_000_000
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, pad, bos, eos = sp.get_piece_size(), sp.get_piece_size(), sp.bos_id(), sp.eos_id()
    vocab = pieces + 3
    model, common_initial_hash = build_model(config, vocab, pad, args.arm, args.seed, pieces)
    directions = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    train_rows, valid_rows, test_rows = c10.collect_wmt_rows(config, sp, directions, eos)
    if args.eval_wmt_data:
        config.wmt_data = args.eval_wmt_data
        _, valid_rows, test_rows = c10.collect_wmt_rows(config, sp, directions, eos)
        config.wmt_data = args.wmt_data
    train_rows, valid_rows, test_rows = (
        train_rows[:args.train_rows], valid_rows[:args.eval_rows], test_rows[:args.eval_rows]
    )
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 21)
    stream_hash = c10.stream_sha256(schedule)
    initial_valid = evaluate(model, valid_rows, pad, bos, args.device, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_nll, best_step, best_state = initial_valid, 0, c10.cpu_state(model)
    trace, started, final_grad_groups = [], time.time(), None
    for step, batch in enumerate(schedule, 1):
        model.train()
        source, length, target = c10.collate_rows(batch, pad, args.device)
        logits, _ = model.teacher(source, length, target, bos)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        final_grad_groups = group_grad_norm(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            valid_nll = evaluate(model, valid_rows, pad, bos, args.device, args.batch_size)
            row = {
                "step": step, "train_nll": float(loss.detach()),
                "valid_nll": valid_nll, "grad_norm": grad_norm,
                "gradient_groups": final_grad_groups,
                "seconds": time.time() - started,
            }
            trace.append(row)
            print(json.dumps({"arm": args.arm, **row}), flush=True)
            if valid_nll < best_nll:
                best_nll, best_step, best_state = valid_nll, step, c10.cpu_state(model)

    model.load_state_dict(best_state, strict=True)
    test_nll = evaluate(model, test_rows, pad, bos, args.device, args.batch_size)
    source_shuffle_nll = evaluate(
        model, test_rows, pad, bos, args.device, args.batch_size,
        intervention="source_shuffle",
    )
    pair_break_nll = evaluate(
        model, test_rows, pad, bos, args.device, args.batch_size,
        pair_break_depth=0,
    )
    runtime_identity_nll = None
    encoder = model.encoder
    previous = encoder.runtime_mode
    encoder.runtime_mode = "identity"
    try:
        runtime_identity_nll = evaluate(model, test_rows, pad, bos, args.device, args.batch_size)
    finally:
        encoder.runtime_mode = previous

    generation_result = generation(
        model, test_rows, pad, bos, eos, pieces, args.device, sp, args.generation_rows,
    )
    evidence_dir = Path(args.evidence_dir)
    write_json(evidence_dir / "generation.json", generation_result)
    if args.save_checkpoint:
        torch.save({
            "claim": "S3-BOUNDED-ANNEALING-FOLD-C13",
            "arm": args.arm,
            "seed": args.seed,
            "best_step": best_step,
            "best_valid_nll": best_nll,
            "model_state": best_state,
            "model_config": vars(config),
            "spm_model": args.spm_model,
        }, evidence_dir / "checkpoint_best.pt")

    summary = {
        "claim": "S3-BOUNDED-ANNEALING-FOLD-C13",
        "scope": "matched formal route probe" if args.steps >= 1000 else "matched smoke training only",
        "arm": args.arm,
        "config": vars(args),
        "common_initial_base_hash": common_initial_hash,
        "stream_sha256": stream_hash,
        "rows": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_valid_nll": initial_valid,
        "best_valid_nll": best_nll,
        "best_step": best_step,
        "test_nll": test_nll,
        "source_shuffle_delta": source_shuffle_nll - test_nll,
        "pair_break_depth_0_delta": pair_break_nll - test_nll,
        "runtime_identity_delta": runtime_identity_nll - test_nll,
        "generation": generation_result,
        "closure": closure(model, test_rows, pad, args.device),
        "final_gradient_groups": final_grad_groups,
        "trace": trace,
        "seconds": time.time() - started,
        "finite": math.isfinite(test_nll),
    }
    write_json(evidence_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
