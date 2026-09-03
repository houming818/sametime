#!/usr/bin/env python3
"""D12: train TreeHeap level states against fixed order-sensitive outcome sketches."""
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

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402
import s3_structural_slot_ownership_d08 as d08  # noqa: E402
import s3_structural_slot_ownership_d09_scale as d09  # noqa: E402


CLAIM = "S3-PREDICTIVE-STATE-SKETCH-D12"
ARMS = ("probe-only", "predictive-gradient")
DEPTHS = d07.DEPTHS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    return c10.state_sha256({key: value.detach().cpu() for key, value in state.items()})


class FixedOutcomeSketch(nn.Module):
    """A fixed random feature map for an ordered token sequence.

    Token and position codes are buffers, never learned. Their normalized sum is
    one sample of a finite-dimensional conditional outcome representation.
    """

    def __init__(self, vocab: int, width: int, max_positions: int, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        token = torch.randint(0, 2, (vocab, width), generator=generator, dtype=torch.int8)
        position = torch.randint(
            0, 2, (max_positions, width), generator=generator, dtype=torch.int8,
        )
        scale = 1.0 / math.sqrt(width)
        self.register_buffer("token_code", (token.float() * 2.0 - 1.0) * scale)
        self.register_buffer("position_sign", position.float() * 2.0 - 1.0)
        self.width = width
        self.max_positions = max_positions

    def forward(self, tokens: torch.Tensor, pad: int) -> torch.Tensor:
        if tokens.shape[1] > self.max_positions:
            raise ValueError(
                f"target length {tokens.shape[1]} exceeds sketch limit {self.max_positions}"
            )
        valid = tokens.ne(pad)
        safe = tokens.masked_fill(~valid, 0)
        features = self.token_code[safe]
        features = features * self.position_sign[:tokens.shape[1]][None]
        features = features * valid[:, :, None]
        count = valid.sum(1).clamp_min(1).to(features.dtype).sqrt()[:, None]
        return features.sum(1) / count

    def digest(self) -> str:
        return tensor_state_sha256({
            "token_code": self.token_code,
            "position_sign": self.position_sign,
        })


class LevelPredictor(nn.Module):
    """Read each convolved TreeHeap level into the same fixed outcome space."""

    def __init__(self, dim: int, levels: int, sketch_width: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(dim, sketch_width) for _ in range(levels)
        ])
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

    @staticmethod
    def summarize(level: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask[:, :, None].to(level.dtype)
        count = mask.sum(1).clamp_min(1).to(level.dtype).sqrt()[:, None]
        state = (level * weight).sum(1) / count
        return F.layer_norm(state, (state.shape[-1],))

    def forward(self, levels, masks, detach_states: bool = False):
        if len(levels) != len(self.heads):
            raise ValueError(f"expected {len(self.heads)} levels, got {len(levels)}")
        states = [self.summarize(level, mask) for level, mask in zip(levels, masks)]
        inputs = [state.detach() if detach_states else state for state in states]
        return [head(state) for head, state in zip(self.heads, inputs)], states


def reverse_valid_tokens(target: torch.Tensor, pad: int) -> torch.Tensor:
    result = torch.full_like(target, pad)
    for row_index in range(target.shape[0]):
        values = target[row_index, target[row_index].ne(pad)].flip(0)
        result[row_index, :values.numel()] = values
    return result


def sketch_loss(predictions, target_sketch: torch.Tensor) -> torch.Tensor:
    # Sum over sketch coordinates keeps a random unit target near loss 1.
    return torch.stack([
        (prediction - target_sketch).square().sum(-1).mean()
        for prediction in predictions
    ]).mean()


def level_geometry(states: list[torch.Tensor]) -> dict:
    result = {}
    for index, state in enumerate(states):
        centered = state - state.mean(0, keepdim=True)
        singular = torch.linalg.svdvals(centered.float())
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-12)
        effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
        normalized = F.normalize(state.float(), dim=-1)
        gram = normalized @ normalized.T
        if state.shape[0] > 1:
            off_diagonal = (gram.sum() - gram.diag().sum()) / (state.shape[0] * (state.shape[0] - 1))
        else:
            off_diagonal = gram.new_tensor(0.0)
        result[str(index)] = {
            "effective_rank": effective_rank,
            "dimension_std_mean": float(centered.std(0, unbiased=False).mean()),
            "mean_pair_cosine": float(off_diagonal),
        }
    return result


def forward_batch(model, predictor, sketcher, source, lengths, target, bos, pad, depth, detach):
    tree, masks, budgets, slots, entropy = model.protocol(source, lengths, depth)
    logits, route = model.reconstructor.teacher(tree, masks, target, bos)
    convolved = model.reconstructor.convolve(tree, masks)
    predictions, states = predictor(convolved, masks, detach_states=detach)
    target_state = sketcher(target, pad)
    return {
        "logits": logits, "route": route, "budgets": budgets, "slots": slots,
        "entropy": entropy, "tree": tree, "convolved": convolved, "predictions": predictions,
        "states": states, "target_state": target_state, "masks": masks,
    }


@torch.no_grad()
def evaluate_predictive(model, predictor, sketcher, rows, args, pad, bos):
    model.eval()
    predictor.eval()
    loss_sum = token_count = retrieval_hits = retrieval_total = 0
    level_squared = level_shuffled = level_reversed = None
    sample_count = 0
    states_by_level = None
    for start in range(0, len(rows), args.eval_batch):
        batch = rows[start:start + args.eval_batch]
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        depth = DEPTHS[(start // args.eval_batch) % len(DEPTHS)]
        payload = forward_batch(
            model, predictor, sketcher, source, lengths, target, bos, pad, depth, False,
        )
        local_tokens = int(target.ne(pad).sum())
        loss_sum += float(F.cross_entropy(
            payload["logits"].reshape(-1, payload["logits"].shape[-1]),
            target.reshape(-1), ignore_index=pad, reduction="sum",
        ))
        token_count += local_tokens
        expected = payload["target_state"]
        shuffled = expected.roll(1, dims=0) if expected.shape[0] > 1 else expected
        reversed_state = sketcher(reverse_valid_tokens(target, pad), pad)
        local_squared = torch.stack([
            (prediction - expected).square().sum(-1).sum()
            for prediction in payload["predictions"]
        ])
        local_shuffled = torch.stack([
            (prediction - shuffled).square().sum(-1).sum()
            for prediction in payload["predictions"]
        ])
        local_reversed = torch.stack([
            (prediction - reversed_state).square().sum(-1).sum()
            for prediction in payload["predictions"]
        ])
        level_squared = local_squared if level_squared is None else level_squared + local_squared
        level_shuffled = local_shuffled if level_shuffled is None else level_shuffled + local_shuffled
        level_reversed = local_reversed if level_reversed is None else level_reversed + local_reversed
        sample_count += len(batch)
        ensemble = torch.stack(payload["predictions"]).mean(0)
        scores = F.normalize(ensemble, dim=-1) @ F.normalize(expected, dim=-1).T
        retrieval_hits += int(scores.argmax(-1).eq(torch.arange(len(batch), device=scores.device)).sum())
        retrieval_total += len(batch)
        if states_by_level is None:
            states_by_level = [[] for _ in payload["states"]]
        for bucket, state in zip(states_by_level, payload["states"]):
            bucket.append(state.detach().cpu())
    mse = level_squared / max(1, sample_count)
    shuffled_mse = level_shuffled / max(1, sample_count)
    reversed_mse = level_reversed / max(1, sample_count)
    geometry = level_geometry([torch.cat(bucket) for bucket in states_by_level])
    nll = loss_sum / max(1, token_count)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": token_count,
        "predictive_mse_by_level": [float(value) for value in mse.cpu()],
        "predictive_mse_mean": float(mse.mean()),
        "shuffle_gap_by_level": [float(value) for value in (shuffled_mse - mse).cpu()],
        "shuffle_gap_mean": float((shuffled_mse - mse).mean()),
        "reverse_order_gap_by_level": [float(value) for value in (reversed_mse - mse).cpu()],
        "reverse_order_gap_mean": float((reversed_mse - mse).mean()),
        "retrieval_at_1": retrieval_hits / max(1, retrieval_total),
        "level_geometry": geometry,
    }


def parameter_summary(model, predictor) -> dict:
    return {
        "model_total": sum(parameter.numel() for parameter in model.parameters()),
        "model_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "predictor_total": sum(parameter.numel() for parameter in predictor.parameters()),
        "combined_trainable": (
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            + sum(parameter.numel() for parameter in predictor.parameters())
        ),
    }


def make_model(source_cpu, config, args, warm_path: Path):
    return d10.make_model(source_cpu, config, args, warm_path)


def save_checkpoint(path: Path, model, predictor, optimizer, arm: str, step: int, cursor: int, run: dict):
    model_state = d08.trainable_state(model)
    predictor_state = {key: value.detach().cpu() for key, value in predictor.state_dict().items()}
    payload = {
        "claim": CLAIM, "arm": arm, "step": step, "cursor": cursor,
        "trainable_state_dict": model_state,
        "trainable_state_sha256": tensor_state_sha256(model_state),
        "predictor_state_dict": predictor_state,
        "predictor_state_sha256": tensor_state_sha256(predictor_state),
        "optimizer_state_dict": optimizer.state_dict(), "run": run,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


def train_arm(arm, source_cpu, config, warm_path, warm, valid_rows, test_rows, excluded, args, sp, pieces, eos, bos, direction_ids):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, _ = make_model(source_cpu, config, args, warm_path)
    predictor = LevelPredictor(config.dim, int(math.log2(args.max_slots)) + 1, args.sketch_width).to(args.device)
    sketcher = FixedOutcomeSketch(args.vocab, args.sketch_width, args.max_target, args.sketch_seed).to(args.device)
    output = Path(args.evidence_dir) / arm
    output.mkdir(parents=True, exist_ok=True)
    initial = evaluate_predictive(model, predictor, sketcher, valid_rows, args, args.pad, bos)
    source_before = tensor_state_sha256(model.frozen_source.state_dict())
    initial_model_hash = tensor_state_sha256(d08.trainable_state(model))
    initial_predictor_hash = tensor_state_sha256(predictor.state_dict())
    run = {
        "claim": CLAIM, "mode": args.mode, "arm": arm, "seed": args.seed,
        "sketch_seed": args.sketch_seed, "sketch_sha256": sketcher.digest(),
        "source_sha256": source_before,
        "warm_start_sha256": warm["trainable_state_sha256"],
        "parallel_sha256": d10.PARALLEL_SHA256,
        "parameters": parameter_summary(model, predictor),
        "config": vars(args), "initial": initial,
        "initial_model_sha256": initial_model_hash,
        "initial_predictor_sha256": initial_predictor_hash,
    }
    write_json(output / "contract.json", run)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    iterator = d10.iter_parallel_batches(
        Path(args.parallel_data), sp, direction_ids, eos, args.batch_size,
        0, args.max_lines, excluded,
    )
    trace = output / "trace.jsonl"
    started = time.time()
    first_aux_level_grad = None
    step = cursor = processed_tokens = 0
    finite = True
    for cursor, batch, _ in iterator:
        step += 1
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        model.train()
        predictor.train()
        source, lengths, target = c10.collate_rows(batch, args.pad, args.device)
        payload = forward_batch(
            model, predictor, sketcher, source, lengths, target, bos, args.pad, depth,
            detach=arm == "probe-only",
        )
        tokens = int(target.ne(args.pad).sum())
        ce = F.cross_entropy(
            payload["logits"].reshape(-1, payload["logits"].shape[-1]),
            target.reshape(-1), ignore_index=args.pad, reduction="sum",
        ) / max(1, tokens)
        predictive = sketch_loss(payload["predictions"], payload["target_state"])
        total = ce + args.predictive_weight * predictive
        optimizer.zero_grad(set_to_none=True)
        if first_aux_level_grad is None:
            if arm == "predictive-gradient":
                gradients = torch.autograd.grad(
                    predictive, payload["convolved"], retain_graph=True, allow_unused=True,
                )
                first_aux_level_grad = [
                    float(gradient.detach().norm()) if gradient is not None else 0.0
                    for gradient in gradients
                ]
            else:
                first_aux_level_grad = [0.0 for _ in payload["convolved"]]
        total.backward()
        finite = finite and d07.finite_trainable_gradients(model)
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in predictor.parameters()
        )
        if not finite:
            raise RuntimeError(f"non-finite gradient in {arm} at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        processed_tokens += tokens
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            event = {
                "event": "train", "arm": arm, "step": step, "cursor": cursor,
                "depth": depth, "ce": float(ce.detach()),
                "predictive_loss": float(predictive.detach()),
                "total_loss": float(total.detach()), "grad_norm": grad_norm,
                "aux_level_grad_norms_step1": first_aux_level_grad,
                "tokens": processed_tokens,
                "route": [float(value) for value in payload["route"].detach().cpu()],
                "slot_route": model.compressor.last_route_statistics,
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if step >= args.steps:
            break
    final = evaluate_predictive(model, predictor, sketcher, valid_rows, args, args.pad, bos)
    generation = d10.generation_summary(
        model, test_rows, args, sp, args.pad, bos, eos, pieces,
    )
    checkpoint = save_checkpoint(
        output / "checkpoint_latest.pt", model, predictor, optimizer, arm, step, cursor, run,
    )
    source_after = tensor_state_sha256(model.frozen_source.state_dict())

    reload_model, _ = make_model(source_cpu, config, args, warm_path)
    d09.load_trainable(reload_model, checkpoint)
    reload_predictor = LevelPredictor(
        config.dim, int(math.log2(args.max_slots)) + 1, args.sketch_width,
    ).to(args.device)
    reload_predictor.load_state_dict(checkpoint["predictor_state_dict"])
    reload_sketcher = FixedOutcomeSketch(
        args.vocab, args.sketch_width, args.max_target, args.sketch_seed,
    ).to(args.device)
    reload_eval = evaluate_predictive(
        reload_model, reload_predictor, reload_sketcher,
        valid_rows[:min(32, len(valid_rows))], args, args.pad, bos,
    )
    reference_eval = evaluate_predictive(
        model, predictor, sketcher, valid_rows[:min(32, len(valid_rows))], args, args.pad, bos,
    )
    summary = {
        "claim": CLAIM, "mode": args.mode, "arm": arm,
        "host": socket.gethostname(), "step": step, "cursor": cursor,
        "processed_tokens": processed_tokens, "initial": initial, "final": final,
        "generation": generation, "first_aux_level_grad_norms": first_aux_level_grad,
        "model_sha256": checkpoint["trainable_state_sha256"],
        "predictor_sha256": checkpoint["predictor_state_sha256"],
        "sketch_sha256": sketcher.digest(),
        "gates": {
            "steps_complete": step == args.steps,
            "finite": finite and math.isfinite(final["nll"]),
            "source_frozen": source_before == source_after,
            "model_updated": initial_model_hash != checkpoint["trainable_state_sha256"],
            "predictor_updated": initial_predictor_hash != checkpoint["predictor_state_sha256"],
            "reload": (
                abs(reference_eval["nll"] - reload_eval["nll"]) < 1e-9
                and abs(reference_eval["predictive_mse_mean"] - reload_eval["predictive_mse_mean"]) < 1e-9
                and reload_sketcher.digest() == sketcher.digest()
            ),
        },
        "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--parallel-data", default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv")
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11201)
    parser.add_argument("--ownership-seed", type=int, default=11202)
    parser.add_argument("--sketch-seed", type=int, default=11203)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--sketch-width", type=int, default=128)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--predictive-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--generation-examples", type=int, default=16)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-lines", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.mode == "smoke":
        args.steps = min(args.steps, 500)
        args.eval_rows = min(args.eval_rows, 128)
        args.generation_examples = min(args.generation_examples, 16)
    if args.max_target < 64:
        raise ValueError("max-target must cover the 64-piece D10 task target")

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
    _, warm = make_model(source_cpu, config, args, Path(args.warm_start))
    valid_rows, test_rows, excluded = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for arm in ARMS:
        summaries[arm] = train_arm(
            arm, source_cpu, config, Path(args.warm_start), warm,
            valid_rows, test_rows, excluded, args, sp, pieces, eos, bos, direction_ids,
        )
    probe = summaries["probe-only"]
    predictive = summaries["predictive-gradient"]
    initial_nll_delta = abs(probe["initial"]["nll"] - predictive["initial"]["nll"])
    initial_predictive_delta = abs(
        probe["initial"]["predictive_mse_mean"]
        - predictive["initial"]["predictive_mse_mean"]
    )
    active_aux_levels = sum(value > 1e-8 for value in predictive["first_aux_level_grad_norms"])
    mse_gain = probe["final"]["predictive_mse_mean"] - predictive["final"]["predictive_mse_mean"]
    nll_delta = predictive["final"]["nll"] - probe["final"]["nll"]
    gates = {
        "matched_initial_function": initial_nll_delta <= 1e-9,
        "matched_initial_predictor": initial_predictive_delta <= 1e-9,
        "fixed_target_match": probe["sketch_sha256"] == predictive["sketch_sha256"],
        "arm_safety": all(all(row["gates"].values()) for row in summaries.values()),
        "probe_blocks_aux_gradient": max(probe["first_aux_level_grad_norms"]) == 0.0,
        "predictive_gradient_reaches_levels": active_aux_levels >= 4,
        "heldout_predictive_mse_better": mse_gain > 0.0,
        "token_nll_not_destroyed": nll_delta <= 0.10,
    }
    comparison = {
        "claim": CLAIM, "mode": args.mode, "source_sha256": source_hash,
        "contract": {
            "same_model_initialization": initial_nll_delta <= 1e-9,
            "same_predictor_initialization": initial_predictive_delta <= 1e-9,
            "same_fixed_target": probe["sketch_sha256"] == predictive["sketch_sha256"],
            "same_data_seed_steps": True,
        },
        "metrics": {
            "initial_nll_delta": initial_nll_delta,
            "initial_predictive_mse_delta": initial_predictive_delta,
            "probe_final_nll": probe["final"]["nll"],
            "predictive_final_nll": predictive["final"]["nll"],
            "predictive_minus_probe_nll": nll_delta,
            "probe_final_predictive_mse": probe["final"]["predictive_mse_mean"],
            "predictive_final_predictive_mse": predictive["final"]["predictive_mse_mean"],
            "predictive_mse_gain": mse_gain,
            "probe_retrieval_at_1": probe["final"]["retrieval_at_1"],
            "predictive_retrieval_at_1": predictive["final"]["retrieval_at_1"],
            "probe_reverse_order_gap": probe["final"]["reverse_order_gap_mean"],
            "predictive_reverse_order_gap": predictive["final"]["reverse_order_gap_mean"],
            "active_aux_levels": active_aux_levels,
        },
        "gates": gates,
        "decision": "mechanism_supported" if all(gates.values()) else "mechanism_open",
        "not_proved": [
            "The fixed sketch is not yet a learned private protocol or a proof of semantic reasoning.",
            "A 500-step paired smoke cannot establish final BLEU or long-training advantage.",
            "This first rung supervises collective states at every convolved level, not each node independently.",
        ],
    }
    write_json(output / "comparison.json", comparison)
    print(json.dumps({"event": "complete", "decision": comparison["decision"], "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
