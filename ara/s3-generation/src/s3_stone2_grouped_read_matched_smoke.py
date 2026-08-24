#!/usr/bin/env python3
"""Matched shared/grouped/interleaved READ-only smoke for STONE-2 C04."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


ARMS = ("shared", "resolution_grouped", "interleaved_control")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def kernel_index(arm: str, depth: int) -> int:
    if arm == "resolution_grouped":
        return 0 if depth <= 2 else 1 if depth <= 5 else 2
    if arm == "interleaved_control":
        return depth % 3
    raise ValueError(arm)


def make_banked_read(arm: str):
    def read(self, hidden, tree, masks, mode="native", ablate_depth=-1):
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
                kernel = self.read_kernel_bank[kernel_index(arm, depth)]
                query = query + gain * kernel(query, local, depth_state)
            entropies.append(
                -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1).mean()
            )
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            branch_query = (self.branch(hidden) + gain * (query - base_query))[:, None, None]
            scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
            scores = scores.masked_fill(~child_valid, -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * child_valid.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return local + (query - base_query), torch.stack(entropies)

    return read


def build_arm(payload, arm, device, vocab, pad):
    config = SimpleNamespace(**payload["config"])
    config.device = device
    model = integrated.build_integrated_model(config, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if arm == "shared":
        trainable = list(model.decoder.read_kernel.parameters())
    else:
        model.decoder.read_kernel_bank = nn.ModuleList(
            copy.deepcopy(model.decoder.read_kernel) for _ in range(3)
        )
        model.decoder.read = MethodType(make_banked_read(arm), model.decoder)
        trainable = list(model.decoder.read_kernel_bank.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    model.eval()
    return model, trainable, config


def batch_logits(model, batch, pad, bos, device):
    source, length, target = c10.collate_rows(batch, pad, device)
    logits, route = model.teacher(source, length, target, bos)
    tokens = int(target.ne(pad).sum())
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        ignore_index=pad, reduction="sum",
    )
    return logits, route, loss_sum / max(1, tokens), tokens


def bank_distances(model, arm):
    if arm == "shared":
        return []
    vectors = [
        torch.cat([parameter.detach().reshape(-1).cpu() for parameter in module.parameters()])
        for module in model.decoder.read_kernel_bank
    ]
    return [
        float((vectors[left] - vectors[right]).norm())
        for left in range(3) for right in range(left + 1, 3)
    ]


def train_arm(model, trainable, arm, schedule, valid_rows, pad, bos, config, output, steps):
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    trace_path = output / f"{arm}.trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    initial = c11.evaluate(model, valid_rows, pad, bos, config.device, config.eval_batch)
    gradient_seen = [False] * (1 if arm == "shared" else 3)
    train_loss = 0.0
    train_tokens = 0
    finite = True
    started = time.time()
    for step, batch in enumerate(schedule, 1):
        model.train()
        _, _, loss, tokens = batch_logits(model, batch, pad, bos, config.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_step = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        finite &= finite_step
        if not finite_step:
            raise RuntimeError(f"non-finite gradient in {arm} step {step}")
        if arm == "shared":
            gradient_seen[0] |= any(
                parameter.grad is not None and float(parameter.grad.norm()) > 0.0
                for parameter in trainable
            )
        else:
            for index, module in enumerate(model.decoder.read_kernel_bank):
                gradient_seen[index] |= any(
                    parameter.grad is not None and float(parameter.grad.norm()) > 0.0
                    for parameter in module.parameters()
                )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        train_loss += float(loss.detach()) * tokens
        train_tokens += tokens
        if step == 1 or step == steps or step % 40 == 0:
            valid = c11.evaluate(model, valid_rows, pad, bos, config.device, config.eval_batch)
            row = {
                "arm": arm,
                "step": step,
                "train_nll": train_loss / max(1, train_tokens),
                "valid_nll": valid["nll"],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    final = c11.evaluate(model, valid_rows, pad, bos, config.device, config.eval_batch)
    return {
        "initial_valid": initial,
        "final_valid": final,
        "train_nll": train_loss / max(1, train_tokens),
        "train_tokens": train_tokens,
        "finite_gradients": finite,
        "gradient_seen": gradient_seen,
        "bank_distances": bank_distances(model, arm),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=16401)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--eval-rows", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.task_train_rows = args.train_rows
    config.task_eval_rows = args.eval_rows
    sp = spm.SentencePieceProcessor(model_file=config.spm_model)
    pieces = sp.get_piece_size()
    pad, bos, eos = pieces, sp.bos_id(), sp.eos_id()
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    train_rows, valid_rows, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 1)
    stream_sha = c10.stream_sha256(schedule)
    output = args.evidence_dir
    output.mkdir(parents=True, exist_ok=True)

    models = {}
    trainables = {}
    configs = {}
    for arm in ARMS:
        models[arm], trainables[arm], configs[arm] = build_arm(
            payload, arm, args.device, pieces + 3, pad,
        )

    first_batch = schedule[0]
    initial_logits = {}
    initial_valid = {}
    initial_test = {}
    for arm in ARMS:
        logits, _, _, _ = batch_logits(models[arm], first_batch, pad, bos, args.device)
        initial_logits[arm] = logits.detach()
        initial_valid[arm] = c11.evaluate(
            models[arm], valid_rows, pad, bos, args.device, config.eval_batch,
        )
        initial_test[arm] = c11.evaluate(
            models[arm], test_rows, pad, bos, args.device, config.eval_batch,
        )
    init_logit_delta = max(
        float((initial_logits["shared"] - initial_logits[arm]).abs().max())
        for arm in ARMS[1:]
    )
    init_valid_delta = max(
        abs(initial_valid["shared"]["nll"] - initial_valid[arm]["nll"])
        for arm in ARMS[1:]
    )
    init_test_delta = max(
        abs(initial_test["shared"]["nll"] - initial_test[arm]["nll"])
        for arm in ARMS[1:]
    )

    arms = {}
    for arm in ARMS:
        arms[arm] = train_arm(
            models[arm], trainables[arm], arm, schedule, valid_rows,
            pad, bos, configs[arm], output, args.steps,
        )
        arms[arm]["final_test"] = c11.evaluate(
            models[arm], test_rows, pad, bos, args.device, config.eval_batch,
        )
        arms[arm]["generation"] = c10.task_generation_metrics(
            models[arm], test_rows, config, sp, pad, bos, eos, pieces, limit=8,
        )

    shared_valid = arms["shared"]["final_valid"]["nll"]
    grouped_valid = arms["resolution_grouped"]["final_valid"]["nll"]
    control_valid = arms["interleaved_control"]["final_valid"]["nll"]
    shared_test = arms["shared"]["final_test"]["nll"]
    grouped_test = arms["resolution_grouped"]["final_test"]["nll"]
    gates = {
        "G0_exact_initial_function": (
            init_logit_delta < 1e-6 and init_valid_delta < 1e-7 and init_test_delta < 1e-7
        ),
        "G1_finite_and_all_banks_receive_gradient": all(
            row["finite_gradients"] and all(row["gradient_seen"])
            for row in arms.values()
        ),
        "G2_grouped_beats_shared_valid_by_0_01": shared_valid - grouped_valid >= 0.01,
        "G3_grouped_beats_interleaved_valid_by_0_01": control_valid - grouped_valid >= 0.01,
        "G4_grouped_test_not_worse_than_shared_by_0_01": grouped_test - shared_test <= 0.01,
        "G5_grouped_banks_diverge": max(
            arms["resolution_grouped"]["bank_distances"]
        ) > 1e-5,
    }
    all_pass = all(gates.values())
    if all_pass:
        decision = "candidate_for_multiseed_grouped_read"
    elif gates["G2_grouped_beats_shared_valid_by_0_01"] and not gates[
        "G3_grouped_beats_interleaved_valid_by_0_01"
    ]:
        decision = "untying_candidate_but_resolution_grouping_not_supported"
    else:
        decision = "grouped_read_short_adaptation_not_supported"
    result = {
        "claim": "S3-STONE2-GROUPED-READ-C04",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "stream_sha256": stream_sha,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "initial_equivalence": {
            "max_logits_abs_delta": init_logit_delta,
            "max_valid_nll_delta": init_valid_delta,
            "max_test_nll_delta": init_test_delta,
        },
        "trainable_parameters": {
            arm: sum(parameter.numel() for parameter in trainables[arm]) for arm in ARMS
        },
        "arms": arms,
        "gates": gates,
        "decision": decision,
        "formal_training_authorized": False,
    }
    write_json(output / "summary.json", result)
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "decision": decision,
        "valid_nll": {arm: arms[arm]["final_valid"]["nll"] for arm in ARMS},
        "test_nll": {arm: arms[arm]["final_test"]["nll"] for arm in ARMS},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
