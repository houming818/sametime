#!/usr/bin/env python3
"""Matched continuous-depth READ smoke for STONE-2 C05."""

from __future__ import annotations

import argparse
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
import s3_stone2_grouped_read_matched_smoke as c04  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


ARMS = ("shared", "smooth_depth", "shuffled_depth_control")


class ContinuousDepthAdapter(nn.Module):
    """Low-rank READ residual with fixed linear depth bases (1, z)."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.common_down = nn.Linear(2 * dim, rank, bias=False)
        self.common_up = nn.Linear(rank, dim, bias=False)
        self.depth_down = nn.Linear(2 * dim, rank, bias=False)
        self.depth_up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.common_up.weight)
        nn.init.zeros_(self.depth_up.weight)

    def forward(self, query, local, z: float):
        features = torch.cat((query, local), dim=-1)
        common = self.common_up(torch.nn.functional.gelu(self.common_down(features)))
        depth = self.depth_up(torch.nn.functional.gelu(self.depth_down(features)))
        return common + z * depth


def depth_coordinate(arm: str, depth: int, count: int) -> float:
    if count <= 1:
        return 0.0
    mapped = depth
    if arm == "shuffled_depth_control":
        # 0,4,1,5,2,6,3,7 for eight levels: same coordinates, lost adjacency.
        half = (count + 1) // 2
        mapped = depth // 2 + (half if depth % 2 else 0)
        mapped = min(mapped, count - 1)
    return 2.0 * mapped / (count - 1) - 1.0


def make_adapted_read(arm: str):
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
                update = self.read_kernel(query, local, depth_state)
                z = depth_coordinate(arm, depth, len(tree))
                update = update + self.depth_adapter(query, local, z)
                query = query + gain * update
            safe = frontier.clamp_min(1e-12)
            entropies.append(-(safe * safe.log()).sum(-1).mean())
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2]
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            branch_query = (self.branch(hidden) + gain * (query - base_query))[:, None, None]
            scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
            scores = scores.masked_fill(~child_valid, -1e9)
            probability = torch.nn.functional.softmax(scores, dim=-1)
            probability = probability * child_valid.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return local + (query - base_query), torch.stack(entropies)

    return read


def build_arm(payload, arm, device, vocab, pad, rank):
    config = SimpleNamespace(**payload["config"])
    config.device = device
    model = integrated.build_integrated_model(config, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.decoder.read_kernel.parameters())
    if arm != "shared":
        dim = model.decoder.depth_embedding.weight.shape[1]
        model.decoder.depth_adapter = ContinuousDepthAdapter(dim, rank).to(device)
        model.decoder.read = MethodType(make_adapted_read(arm), model.decoder)
        trainable += list(model.decoder.depth_adapter.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    model.eval()
    return model, trainable, config


def adapter_stats(model, arm):
    if arm == "shared":
        return {"gradient_seen": [True], "output_norm": 0.0}
    adapter = model.decoder.depth_adapter
    output_norm = float(
        adapter.common_up.weight.detach().norm() + adapter.depth_up.weight.detach().norm()
    )
    return {"output_norm": output_norm}


def train_arm(model, trainable, arm, schedule, valid_rows, pad, bos, config, output, steps):
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    trace_path = output / f"{arm}.trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    initial = c11.evaluate(model, valid_rows, pad, bos, config.device, config.eval_batch)
    train_loss = 0.0
    train_tokens = 0
    finite = True
    started = time.time()
    for step, batch in enumerate(schedule, 1):
        model.train()
        _, _, loss, tokens = c04.batch_logits(model, batch, pad, bos, config.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_step = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        finite &= finite_step
        if not finite_step:
            raise RuntimeError(f"non-finite gradient in {arm} step {step}")
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
            c04.append_jsonl(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return {
        "initial_valid": initial,
        "final_valid": c11.evaluate(
            model, valid_rows, pad, bos, config.device, config.eval_batch
        ),
        "train_nll": train_loss / max(1, train_tokens),
        "train_tokens": train_tokens,
        "finite_gradients": finite,
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=16501)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--rank", type=int, default=32)
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
    output = args.evidence_dir
    output.mkdir(parents=True, exist_ok=True)

    models, trainables, configs = {}, {}, {}
    for arm in ARMS:
        torch.manual_seed(args.seed)
        models[arm], trainables[arm], configs[arm] = build_arm(
            payload, arm, args.device, pieces + 3, pad, args.rank
        )

    initial_logits, initial_valid, initial_test = {}, {}, {}
    for arm in ARMS:
        logits, _, _, _ = c04.batch_logits(models[arm], schedule[0], pad, bos, args.device)
        initial_logits[arm] = logits.detach()
        initial_valid[arm] = c11.evaluate(
            models[arm], valid_rows, pad, bos, args.device, config.eval_batch
        )
        initial_test[arm] = c11.evaluate(
            models[arm], test_rows, pad, bos, args.device, config.eval_batch
        )
    logit_delta = max(
        float((initial_logits["shared"] - initial_logits[arm]).abs().max())
        for arm in ARMS[1:]
    )
    valid_delta = max(
        abs(initial_valid["shared"]["nll"] - initial_valid[arm]["nll"])
        for arm in ARMS[1:]
    )
    test_delta = max(
        abs(initial_test["shared"]["nll"] - initial_test[arm]["nll"])
        for arm in ARMS[1:]
    )

    arms = {}
    for arm in ARMS:
        arms[arm] = train_arm(
            models[arm], trainables[arm], arm, schedule, valid_rows,
            pad, bos, configs[arm], output, args.steps
        )
        arms[arm].update(adapter_stats(models[arm], arm))
        if arm != "shared":
            adapter = models[arm].decoder.depth_adapter
            arms[arm]["adapter_basis_gradient_seen"] = [
                bool(module.weight.grad is not None and float(module.weight.grad.norm()) > 0.0)
                for module in (adapter.common_up, adapter.depth_up)
            ]
        arms[arm]["final_test"] = c11.evaluate(
            models[arm], test_rows, pad, bos, args.device, config.eval_batch
        )
        arms[arm]["generation"] = c10.task_generation_metrics(
            models[arm], test_rows, config, sp, pad, bos, eos, pieces, limit=8
        )

    shared = arms["shared"]
    smooth = arms["smooth_depth"]
    control = arms["shuffled_depth_control"]
    gates = {
        "G0_exact_initial_function": logit_delta < 1e-6 and valid_delta < 1e-7 and test_delta < 1e-7,
        "G1_finite_and_bases_receive_gradient": all(
            row["finite_gradients"] for row in arms.values()
        ) and all(smooth["adapter_basis_gradient_seen"]) and all(control["adapter_basis_gradient_seen"]),
        "G2_smooth_beats_shared_valid_by_0_005": shared["final_valid"]["nll"] - smooth["final_valid"]["nll"] >= 0.005,
        "G3_smooth_beats_shuffled_valid_by_0_005": control["final_valid"]["nll"] - smooth["final_valid"]["nll"] >= 0.005,
        "G4_smooth_test_not_worse_than_shared_by_0_01": smooth["final_test"]["nll"] - shared["final_test"]["nll"] <= 0.01,
        "G5_adapter_diverges": smooth["output_norm"] > 1e-5,
    }
    decision = "candidate_for_multiseed" if all(gates.values()) else (
        "freedom_candidate_not_continuity" if gates["G2_smooth_beats_shared_valid_by_0_005"] and not gates["G3_smooth_beats_shuffled_valid_by_0_005"]
        else "continuous_depth_read_not_supported"
    )
    result = {
        "claim": "S3-STONE2-CONTINUOUS-DEPTH-READ-C05",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "stream_sha256": c10.stream_sha256(schedule),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "initial_equivalence": {
            "max_logits_abs_delta": logit_delta,
            "max_valid_nll_delta": valid_delta,
            "max_test_nll_delta": test_delta,
        },
        "trainable_parameters": {
            arm: sum(parameter.numel() for parameter in trainables[arm]) for arm in ARMS
        },
        "arms": arms,
        "gates": gates,
        "decision": decision,
        "formal_training_authorized": all(gates.values()),
    }
    c04.write_json(output / "summary.json", result)
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "decision": decision,
        "valid_nll": {arm: arms[arm]["final_valid"]["nll"] for arm in ARMS},
        "test_nll": {arm: arms[arm]["final_test"]["nll"] for arm in ARMS},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
