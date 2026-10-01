#!/usr/bin/env python3
"""A21 frozen-encoder multiscale Flat READ comparison."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from s1_conditional_residual_evidence_fold_a20 import (
    ConditionalResidualCodec,
    RMSNorm,
    make_features,
)
from s1_recursive_context_codec_dimension_ladder_a19 import (
    EPS,
    evaluate_logits,
    file_sha256,
    fixed_batches,
    pair_nll,
    state_hash,
    tensor_hash,
    write_json,
)


ARMS = ("root_unfold", "leaf_only", "flat_coarse", "flat_all")


def extract_frozen_levels(
    encoder: ConditionalResidualCodec,
    features: torch.Tensor,
    probability: torch.Tensor,
) -> List[torch.Tensor]:
    state = encoder.leaf_state(features)
    mass = probability
    levels = [state.detach()]
    for _ in range(encoder.levels):
        left_mass, right_mass = mass[:, 0::2], mass[:, 1::2]
        state = encoder.fold_pair(
            state[:, 0::2], state[:, 1::2], left_mass, right_mass
        )
        mass = left_mass + right_mass
        levels.append(state.detach())
    return levels


class FlatReadProbe(nn.Module):
    def __init__(self, dimension: int, levels: int, global_logit: torch.Tensor, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.dimension = dimension
        self.levels = levels
        self.arm = arm
        self.register_buffer("global_logit", global_logit.reshape(1, -1))
        self.read_candidate = nn.Linear(dimension, 2 * dimension)
        self.read_gate = nn.Linear(dimension, 2 * dimension)
        self.read_norm = RMSNorm(dimension)
        self.level_score = nn.Linear(dimension, 1, bias=False)
        self.level_bias = nn.Parameter(torch.zeros(levels + 1))
        self.output = nn.Linear(dimension, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def read_once(self, parent: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.read_candidate(parent)).reshape(
            *parent.shape[:-1], 2, self.dimension
        )
        gate = torch.sigmoid(self.read_gate(parent)).reshape(
            *parent.shape[:-1], 2, self.dimension
        )
        return self.read_norm(parent.unsqueeze(-2) + gate * candidate)

    def aligned_level_views(self, states: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        views = [states[0]]
        for level in range(1, self.levels + 1):
            children = self.read_once(states[level]).reshape(
                states[level].shape[0], -1, self.dimension
            )
            views.append(children.repeat_interleave(2 ** (level - 1), dim=1))
        return views

    def decode(
        self,
        states: Sequence[torch.Tensor],
        disabled_levels: Sequence[int] = (),
        return_attention: bool = False,
    ):
        if self.arm == "root_unfold":
            value = states[-1]
            for _ in range(self.levels):
                value = self.read_once(value).reshape(value.shape[0], -1, self.dimension)
            logits = self.global_logit + self.output(value).squeeze(-1)
            return (logits, None) if return_attention else logits

        if self.arm == "leaf_only":
            logits = self.global_logit + self.output(states[0]).squeeze(-1)
            return (logits, None) if return_attention else logits

        views = self.aligned_level_views(states)
        selected = list(range(1, self.levels + 1)) if self.arm == "flat_coarse" else list(
            range(self.levels + 1)
        )
        disabled = set(disabled_levels)
        selected = [level for level in selected if level not in disabled]
        if not selected:
            raise ValueError("all readable levels were disabled")
        stack = torch.stack([views[level] for level in selected], dim=-2)
        bias = self.level_bias[selected].reshape(1, 1, -1)
        scores = self.level_score(stack).squeeze(-1) + bias
        attention = torch.softmax(scores, dim=-1)
        value = (attention.unsqueeze(-1) * stack).sum(dim=-2)
        logits = self.global_logit + self.output(value).squeeze(-1)
        if return_attention:
            return logits, {str(level): float(attention[..., index].mean().detach().cpu())
                            for index, level in enumerate(selected)}
        return logits


def subset_states(states: Sequence[torch.Tensor], batch: torch.Tensor) -> List[torch.Tensor]:
    return [state[batch] for state in states]


def train_arm(
    arm: str,
    frozen_states: Sequence[torch.Tensor],
    global_logit: torch.Tensor,
    fit_counts: torch.Tensor,
    dev_counts: torch.Tensor,
    test_counts: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    eval_epochs: Sequence[int],
    out: Path,
) -> Dict[str, object]:
    device = frozen_states[0].device
    dimension = frozen_states[0].shape[-1]
    levels = len(frozen_states) - 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FlatReadProbe(dimension, levels, global_logit, arm).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(frozen_states[0]) / batch_size)
    total_steps = epochs * steps_per_epoch
    eval_steps = {0, total_steps}
    for epoch in eval_epochs:
        if epoch <= epochs:
            eval_steps.add(epoch * steps_per_epoch)
    trajectory = []
    finite = True
    maximum_gradient_norm = 0.0
    started = time.time()

    def snapshot(step: int) -> None:
        model.eval()
        with torch.no_grad():
            logits, attention = model.decode(frozen_states, return_attention=True)
            row = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "fit": evaluate_logits(logits, fit_counts),
                "dev": evaluate_logits(logits, dev_counts),
                "test": evaluate_logits(logits, test_counts),
                "attention": attention,
            }
        trajectory.append(row)
        print(json.dumps({"arm": arm, **row}), flush=True)
        model.train()

    snapshot(0)
    for step, batch_cpu in fixed_batches(
        len(frozen_states[0]), batch_size, epochs, seed + 17
    ):
        batch = batch_cpu.to(device)
        logits = model.decode(subset_states(frozen_states, batch))
        loss = pair_nll(logits, fit_counts[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        finite = finite and bool(torch.isfinite(loss)) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        gradient_norm = math.sqrt(sum(
            float(gradient.detach().square().sum().cpu()) for gradient in gradients
        ))
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        if not finite:
            raise RuntimeError(f"non-finite state arm={arm} step={step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step + 1 in eval_steps:
            snapshot(step + 1)

    model.eval()
    with torch.no_grad():
        final_logits, attention = model.decode(frozen_states, return_attention=True)
        final = {
            "fit": evaluate_logits(final_logits, fit_counts),
            "dev": evaluate_logits(final_logits, dev_counts),
            "test": evaluate_logits(final_logits, test_counts),
        }
        controls = {}
        if arm == "flat_all":
            no_coarse = model.decode(frozen_states, disabled_levels=range(1, levels + 1))
            controls["disable_all_coarse_test"] = evaluate_logits(no_coarse, test_counts)
            controls["level_only_test"] = {}
            for level in range(levels + 1):
                disabled = [other for other in range(levels + 1) if other != level]
                logits = model.decode(frozen_states, disabled_levels=disabled)
                controls["level_only_test"][str(level)] = evaluate_logits(logits, test_counts)

    checkpoint = out / f"{arm}.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = FlatReadProbe(dimension, levels, global_logit, arm).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        reload_logits = reloaded.decode(frozen_states)
        reload_max_abs = float((reload_logits - final_logits).abs().max().cpu())
    initial = FlatReadProbe(dimension, levels, global_logit, arm).to(device)
    initial.load_state_dict(initial_state)
    initial.eval()
    with torch.no_grad():
        initial_logits = initial.decode(frozen_states)
    result = {
        "arm": arm,
        "dimension": dimension,
        "levels": levels,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_with_gradient": sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.grad is not None),
        "epochs": epochs,
        "steps": total_steps,
        "finite": finite,
        "maximum_gradient_norm": maximum_gradient_norm,
        "trajectory": trajectory,
        "final": final,
        "attention": attention,
        "controls": controls,
        "initial_logit_max_abs_from_prior": float(
            (initial_logits - global_logit.reshape(1, -1)).abs().max().cpu()
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "state_sha256": state_hash(model),
            "reload_max_abs": reload_max_abs,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(out / f"{arm}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--fit-ratio", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--eval-epochs", default="1,5,10,20,50,100,150,200")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    if any(arm not in ARMS for arm in arms):
        raise ValueError(f"arms must be drawn from {ARMS}")
    eval_epochs = [int(value) for value in args.eval_epochs.split(",") if value.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts_path = Path(args.counts)
    source_checkpoint = Path(args.source_checkpoint)
    payload = torch.load(counts_path, map_location="cpu", weights_only=False)
    original_train = payload["train"].to(torch.float64)
    sealed_test = payload["test"].to(torch.float64)
    levels = int(math.log2(original_train.shape[1]))
    generator = torch.Generator(device="cpu").manual_seed(args.split_seed)
    fit = torch.binomial(
        original_train,
        torch.full_like(original_train, args.fit_ratio),
        generator=generator,
    )
    dev = original_train - fit
    global_probability = (
        (fit.sum(dim=0) + args.alpha) /
        (fit.sum() + args.alpha * fit.shape[1])
    )
    probability = (
        (fit + args.alpha) /
        (fit.sum(dim=1, keepdim=True) + args.alpha * fit.shape[1])
    )
    device = torch.device(args.device)
    probability = probability.to(device=device, dtype=torch.float32)
    global_probability = global_probability.to(device=device, dtype=torch.float32)
    features = make_features(probability, global_probability)
    fit = fit.to(device=device, dtype=torch.float32)
    dev = dev.to(device=device, dtype=torch.float32)
    sealed_test = sealed_test.to(device=device, dtype=torch.float32)
    feature_mean = features.mean(dim=(0, 1))
    feature_std = features.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    global_logit = global_probability.clamp_min(EPS).log()

    encoder = ConditionalResidualCodec(
        args.dimension, levels, feature_mean, feature_std, global_logit,
        "evidence_fold",
    ).to(device)
    encoder.load_state_dict(torch.load(source_checkpoint, map_location=device,
                                       weights_only=True))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        source_logits, source_root, _ = encoder(features, probability)
        frozen_states = extract_frozen_levels(encoder, features, probability)
    source_encoder_hash = state_hash(encoder)
    frozen_hashes = [tensor_hash(state) for state in frozen_states]
    baseline_logits = global_logit.reshape(1, -1).expand(len(probability), -1)
    global_baseline = evaluate_logits(baseline_logits, sealed_test)
    source_native = evaluate_logits(source_logits, sealed_test)

    material = {
        "claims": [
            "S1-MULTISCALE-FLAT-READ-A21-C01",
            "S1-MULTISCALE-FLAT-READ-A21-C02",
        ],
        "counts": str(counts_path),
        "counts_sha256": file_sha256(counts_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_encoder_state_sha256": source_encoder_hash,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "shape": list(original_train.shape),
        "levels": levels,
        "dimension": args.dimension,
        "frozen_level_shapes": [list(state.shape) for state in frozen_states],
        "frozen_level_sha256": frozen_hashes,
        "frozen_root_sha256": tensor_hash(source_root),
        "global_prior_test": global_baseline,
        "source_native_test": source_native,
        "config": vars(args),
    }
    write_json(out / "material.json", material)

    results = []
    for arm in arms:
        result = train_arm(
            arm, frozen_states, global_logit, fit, dev, sealed_test,
            args.epochs, args.batch_size, args.learning_rate,
            args.weight_decay, args.seed, eval_epochs, out,
        )
        if state_hash(encoder) != source_encoder_hash:
            raise RuntimeError(f"frozen encoder changed while training arm={arm}")
        results.append(result)

    rows = []
    for result in results:
        rows.append({
            "arm": result["arm"],
            "test_nll": result["final"]["test"]["nll"],
            "test_ppl": result["final"]["test"]["ppl"],
            "test_row_top1": result["final"]["test"]["row_top1"],
            "parameters_total": result["parameters_total"],
            "parameters_with_gradient": result["parameters_with_gradient"],
            "finite": result["finite"],
            "full_budget": result["steps"] == args.epochs * math.ceil(
                len(probability) / args.batch_size
            ),
            "reload_exact": result["checkpoint"]["reload_max_abs"] == 0.0,
            "step0_exact": result["initial_logit_max_abs_from_prior"] == 0.0,
        })
    by_arm = {row["arm"]: row for row in rows}
    flat_result = next((result for result in results if result["arm"] == "flat_all"), None)
    flat_gain = None
    coarse_damage = None
    coarse_attention = None
    if "flat_all" in by_arm and "root_unfold" in by_arm:
        flat_gain = by_arm["root_unfold"]["test_nll"] - by_arm["flat_all"]["test_nll"]
    if flat_result is not None:
        coarse_damage = (
            flat_result["controls"]["disable_all_coarse_test"]["nll"] -
            flat_result["final"]["test"]["nll"]
        )
        coarse_attention = sum(
            value for level, value in flat_result["attention"].items() if level != "0"
        )
    integrity = {
        "all_arms_completed": len(rows) == len(arms),
        "all_finite": all(row["finite"] for row in rows),
        "all_full_budget": all(row["full_budget"] for row in rows),
        "all_reload_exact": all(row["reload_exact"] for row in rows),
        "all_step0_exact": all(row["step0_exact"] for row in rows),
        "encoder_unchanged": state_hash(encoder) == source_encoder_hash,
    }
    claims = {
        "flat_exposure_c01": flat_gain is not None and flat_gain >= 0.005,
        "coarse_contribution_c02": (
            coarse_damage is not None and coarse_damage >= 0.005 and
            coarse_attention is not None and coarse_attention >= 0.10
        ),
    }
    summary = {
        "claims": material["claims"],
        "global_prior_test": global_baseline,
        "source_native_test": source_native,
        "rows": rows,
        "flat_gain_over_root_unfold": flat_gain,
        "flat_disable_coarse_damage": coarse_damage,
        "flat_coarse_attention": coarse_attention,
        "integrity": integrity,
        "claim_gates": claims,
        "supported": all(integrity.values()) and all(claims.values()),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
