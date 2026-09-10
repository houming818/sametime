#!/usr/bin/env python3
"""Train matched radial and orthogonal directional TreeHeap theta heads."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import s3_filter_guided_theta_calibration_f04 as f04
import s3_cross_language_echo_controller_f08 as f08
import s3_structural_protocol_full_pipeline_d10 as d10
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-DIRECTIONAL-THETA-TRAINING-F11"
NATIVE_SCALE = math.sqrt(0.5)
U_LIMIT = 0.5


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class DirectionalAnnealingFold(nn.Module):
    """Content-conditioned bounded orthogonal parent, identity at theta zero."""

    def __init__(self, dim: int, rank: int, max_merges: int = 5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.max_merges = max_merges
        self.projection = nn.Linear(3 * dim, rank, bias=False)
        self.head = nn.Parameter(torch.zeros(max_merges, rank))
        self.bias = nn.Parameter(torch.zeros(max_merges))
        self.last_u_rows: list[torch.Tensor] = []
        self.last_energy_relative_error = 0.0

    def forward(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
        filter_u: torch.Tensor | None = None,
    ):
        if filter_u is not None:
            raise ValueError("F11 directional arm does not accept radial filter interventions")
        levels = [slots]
        masks = [slot_mask]
        node, valid = slots, slot_mask
        u_rows = []
        max_energy_error = 0.0
        merge = 0
        while node.shape[1] > 1:
            if merge >= self.max_merges:
                raise RuntimeError(f"F11 fold width requires more than {self.max_merges} merges")
            left, right = node[:, 0::2], node[:, 1::2]
            left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
            both = left_valid & right_valid
            common = (left + right) * NATIVE_SCALE
            difference = (left - right) * NATIVE_SCALE
            feature = torch.cat((
                F.layer_norm(left, (self.dim,)),
                F.layer_norm(right, (self.dim,)),
                F.layer_norm(left - right, (self.dim,)),
            ), dim=-1)
            hidden = torch.tanh(self.projection(feature))
            raw = self.bias[merge] + (
                hidden * self.head[merge][None, None]
            ).sum(-1) / math.sqrt(self.rank)
            u = U_LIMIT * torch.tanh(raw)
            inv_norm = torch.rsqrt(1.0 + u.square())
            parent_pair = (common + u[:, :, None] * difference) * inv_norm[:, :, None]
            detail_pair = (-u[:, :, None] * common + difference) * inv_norm[:, :, None]

            child = torch.where(left_valid[:, :, None], left, right)
            parent = torch.where(both[:, :, None], parent_pair, child)
            valid = left_valid | right_valid
            parent = parent * valid[:, :, None]
            if bool(both.any()):
                before = left.square().sum(-1) + right.square().sum(-1)
                after = parent_pair.square().sum(-1) + detail_pair.square().sum(-1)
                relative = ((after - before).abs() / before.clamp_min(1e-12))[both]
                max_energy_error = max(max_energy_error, float(relative.max().detach().cpu()))
                u_rows.append(u[both])
            levels.append(parent)
            masks.append(valid)
            node = parent
            merge += 1
        self.last_u_rows = u_rows
        self.last_energy_relative_error = max_energy_error
        return list(reversed(levels)), list(reversed(masks))

    def gain_summary(self) -> dict:
        available = [row.detach().flatten() for row in self.last_u_rows if row.numel()]
        if not available:
            return {
                "u_mean": 0.0, "u_abs_mean": 0.0, "u_min": 0.0, "u_max": 0.0,
                "energy_relative_error": self.last_energy_relative_error,
            }
        values = torch.cat(available)
        return {
            "u_mean": float(values.mean().cpu()),
            "u_abs_mean": float(values.abs().mean().cpu()),
            "u_min": float(values.min().cpu()),
            "u_max": float(values.max().cpu()),
            "energy_relative_error": self.last_energy_relative_error,
        }


class DirectionalAnnealingTheta(nn.Module):
    def __init__(self, base_dim: int, extra_dim: int, rank: int):
        super().__init__()
        self.base = DirectionalAnnealingFold(base_dim, rank)
        self.extra = DirectionalAnnealingFold(extra_dim, rank)

    def choose(self, dim: int) -> DirectionalAnnealingFold:
        if dim == self.base.dim:
            return self.base
        if dim == self.extra.dim:
            return self.extra
        raise ValueError(f"unexpected protocol dim {dim}")

    def gain_summary(self) -> dict:
        return {"base": self.base.gain_summary(), "extra": self.extra.gain_summary()}


def states_exact(left: nn.Module, right: nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


@torch.no_grad()
def hard_generation(
    model, theta, source, lengths, rows, concepts, sp, pieces, eos, bos, max_len,
):
    per_depth = {}
    total_all_positive = 0
    total_any_positive = 0
    total_nonempty = 0
    for depth in DEPTHS:
        _, generated, _ = f04.decode_logits(model, theta, source, lengths, bos, depth, max_len)
        outputs = []
        for index, row in enumerate(rows):
            ids = f04.clean(generated[index].tolist(), eos, pieces)
            hits = {
                name: any(f08.contains_subsequence(ids, target.tolist()) for target in concepts[name])
                for name in row["positive"]
            }
            all_positive = bool(hits) and all(hits.values())
            any_positive = any(hits.values())
            total_all_positive += int(all_positive)
            total_any_positive += int(any_positive)
            total_nonempty += int(bool(ids))
            outputs.append({
                "id": row["id"], "source": row["source"], "positive_hits": hits,
                "all_positive_hit": all_positive, "any_positive_hit": any_positive,
                "tokens": len(ids), "text": sp.decode(ids),
            })
        per_depth[str(depth)] = outputs
    return {
        "all_positive_hits": total_all_positive,
        "any_positive_hits": total_any_positive,
        "nonempty": total_nonempty,
        "opportunities": len(rows) * len(DEPTHS),
        "per_depth": per_depth,
    }


def train_arm(name, model, theta, source, lengths, rows, concepts, bos, eos, args, path):
    optimizer = torch.optim.AdamW(theta.parameters(), lr=args.lr, weight_decay=1e-4)
    nonzero_gradient = False
    for step in range(1, args.steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        logits, _, _ = f04.decode_logits(
            model, theta, source, lengths, bos, depth, args.behavior_steps,
        )
        loss, metrics = f04.behavior_loss(logits, rows, concepts, eos)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in theta.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"non-finite or absent gradient in {name} step {step}")
        nonzero_gradient |= any(float(gradient.abs().max()) > 1e-12 for gradient in gradients)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(theta.parameters(), 1.0))
        optimizer.step()
        if not f04.theta_finite(theta):
            raise RuntimeError(f"non-finite theta in {name} step {step}")
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            event = {
                "event": "train", "arm": name, "step": step, "depth": depth,
                "loss": float(loss.detach().cpu()), "behavior": metrics,
                "grad_norm": grad_norm, "theta": theta.gain_summary(),
            }
            append_jsonl(path, event)
            print(json.dumps({
                "event": "train", "arm": name, "step": step, "depth": depth,
                "loss": event["loss"], "positive": metrics["positive_coverage"],
                "negative": metrics["negative_activation"],
            }, ensure_ascii=False), flush=True)
    return {"nonzero_gradient": nonzero_gradient}


def evaluate_arm(
    name, model, theta, theta_class, output, train_source, train_lengths, train_rows,
    test_source, test_lengths, test_rows, concepts, sp, pieces, eos, bos,
    valid_rows, base_args, args, train_state,
):
    train_eval = f04.evaluate_rows(
        model, theta, train_source, train_lengths, train_rows, concepts, sp, pieces,
        eos, bos, args.behavior_steps, args.generation_steps,
    )
    test_eval = f04.evaluate_rows(
        model, theta, test_source, test_lengths, test_rows, concepts, sp, pieces,
        eos, bos, args.behavior_steps, args.generation_steps,
    )
    hard = hard_generation(
        model, theta, test_source, test_lengths, test_rows, concepts, sp, pieces,
        eos, bos, args.generation_steps,
    )
    with f04.routed_fold(theta):
        valid = d10.valid_summary(model, valid_rows, base_args, pieces, bos)
    checkpoint_path = output / f"{name}.theta.pt"
    torch.save({
        "claim": CLAIM, "arm": name, "theta_state_dict": theta.state_dict(),
        "config": vars(args),
    }, checkpoint_path)
    reloaded = theta_class(256, model.extra_dim, args.rank).to(args.device)
    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    reloaded.load_state_dict(saved["theta_state_dict"])
    reload_test = f04.evaluate_rows(
        model, reloaded, test_source, test_lengths, test_rows, concepts, sp, pieces,
        eos, bos, args.behavior_steps, args.generation_steps,
    )
    return {
        "train_state": train_state, "train": train_eval, "test": test_eval,
        "hard": hard, "valid": valid, "theta": theta.gain_summary(),
        "finite": f04.theta_finite(theta),
        "reload_test_loss_delta": abs(
            test_eval["mean"]["loss"] - reload_test["mean"]["loss"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--specimens", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12201)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--behavior-steps", type=int, default=24)
    parser.add_argument("--generation-steps", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=15)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    checkpoint = Path(args.checkpoint)
    specimen_path = Path(args.specimens)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)

    concepts_raw, rows = f04.load_specimens(specimen_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    payload, base_args, sp, model, pieces, eos, bos, source_hash = load_runtime(checkpoint, args.device)
    if model.extra is None:
        raise RuntimeError("F11 requires the TreeHeap-106M extra channel")
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    base_args.device = args.device
    base_args.eval_batch = args.eval_batch
    base_args.wmt_data = args.eval_wmt_data
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)
    train_source, train_lengths = f04.encode_rows(train_rows, sp, pieces, eos, pieces, args.device)
    test_source, test_lengths = f04.encode_rows(test_rows, sp, pieces, eos, pieces, args.device)
    all_source, all_lengths = f04.encode_rows(rows, sp, pieces, eos, pieces, args.device)

    radial = f04.AnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
    directional = DirectionalAnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
    directional.load_state_dict(radial.state_dict(), strict=True)
    matched_initial_state = states_exact(radial, directional)
    radial_parameters = sum(parameter.numel() for parameter in radial.parameters())
    directional_parameters = sum(parameter.numel() for parameter in directional.parameters())

    native_text = f04.native_outputs(
        model, all_source, all_lengths, rows, sp, pieces, eos, bos, args.generation_steps,
    )
    radial_text = f04.theta_outputs(
        model, radial, all_source, all_lengths, rows, sp, pieces, eos, bos, args.generation_steps,
    )
    directional_text = f04.theta_outputs(
        model, directional, all_source, all_lengths, rows, sp, pieces, eos, bos, args.generation_steps,
    )
    radial_text_exact = all(left[:3] == right[:3] for left, right in zip(native_text, radial_text))
    directional_text_exact = all(left[:3] == right[:3] for left, right in zip(native_text, directional_text))

    valid_rows, _, _ = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )
    native_valid = d10.valid_summary(model, valid_rows, base_args, pieces, bos)
    with f04.routed_fold(radial):
        radial_valid0 = d10.valid_summary(model, valid_rows, base_args, pieces, bos)
    with f04.routed_fold(directional):
        directional_valid0 = d10.valid_summary(model, valid_rows, base_args, pieces, bos)

    initial = {
        "train": f04.evaluate_rows(
            model, radial, train_source, train_lengths, train_rows, concepts, sp, pieces,
            eos, bos, args.behavior_steps, args.generation_steps,
        ),
        "test": f04.evaluate_rows(
            model, radial, test_source, test_lengths, test_rows, concepts, sp, pieces,
            eos, bos, args.behavior_steps, args.generation_steps,
        ),
        "hard": hard_generation(
            model, radial, test_source, test_lengths, test_rows, concepts, sp, pieces,
            eos, bos, args.generation_steps,
        ),
    }
    contract = {
        "claim": CLAIM, "host": socket.gethostname(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": f04.sha256(checkpoint),
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_sha256": source_hash,
        "specimens": str(specimen_path), "specimens_sha256": f04.sha256(specimen_path),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "radial_parameters": radial_parameters,
        "directional_parameters": directional_parameters,
        "matched_initial_state": matched_initial_state,
        "config": vars(args),
        "step0": {
            "radial_text_exact": radial_text_exact,
            "directional_text_exact": directional_text_exact,
            "native_valid": native_valid,
            "radial_valid": radial_valid0,
            "directional_valid": directional_valid0,
            "radial_nll_delta": abs(native_valid["mean_nll"] - radial_valid0["mean_nll"]),
            "directional_nll_delta": abs(native_valid["mean_nll"] - directional_valid0["mean_nll"]),
            "initial": initial,
        },
    }
    write_json(output / "contract.json", contract)

    radial_trace = output / "radial.trace.jsonl"
    directional_trace = output / "directional.trace.jsonl"
    for path in (radial_trace, directional_trace):
        if path.exists():
            path.unlink()
    radial_train = train_arm(
        "radial", model, radial, train_source, train_lengths, train_rows,
        concepts, bos, eos, args, radial_trace,
    )
    directional_train = train_arm(
        "directional", model, directional, train_source, train_lengths, train_rows,
        concepts, bos, eos, args, directional_trace,
    )

    arms = {
        "radial": evaluate_arm(
            "radial", model, radial, f04.AnnealingTheta, output,
            train_source, train_lengths, train_rows, test_source, test_lengths, test_rows,
            concepts, sp, pieces, eos, bos, valid_rows, base_args, args, radial_train,
        ),
        "directional": evaluate_arm(
            "directional", model, directional, DirectionalAnnealingTheta, output,
            train_source, train_lengths, train_rows, test_source, test_lengths, test_rows,
            concepts, sp, pieces, eos, bos, valid_rows, base_args, args, directional_train,
        ),
    }

    initial_test = initial["test"]["mean"]
    directional_test = arms["directional"]["test"]["mean"]
    radial_test = arms["radial"]["test"]["mean"]
    directional_u = arms["directional"]["theta"]
    max_abs_u = max(
        abs(directional_u[side][key])
        for side in ("base", "extra")
        for key in ("u_min", "u_max")
    )
    gates = {
        "O0_strict_origin": (
            radial_text_exact and directional_text_exact
            and contract["step0"]["radial_nll_delta"] <= 1e-8
            and contract["step0"]["directional_nll_delta"] <= 1e-8
        ),
        "O1_matched_parameters_and_initialization": (
            radial_parameters == directional_parameters and matched_initial_state
        ),
        "O2_gradient_frozen_reload": (
            radial_train["nonzero_gradient"] and directional_train["nonzero_gradient"]
            and arms["radial"]["finite"] and arms["directional"]["finite"]
            and f04.frozen_exact(model, expected_model)
            and arms["radial"]["reload_test_loss_delta"] <= 1e-7
            and arms["directional"]["reload_test_loss_delta"] <= 1e-7
        ),
        "P1_direction_moved": max_abs_u > 1e-4,
        "P2_heldout_soft_increment": (
            directional_test["positive_coverage"] > initial_test["positive_coverage"]
            and directional_test["positive_coverage"]
            >= radial_test["positive_coverage"] + 0.002
        ),
        "P3_heldout_hard_increment": (
            arms["directional"]["hard"]["all_positive_hits"]
            > arms["radial"]["hard"]["all_positive_hits"]
        ),
        "P4_behavior_health": (
            directional_test["negative_activation"] <= initial_test["negative_activation"] + 0.05
            and directional_test["expected_repetition"] <= initial_test["expected_repetition"] + 0.05
        ),
    }
    summary = {
        "claim": CLAIM, "host": socket.gethostname(),
        "contract": contract, "initial": initial, "arms": arms, "gates": gates,
        "comparison": {
            "directional_minus_radial_test_positive": (
                directional_test["positive_coverage"] - radial_test["positive_coverage"]
            ),
            "directional_minus_radial_hard_all": (
                arms["directional"]["hard"]["all_positive_hits"]
                - arms["radial"]["hard"]["all_positive_hits"]
            ),
            "directional_minus_radial_valid_nll": (
                arms["directional"]["valid"]["mean_nll"]
                - arms["radial"]["valid"]["mean_nll"]
            ),
            "max_abs_u": max_abs_u,
        },
        "frozen_exact": f04.frozen_exact(model, expected_model),
        "seconds": time.time() - started,
        "claim_boundary": "matched frozen-checkpoint lexical routing smoke; no production FOLD replacement or full translation claim",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# F11 matched radial vs directional theta smoke\n\n"
        f"- Claim: `{CLAIM}`\n"
        f"- Gates: `{json.dumps(gates, ensure_ascii=False)}`\n"
        f"- Comparison: `{json.dumps(summary['comparison'], ensure_ascii=False)}`\n"
        f"- Runtime seconds: `{summary['seconds']:.3f}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete", "gates": gates, "comparison": summary["comparison"],
        "initial_test_positive": initial_test["positive_coverage"],
        "radial_test_positive": radial_test["positive_coverage"],
        "directional_test_positive": directional_test["positive_coverage"],
        "radial_hard_all": arms["radial"]["hard"]["all_positive_hits"],
        "directional_hard_all": arms["directional"]["hard"]["all_positive_hits"],
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)
    if not all(gates[key] for key in ("O0_strict_origin", "O1_matched_parameters_and_initialization", "O2_gradient_frozen_reload")):
        raise SystemExit(5)


if __name__ == "__main__":
    main()
