#!/usr/bin/env python3
"""Train matched scalarized and grouped orthogonal TreeHeap routers."""
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

import s3_pretrain_task_posterior_pipeline as c10
import s3_filter_guided_theta_calibration_f04 as f04
import s3_structural_protocol_full_pipeline_d10 as d10
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13"
NATIVE_SCALE = math.sqrt(0.5)
U_LIMIT = 0.5
MAX_MERGES = 5


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class LearnedGroupedFold(nn.Module):
    """Predict bounded orthogonal rotations from local child states."""

    def __init__(self, dim: int, rank: int, groups: int, scalarized: bool):
        super().__init__()
        if dim % groups:
            raise ValueError(f"groups={groups} does not divide dim={dim}")
        self.dim = dim
        self.rank = rank
        self.groups = groups
        self.scalarized = scalarized
        self.projection = nn.Linear(3 * dim, rank, bias=False)
        self.head = nn.Parameter(torch.zeros(MAX_MERGES, groups, rank))
        self.bias = nn.Parameter(torch.zeros(MAX_MERGES, groups))
        self.last_u_rows: list[torch.Tensor] = []
        self.last_energy_relative_error = 0.0

    def forward(self, slots, slot_mask, filter_u=None):
        if filter_u is not None:
            raise ValueError("F13 does not combine an external filter")
        batch, _, dim = slots.shape
        group_width = dim // self.groups
        levels, masks = [slots], [slot_mask]
        node, valid = slots, slot_mask
        u_rows = []
        max_energy_error = 0.0
        merge = 0
        while node.shape[1] > 1:
            if merge >= MAX_MERGES:
                raise RuntimeError("F13 requires more than five merge levels")
            left, right = node[:, 0::2], node[:, 1::2]
            left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
            both = left_valid & right_valid
            feature = torch.cat((
                F.layer_norm(left, (dim,)),
                F.layer_norm(right, (dim,)),
                F.layer_norm(left - right, (dim,)),
            ), dim=-1)
            hidden = torch.tanh(self.projection(feature))
            raw = self.bias[merge][None, None] + torch.einsum(
                "bnr,gr->bng", hidden, self.head[merge],
            ) / math.sqrt(self.rank)
            if self.scalarized:
                raw = raw.mean(-1, keepdim=True).expand_as(raw)
            u = U_LIMIT * torch.tanh(raw)

            nodes = left.shape[1]
            left_group = left.reshape(batch, nodes, self.groups, group_width)
            right_group = right.reshape(batch, nodes, self.groups, group_width)
            common = (left_group + right_group) * NATIVE_SCALE
            difference = (left_group - right_group) * NATIVE_SCALE
            angle = u[:, :, :, None]
            inv_norm = torch.rsqrt(1.0 + angle.square())
            parent_pair = ((common + angle * difference) * inv_norm).reshape_as(left)
            detail_pair = ((-angle * common + difference) * inv_norm).reshape_as(left)
            child = torch.where(left_valid[:, :, None], left, right)
            parent = torch.where(both[:, :, None], parent_pair, child)
            next_valid = left_valid | right_valid
            parent = parent * next_valid[:, :, None]
            if bool(both.any()):
                before = left.square().sum(-1) + right.square().sum(-1)
                after = parent_pair.square().sum(-1) + detail_pair.square().sum(-1)
                relative = ((after - before).abs() / before.clamp_min(1e-12))[both]
                max_energy_error = max(max_energy_error, float(relative.max().detach().cpu()))
                u_rows.append(u[both])
            levels.append(parent)
            masks.append(next_valid)
            node, valid = parent, next_valid
            merge += 1
        self.last_u_rows = u_rows
        self.last_energy_relative_error = max_energy_error
        return list(reversed(levels)), list(reversed(masks))

    def summary(self) -> dict:
        available = [row.detach().reshape(-1, self.groups) for row in self.last_u_rows if row.numel()]
        if not available:
            return {"u_abs_mean": 0.0, "u_group_std_mean": 0.0, "energy_relative_error": 0.0}
        values = torch.cat(available)
        return {
            "u_abs_mean": float(values.abs().mean().cpu()),
            "u_min": float(values.min().cpu()),
            "u_max": float(values.max().cpu()),
            "u_group_std_mean": float(values.std(-1, unbiased=False).mean().cpu()),
            "energy_relative_error": self.last_energy_relative_error,
        }


class LearnedGroupedTheta(nn.Module):
    def __init__(self, extra_dim: int, rank: int, groups: int, scalarized: bool):
        super().__init__()
        self.base = LearnedGroupedFold(256, rank, groups, scalarized)
        self.extra = LearnedGroupedFold(extra_dim, rank, groups, scalarized)

    def choose(self, dim: int):
        if dim == self.base.dim:
            return self.base
        if dim == self.extra.dim:
            return self.extra
        raise ValueError(f"unexpected protocol dimension {dim}")

    def summary(self) -> dict:
        return {"base": self.base.summary(), "extra": self.extra.summary()}


def states_exact(left: nn.Module, right: nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def theta_finite(theta: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in theta.parameters())


def evaluate_theta(model, theta, valid_rows, test_rows, base_args, sp, pad, bos, eos, pieces):
    with f04.routed_fold(theta):
        valid = d10.valid_summary(model, valid_rows, base_args, pad, bos)
        test = d10.valid_summary(model, test_rows, base_args, pad, bos)
        generation = d10.generation_summary(model, test_rows, base_args, sp, pad, bos, eos, pieces)
    return {"valid": valid, "test": test, "generation": generation, "theta": theta.summary()}


def train_arm(name, model, theta, schedule, depths, pad, bos, args, trace_path):
    if trace_path.exists():
        trace_path.unlink()
    optimizer = torch.optim.AdamW(theta.parameters(), lr=args.lr, weight_decay=1e-4)
    nonzero_gradient = False
    max_energy_error = 0.0
    for step, batch in enumerate(schedule, 1):
        depth = depths[step - 1]
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        with f04.routed_fold(theta) as router:
            logits, _, _, _, _ = model.teacher(source, lengths, target, bos, depth)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / target.ne(pad).sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in theta.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"{name}: absent or non-finite gradient at step {step}")
        nonzero_gradient |= any(float(gradient.abs().max()) > 1e-12 for gradient in gradients)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(theta.parameters(), 1.0))
        optimizer.step()
        if not theta_finite(theta):
            raise RuntimeError(f"{name}: non-finite theta at step {step}")
        current_error = max(theta.base.last_energy_relative_error, theta.extra.last_energy_relative_error)
        max_energy_error = max(max_energy_error, current_error)
        if step == 1 or step % args.log_every == 0 or step == len(schedule):
            event = {
                "event": "train", "arm": name, "step": step, "depth": depth,
                "loss": float(loss.detach().cpu()), "grad_norm": grad_norm,
                "theta": theta.summary(),
            }
            append_jsonl(trace_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
    return {"nonzero_gradient": nonzero_gradient, "max_energy_relative_error": max_energy_error}


def save_and_reload(name, theta, output, args, extra_dim):
    path = output / f"{name}.theta.pt"
    torch.save({"claim": CLAIM, "arm": name, "state_dict": theta.state_dict(), "config": vars(args)}, path)
    reloaded = LearnedGroupedTheta(
        extra_dim, args.rank, args.groups, scalarized=(name == "scalarized"),
    ).to(args.device)
    reloaded.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["state_dict"])
    return reloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12401)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--generation-examples", type=int, default=16)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    wmt_path = Path(args.wmt_data)

    payload, base_args, sp, model, pieces, eos, bos, source_hash = load_runtime(checkpoint, args.device)
    if model.extra is None:
        raise RuntimeError("F13 requires the TreeHeap-106M extra channel")
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    base_args.device = args.device
    base_args.eval_batch = args.batch_size
    base_args.generation_examples = args.generation_examples
    base_args.max_generation = args.max_generation
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    valid_rows, test_rows, excluded_pairs = d10.collect_wmt_eval(
        wmt_path, sp, direction_ids, eos, args.eval_rows,
    )
    iterator = d10.iter_parallel_batches(
        wmt_path, sp, direction_ids, eos, args.batch_size, 0, 0, excluded_pairs,
    )
    schedule = []
    for _, batch, _ in iterator:
        schedule.append(batch)
        if len(schedule) >= args.steps:
            break
    if len(schedule) != args.steps:
        raise RuntimeError(f"WMT stream ended at {len(schedule)}/{args.steps} batches")
    depth_rng = random.Random(args.seed + 1)
    depths = []
    while len(depths) < args.steps:
        block = list(DEPTHS)
        depth_rng.shuffle(block)
        depths.extend(block)
    depths = depths[:args.steps]

    scalarized = LearnedGroupedTheta(model.extra_dim, args.rank, args.groups, True).to(args.device)
    grouped = LearnedGroupedTheta(model.extra_dim, args.rank, args.groups, False).to(args.device)
    grouped.load_state_dict(copy.deepcopy(scalarized.state_dict()))
    matched_initial_state = states_exact(scalarized, grouped)
    scalar_parameters = sum(parameter.numel() for parameter in scalarized.parameters())
    grouped_parameters = sum(parameter.numel() for parameter in grouped.parameters())

    native = {
        "valid": d10.valid_summary(model, valid_rows, base_args, pieces, bos),
        "test": d10.valid_summary(model, test_rows, base_args, pieces, bos),
        "generation": d10.generation_summary(
            model, test_rows, base_args, sp, pieces, bos, eos, pieces,
        ),
    }
    scalar0 = evaluate_theta(
        model, scalarized, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )
    grouped0 = evaluate_theta(
        model, grouped, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )
    contract = {
        "claim": CLAIM, "host": socket.gethostname(), "config": vars(args),
        "checkpoint": str(checkpoint), "checkpoint_sha256": f04.sha256(checkpoint),
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_sha256": source_hash, "wmt_data": str(wmt_path),
        "valid_rows": len(valid_rows), "test_rows": len(test_rows),
        "excluded_pairs": len(excluded_pairs), "train_batches": len(schedule),
        "scalar_parameters": scalar_parameters, "grouped_parameters": grouped_parameters,
        "matched_initial_state": matched_initial_state,
        "step0": {
            "native": native, "scalarized": scalar0, "grouped": grouped0,
            "scalar_valid_nll_delta": abs(scalar0["valid"]["mean_nll"] - native["valid"]["mean_nll"]),
            "grouped_valid_nll_delta": abs(grouped0["valid"]["mean_nll"] - native["valid"]["mean_nll"]),
        },
    }
    write_json(output / "contract.json", contract)

    scalar_train = train_arm(
        "scalarized", model, scalarized, schedule, depths, pieces, bos, args,
        output / "scalarized.trace.jsonl",
    )
    grouped_train = train_arm(
        "grouped", model, grouped, schedule, depths, pieces, bos, args,
        output / "grouped.trace.jsonl",
    )
    scalar_eval = evaluate_theta(
        model, scalarized, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )
    grouped_eval = evaluate_theta(
        model, grouped, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )
    scalar_reload = save_and_reload("scalarized", scalarized, output, args, model.extra_dim)
    grouped_reload = save_and_reload("grouped", grouped, output, args, model.extra_dim)
    scalar_reload_eval = evaluate_theta(
        model, scalar_reload, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )
    grouped_reload_eval = evaluate_theta(
        model, grouped_reload, valid_rows, test_rows, base_args, sp, pieces, bos, eos, pieces,
    )

    grouped_std = (grouped_eval["theta"]["base"]["u_group_std_mean"] + grouped_eval["theta"]["extra"]["u_group_std_mean"]) * 0.5
    grouped_generation = grouped_eval["generation"]
    scalar_generation = scalar_eval["generation"]
    reload_delta = max(
        abs(scalar_reload_eval["test"]["mean_nll"] - scalar_eval["test"]["mean_nll"]),
        abs(grouped_reload_eval["test"]["mean_nll"] - grouped_eval["test"]["mean_nll"]),
    )
    gates = {
        "O0_matched_exact_origin": (
            scalar_parameters == grouped_parameters and matched_initial_state
            and contract["step0"]["scalar_valid_nll_delta"] <= 1e-7
            and contract["step0"]["grouped_valid_nll_delta"] <= 1e-7
        ),
        "O1_finite_gradient_and_energy": (
            scalar_train["nonzero_gradient"] and grouped_train["nonzero_gradient"]
            and scalar_train["max_energy_relative_error"] <= 1e-5
            and grouped_train["max_energy_relative_error"] <= 1e-5
        ),
        "O2_frozen_and_reload": f04.frozen_exact(model, expected_model) and reload_delta <= 1e-7,
        "P1_group_differentiation": grouped_std > 1e-4,
        "P2_heldout_nll_advantage": (
            grouped_eval["test"]["mean_nll"] <= scalar_eval["test"]["mean_nll"] - 0.005
        ),
        "P3_generation_health": (
            grouped_generation["bleu4_median"] >= scalar_generation["bleu4_median"]
            and grouped_generation["nonempty_min"] >= scalar_generation["nonempty_min"] - 0.02
            and grouped_generation["repetition_max"] <= scalar_generation["repetition_max"] + 0.02
        ),
    }
    summary = {
        "claim": CLAIM, "host": socket.gethostname(), "contract": contract,
        "training": {"scalarized": scalar_train, "grouped": grouped_train},
        "final": {"scalarized": scalar_eval, "grouped": grouped_eval},
        "comparison": {
            "grouped_minus_scalar_valid_nll": grouped_eval["valid"]["mean_nll"] - scalar_eval["valid"]["mean_nll"],
            "grouped_minus_scalar_test_nll": grouped_eval["test"]["mean_nll"] - scalar_eval["test"]["mean_nll"],
            "grouped_minus_scalar_bleu4_median": grouped_generation["bleu4_median"] - scalar_generation["bleu4_median"],
            "grouped_u_group_std_mean": grouped_std,
            "reload_test_nll_max_delta": reload_delta,
        },
        "gates": gates, "frozen_exact": f04.frozen_exact(model, expected_model),
        "seconds": time.time() - started,
        "claim_boundary": "frozen 106M checkpoint, 300-step matched WMT smoke; no default FOLD replacement",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# F13 learned TreeHeap polytope router smoke\n\n"
        f"- Claim: `{CLAIM}`\n"
        f"- Gates: `{json.dumps(gates, ensure_ascii=False)}`\n"
        f"- Comparison: `{json.dumps(summary['comparison'], ensure_ascii=False)}`\n"
        f"- Runtime seconds: `{summary['seconds']:.3f}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete", "gates": gates, "comparison": summary["comparison"],
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)
    if not all(gates[key] for key in (
        "O0_matched_exact_origin", "O1_finite_gradient_and_energy", "O2_frozen_and_reload",
    )):
        raise SystemExit(5)


if __name__ == "__main__":
    main()
