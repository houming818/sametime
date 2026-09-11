#!/usr/bin/env python3
"""Audit repeated-token accumulation through TreeHeap FOLD and READ."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import s3_filter_guided_theta_calibration_f04 as f04
import s3_pretrain_task_posterior_pipeline as c10
import s3_recursive_depth_pressure_protocol_training as d07
import s3_treeheap_polytope_probability_field_f12 as f12
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-COHERENT-ACCUMULATION-F14"
GROUPS = 8
MAX_MERGES = 5
SOURCE_SLOTS = 8
SOURCE_WIDTH = 16


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaled_fold(slots: torch.Tensor, slot_mask: torch.Tensor, scale: float):
    levels, masks = [slots], [slot_mask]
    node, valid = slots, slot_mask
    while node.shape[1] > 1:
        left, right = node[:, 0::2], node[:, 1::2]
        left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
        both = left_valid & right_valid
        parent_pair = (left + right) * scale
        child = torch.where(left_valid[:, :, None], left, right)
        parent = torch.where(both[:, :, None], parent_pair, child)
        valid = left_valid | right_valid
        parent = parent * valid[:, :, None]
        levels.append(parent)
        masks.append(valid)
        node = parent
    return list(reversed(levels)), list(reversed(masks))


@contextmanager
def routed_scale(scale: float):
    original = d07.fold_protocol
    d07.fold_protocol = lambda slots, mask: scaled_fold(slots, mask, scale)
    try:
        yield
    finally:
        d07.fold_protocol = original


def make_cases(push_id: int, neutral_id: int) -> list[dict]:
    cases = [{"id": "count0", "count": 0, "offset": 0, "tokens": [neutral_id] * SOURCE_SLOTS}]
    for count in (1, 2, 4):
        stride = SOURCE_SLOTS // count
        for offset in range(4):
            positions = {(offset + index * stride) % SOURCE_SLOTS for index in range(count)}
            tokens = [push_id if index in positions else neutral_id for index in range(SOURCE_SLOTS)]
            cases.append({
                "id": f"count{count}-offset{offset}", "count": count,
                "offset": offset, "positions": sorted(positions), "tokens": tokens,
            })
    cases.append({"id": "count8", "count": 8, "offset": 0, "tokens": [push_id] * SOURCE_SLOTS})
    return cases


def collate_cases(cases, direction_id: int, eos: int, pad: int, device: str):
    source = torch.full((len(cases), SOURCE_WIDTH), pad, dtype=torch.long, device=device)
    lengths = torch.full((len(cases),), SOURCE_SLOTS + 2, dtype=torch.long, device=device)
    for index, case in enumerate(cases):
        row = [direction_id, *case["tokens"], eos]
        source[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return source, lengths


def token_metrics(logits: torch.Tensor, token_id: int) -> list[dict]:
    step = logits[:, 0].float()
    target = step[:, token_id]
    competitors = step.clone()
    competitors[:, token_id] = -torch.inf
    competitor_lse = torch.logsumexp(competitors, dim=-1)
    probability = F.softmax(step, dim=-1)
    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(-1)
    rank = (step > target[:, None]).sum(-1) + 1
    return [{
        "target_logit": float(target[index].detach().cpu()),
        "target_probability": float(probability[index, token_id].detach().cpu()),
        "target_rank": int(rank[index].detach().cpu()),
        "target_margin": float((target[index] - competitor_lse[index]).detach().cpu()),
        "vocab_entropy": float(entropy[index].detach().cpu()),
    } for index in range(step.shape[0])]


def read_cutoff(decoder, hidden, tree, masks, cutoff: int):
    base_query = decoder.query(hidden)
    query = base_query
    frontier = masks[0].to(query.dtype)
    gain = torch.sigmoid(decoder.read_gain_logit)
    local = None
    entropy = None
    for depth, (nodes, valid) in enumerate(zip(tree, masks)):
        frontier = frontier * valid.to(frontier.dtype)
        frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
        local = (frontier[:, :, None] * nodes).sum(1)
        depth_state = decoder.depth_embedding.weight[depth][None].expand_as(local)
        query = query + gain * decoder.read_kernel(query, local, depth_state)
        entropy = -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1)
        if depth == cutoff:
            return local + (query - base_query), entropy, local.norm(dim=-1), frontier.max(-1).values
        children = tree[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2, nodes.shape[2])
        child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        branch_query = (decoder.branch(hidden) + gain * (query - base_query))[:, None, None]
        scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
        scores = scores.masked_fill(~child_valid, -1e9)
        probability = F.softmax(scores, dim=-1) * child_valid.to(scores.dtype)
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
        frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
    raise IndexError(cutoff)


def first_step_cutoff(decoder, levels, masks, bos: int, cutoff: int):
    tree = decoder.convolve(levels, masks)
    hidden = tree[0].new_zeros((tree[0].shape[0], decoder.hidden))
    context, entropy, local_norm, route_max = read_cutoff(decoder, hidden, tree, masks, cutoff)
    previous = torch.full((tree[0].shape[0],), bos, dtype=torch.long, device=tree[0].device)
    hidden = decoder.cell(torch.cat((decoder.embedding(previous), context), dim=-1), hidden)
    logits = decoder.output(torch.cat((hidden, context), dim=-1))[:, None, :]
    return logits, entropy, local_norm, route_max


def combined_cutoff(model, protocol, bos: int, cutoff: int):
    base_tree, base_masks, _, _, _, extra = protocol
    base_logits, base_entropy, base_norm, base_route = first_step_cutoff(
        model.reconstructor, base_tree, base_masks, bos, cutoff,
    )
    extra_entropy = extra_norm = extra_route = None
    if extra is not None:
        extra_tree, extra_masks, _, _ = extra
        extra_logits, extra_entropy, extra_norm, extra_route = first_step_cutoff(
            model.extra.reconstructor, extra_tree, extra_masks, bos, cutoff,
        )
        base_logits = base_logits + torch.tanh(model.extra_logit_gain) * extra_logits
    return base_logits, {
        "base_route_entropy": base_entropy, "base_local_norm": base_norm,
        "base_route_max": base_route, "extra_route_entropy": extra_entropy,
        "extra_local_norm": extra_norm, "extra_route_max": extra_route,
    }


def combined_teacher(model, protocol, target, bos: int, ablate_depth: int = -1):
    base_tree, base_masks, _, _, _, extra = protocol
    logits, _ = model.reconstructor.teacher(base_tree, base_masks, target, bos, ablate_depth=ablate_depth)
    if extra is not None:
        extra_tree, extra_masks, _, _ = extra
        extra_logits, _ = model.extra.reconstructor.teacher(
            extra_tree, extra_masks, target, bos, ablate_depth=ablate_depth,
        )
        logits = logits + torch.tanh(model.extra_logit_gain) * extra_logits
    return logits


def aggregate(rows: list[dict], mode: str, location: str) -> dict[str, float]:
    output = {}
    counts = sorted({row["count"] for row in rows})
    for count in counts:
        values = [
            row["target_margin"] for row in rows
            if row["mode"] == mode and row["location"] == location and row["count"] == count
        ]
        if values:
            output[str(count)] = sum(values) / len(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-steps", type=int, default=24)
    args = parser.parse_args()

    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload, base_args, sp, model, pieces, eos, bos, source_hash = load_runtime(
        Path(args.checkpoint), args.device,
    )
    f04.freeze_model(model)
    expected_state = payload["trainable_state_dict"]

    push_source = sp.encode("推", out_type=int)
    neutral_source = sp.encode("看", out_type=int)
    push_target = sp.encode("push", out_type=int)
    token_contract = {
        "push_source": push_source, "neutral_source": neutral_source,
        "push_target": push_target, "push_target_pieces": [sp.id_to_piece(value) for value in push_target],
    }
    if not (len(push_source) == len(neutral_source) == len(push_target) == 1):
        raise ValueError(f"single-token contract failed: {token_contract}")

    cases = make_cases(push_source[0], neutral_source[0])
    direction_id = pieces + 2
    source, lengths = collate_cases(cases, direction_id, eos, pieces, args.device)
    target = torch.tensor([[push_target[0], eos]] * len(cases), dtype=torch.long, device=args.device)
    write_json(output / "cases.json", {"token_contract": token_contract, "cases": cases})

    zero_base = torch.zeros((len(cases), MAX_MERGES, GROUPS), device=args.device)
    zero_extra = torch.zeros_like(zero_base)
    with torch.no_grad():
        native_logits, _, _, _, _ = model.teacher(source, lengths, target, bos, DEPTHS[0])
        with f12.routed_groupwise_fold(zero_base, zero_extra, GROUPS, model.extra_dim):
            zero_logits, _, _, _, _ = model.teacher(source, lengths, target, bos, DEPTHS[0])
    zero_parity = float((native_logits - zero_logits).abs().max().cpu())

    metric_rows = []
    ablation_rows = []
    gradient_rows = []
    generations = []
    finite = True

    for protocol_depth in DEPTHS:
        protocols = {}
        with torch.no_grad():
            protocols["native"] = model.protocol(source, lengths, protocol_depth)
            with routed_scale(0.5):
                protocols["mean"] = model.protocol(source, lengths, protocol_depth)

        for mode, protocol in protocols.items():
            level_count = len(protocol[0])
            for cutoff in range(level_count):
                with torch.no_grad():
                    logits, diagnostics = combined_cutoff(model, protocol, bos, cutoff)
                metrics = token_metrics(logits, push_target[0])
                for index, values in enumerate(metrics):
                    diagnostic = {
                        key: None if tensor is None else float(tensor[index].detach().cpu())
                        for key, tensor in diagnostics.items()
                    }
                    metric_rows.append({
                        "case": cases[index]["id"], "count": cases[index]["count"],
                        "offset": cases[index]["offset"], "protocol_depth": protocol_depth,
                        "mode": mode, "location": f"cutoff-{cutoff}",
                        "tree_nodes": int(protocol[0][cutoff].shape[1]), **values, **diagnostic,
                    })

            with torch.no_grad():
                full_logits = combined_teacher(model, protocol, target, bos)
            full_metrics = token_metrics(full_logits, push_target[0])
            for index, values in enumerate(full_metrics):
                metric_rows.append({
                    "case": cases[index]["id"], "count": cases[index]["count"],
                    "offset": cases[index]["offset"], "protocol_depth": protocol_depth,
                    "mode": mode, "location": "full", "tree_nodes": -1,
                    **values, "base_route_entropy": None, "base_local_norm": None,
                    "base_route_max": None, "extra_route_entropy": None,
                    "extra_local_norm": None, "extra_route_max": None,
                })
            finite = finite and bool(torch.isfinite(full_logits).all())

            if mode == "native":
                for ablate_depth in range(level_count):
                    with torch.no_grad():
                        ablated = combined_teacher(model, protocol, target, bos, ablate_depth)
                    ablated_metrics = token_metrics(ablated, push_target[0])
                    for index, values in enumerate(ablated_metrics):
                        ablation_rows.append({
                            "case": cases[index]["id"], "count": cases[index]["count"],
                            "offset": cases[index]["offset"], "protocol_depth": protocol_depth,
                            "ablate_depth": ablate_depth,
                            "margin_delta_vs_full": values["target_margin"] - full_metrics[index]["target_margin"],
                            **values,
                        })

        raw_base = torch.zeros(
            (len(cases), MAX_MERGES, GROUPS), device=args.device, requires_grad=True,
        )
        raw_extra = torch.zeros_like(raw_base, requires_grad=True)
        with f12.routed_groupwise_fold(raw_base, raw_extra, GROUPS, model.extra_dim):
            logits, _, _, _, _ = model.teacher(source, lengths, target, bos, protocol_depth)
        focus_loss = F.cross_entropy(
            logits[:, 0].float(), target[:, 0], reduction="sum",
        )
        grad_base, grad_extra = torch.autograd.grad(focus_loss, (raw_base, raw_extra))
        for channel, gradient in (("base", grad_base), ("extra", grad_extra)):
            finite = finite and bool(torch.isfinite(gradient).all())
            for case_index, case in enumerate(cases):
                for merge in range(MAX_MERGES):
                    for group in range(GROUPS):
                        gradient_rows.append({
                            "case": case["id"], "count": case["count"],
                            "offset": case["offset"], "protocol_depth": protocol_depth,
                            "channel": channel, "merge": merge, "group": group,
                            "gradient": float(gradient[case_index, merge, group].detach().cpu()),
                        })

        with torch.no_grad():
            generated, _, budgets, _, _ = model.greedy(
                source, lengths, bos, eos, args.generation_steps, protocol_depth,
            )
        for index, case in enumerate(cases):
            ids = f04.clean(generated[index].tolist(), eos, pieces)
            generations.append({
                "case": case["id"], "count": case["count"], "offset": case["offset"],
                "protocol_depth": protocol_depth, "budget": int(budgets[index]),
                "text": sp.decode(ids), "contains_push_token": push_target[0] in ids,
            })

    write_csv(output / "token_layers.csv", metric_rows)
    write_csv(output / "depth_ablations.csv", ablation_rows)
    write_csv(output / "group_gradients.csv", gradient_rows)
    write_json(output / "free_generation.json", generations)

    root_native = aggregate(metric_rows, "native", "cutoff-0")
    root_mean = aggregate(metric_rows, "mean", "cutoff-0")
    full_native = aggregate(metric_rows, "native", "full")
    root_ablation = {}
    for count in (1, 8):
        values = [
            -row["margin_delta_vs_full"] for row in ablation_rows
            if row["ablate_depth"] == 0 and row["count"] == count
        ]
        root_ablation[str(count)] = sum(values) / len(values)

    gates = {
        "O0_fixed_single_token_contract": (
            len(push_source) == len(neutral_source) == len(push_target) == 1
            and bool((lengths == SOURCE_SLOTS + 2).all()) and source.shape[1] == SOURCE_WIDTH
        ),
        "O1_zero_parity_and_finite": zero_parity <= 1e-7 and finite,
        "O2_frozen_checkpoint": f04.frozen_exact(model, expected_state),
        "P1_native_root_count_response": root_native["8"] > root_native["1"],
        "P2_native_full_count_response": full_native["8"] > full_native["1"],
        "P3_normalized_sum_root_advantage": (
            root_native["8"] - root_native["1"] > root_mean["8"] - root_mean["1"]
        ),
        "P4_root_read_causal_growth": root_ablation["8"] > root_ablation["1"],
    }
    summary = {
        "claim": CLAIM, "host": os.uname().nodename, "checkpoint": args.checkpoint,
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_state_sha256": source_hash, "token_contract": token_contract,
        "case_count": len(cases), "protocol_depths": list(DEPTHS),
        "source_true_length": SOURCE_SLOTS + 2, "source_width": SOURCE_WIDTH,
        "zero_grouped_native_max_logit_delta": zero_parity,
        "root_margin_by_count": {"native": root_native, "mean": root_mean},
        "full_margin_by_count": {"native": full_native},
        "root_ablation_damage": root_ablation, "gates": gates,
        "seconds": time.time() - started,
        "claim_boundary": "synthetic fixed-token mechanism audit; not natural-corpus translation evidence",
    }
    write_json(output / "summary.json", summary)
    readme = [
        f"# {CLAIM}", "", f"- Cases: `{len(cases)}`", f"- Gates: `{gates}`",
        f"- Native root margins: `{root_native}`", f"- Mean-FOLD root margins: `{root_mean}`",
        f"- Native full margins: `{full_native}`", f"- Runtime seconds: `{summary['seconds']:.2f}`",
        "", "This is a frozen synthetic mechanism audit, not a translation-quality result.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
