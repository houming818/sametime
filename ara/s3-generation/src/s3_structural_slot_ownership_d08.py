#!/usr/bin/env python3
"""Matched smoke for free, recursive-subheap, and random protocol-slot ownership."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402


CLAIM = "S3-STRUCTURAL-SLOT-OWNERSHIP-D08R1"
ARMS = ("free", "subheap", "random")
DEPTHS = d07.DEPTHS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def nonempty_subheap_frontier(valid: torch.Tensor, budget: int) -> list[tuple[int, int]]:
    """Split valid leaves into at most budget exact binary subheaps."""
    width = int(valid.numel())
    if width < 1 or width & (width - 1):
        raise ValueError("terminal TreeHeap width must be a positive power of two")
    valid_cpu = valid.detach().to("cpu", torch.bool)
    segments = [(0, width)]
    while len(segments) < budget:
        candidates = []
        for index, (start, end) in enumerate(segments):
            count = int(valid_cpu[start:end].sum())
            if end - start > 1 and count > 1:
                candidates.append((count, end - start, -start, index))
        if not candidates:
            break
        _, _, _, selected = max(candidates)
        start, end = segments[selected]
        middle = (start + end) // 2
        children = [
            (child_start, child_end)
            for child_start, child_end in ((start, middle), (middle, end))
            if bool(valid_cpu[child_start:child_end].any())
        ]
        segments[selected:selected + 1] = children
    return segments


def ownership_mask(
    terminal_valid: torch.Tensor,
    budgets: torch.Tensor,
    max_slots: int,
    mode: str,
    random_seed: int,
) -> torch.Tensor | None:
    """Assign every valid terminal leaf to one active protocol slot."""
    if mode == "free":
        return None
    batch, width = terminal_valid.shape
    owner = torch.zeros(
        batch, max_slots, width, dtype=torch.bool, device=terminal_valid.device,
    )
    for batch_index in range(batch):
        valid = terminal_valid[batch_index]
        valid_index = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        budget = min(int(budgets[batch_index]), len(valid_index), max_slots)
        frontier = nonempty_subheap_frontier(valid, budget)
        groups = [
            [index for index in valid_index if start <= index < end]
            for start, end in frontier
        ]
        groups = [group for group in groups if group]
        if mode == "random":
            # The random control must be stable for a given shape. Depending on
            # batch position would add protocol drift to the geometry control.
            generator = random.Random(
                random_seed + 1009 * len(valid_index) + 9176 * budget,
            )
            shuffled = list(valid_index)
            generator.shuffle(shuffled)
            sizes = [len(group) for group in groups]
            groups, offset = [], 0
            for size in sizes:
                groups.append(shuffled[offset:offset + size])
                offset += size
        for slot_index, group in enumerate(groups[:budget]):
            owner[batch_index, slot_index, group] = True
    return owner


def route_statistics(
    frontier: torch.Tensor,
    slot_mask: torch.Tensor,
    terminal_valid: torch.Tensor,
    owner: torch.Tensor | None,
) -> dict[str, float | None]:
    overlaps, cosines, argmax_coverages, owner_coverages = [], [], [], []
    frontier = frontier.detach().float().cpu()
    slot_mask = slot_mask.detach().cpu()
    terminal_valid = terminal_valid.detach().cpu()
    owner_cpu = owner.detach().cpu() if owner is not None else None
    for batch_index in range(frontier.shape[0]):
        active = int(slot_mask[batch_index].sum())
        valid = terminal_valid[batch_index]
        routes = frontier[batch_index, :active, valid]
        if active:
            routes = routes / routes.sum(-1, keepdim=True).clamp_min(1e-12)
            endpoints = routes.argmax(-1)
            argmax_coverages.append(float(endpoints.unique().numel() / active))
        if active > 1:
            for left in range(active):
                for right in range(left + 1, active):
                    overlaps.append(float(torch.minimum(routes[left], routes[right]).sum()))
                    denominator = routes[left].norm() * routes[right].norm()
                    cosines.append(float((routes[left] @ routes[right]) / denominator.clamp_min(1e-12)))
        if owner_cpu is not None and bool(valid.any()):
            covered = owner_cpu[batch_index, :active].any(0) & valid
            owner_coverages.append(float(covered.sum() / valid.sum()))
    return {
        "route_pair_overlap": sum(overlaps) / max(1, len(overlaps)),
        "route_pair_cosine": sum(cosines) / max(1, len(cosines)),
        "argmax_coverage": sum(argmax_coverages) / max(1, len(argmax_coverages)),
        "owner_leaf_coverage": (
            sum(owner_coverages) / len(owner_coverages) if owner_coverages else None
        ),
    }


class OwnershipCompressor(d07.DepthPressureCompressor):
    """D07 compressor with optional recursive address ownership."""

    def __init__(self, *args, ownership_mode: str, ownership_seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        if ownership_mode not in ARMS:
            raise ValueError(ownership_mode)
        self.ownership_mode = ownership_mode
        self.ownership_seed = ownership_seed
        self.last_route_statistics = None

    def forward(self, tree, masks, max_depth: int, budgets):
        batch = tree[0].shape[0]
        slot_index = torch.arange(self.max_slots, device=tree[0].device)
        query = self.slot_query(slot_index)[None].expand(batch, -1, -1)
        query = query + self.depth_embedding.weight[max_depth][None, None]
        last_depth = min(max_depth, len(tree) - 1)
        terminal_valid = masks[last_depth]
        owner = ownership_mask(
            terminal_valid, budgets, self.max_slots,
            self.ownership_mode, self.ownership_seed,
        )
        allowed = None
        if owner is not None:
            terminal_width = terminal_valid.shape[1]
            allowed = []
            for depth in range(last_depth + 1):
                width = masks[depth].shape[1]
                factor = terminal_width // width
                allowed.append(owner.reshape(batch, self.max_slots, width, factor).any(-1))

        frontier = masks[0][:, None].expand(-1, self.max_slots, -1).to(query.dtype)
        if allowed is not None:
            frontier = frontier * allowed[0].to(frontier.dtype)
        entropy, local = [], None
        for depth in range(last_depth + 1):
            nodes = tree[depth]
            valid = masks[depth]
            frontier = frontier * valid[:, None].to(frontier.dtype)
            if allowed is not None:
                frontier = frontier * allowed[depth].to(frontier.dtype)
            frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
            local = torch.einsum("bsn,bnd->bsd", frontier, nodes)
            depth_state = self.depth_embedding.weight[depth][None, None].expand_as(local)
            query = self.query_norm(query + self.read_kernel(query, local, depth_state))
            entropy.append(
                -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1).mean()
            )
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                batch, nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(batch, nodes.shape[1], 2)
            branch_query = self.branch(query)[:, :, None, None]
            scores = (branch_query * children[:, None]).sum(-1) / math.sqrt(self.dim)
            valid_child = child_valid[:, None]
            if allowed is not None:
                valid_child = valid_child & allowed[depth + 1].reshape(
                    batch, self.max_slots, nodes.shape[1], 2,
                )
            scores = scores.masked_fill(~valid_child, -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * valid_child.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, :, None] * probability).flatten(2)

        slot_mask = slot_index[None] < budgets[:, None]
        slots = self.slot_out(torch.cat((query, local), dim=-1))
        slots = slots * slot_mask[:, :, None]
        self.last_route_statistics = route_statistics(
            frontier, slot_mask, terminal_valid, owner,
        )
        return slots, slot_mask, torch.stack(entropy)


class OwnershipPressureProtocolModel(d07.PressureProtocolModel):
    def __init__(
        self, frozen_source, dim: int, hidden: int, max_slots: int,
        ownership_mode: str, ownership_seed: int,
    ):
        super().__init__(
            frozen_source, dim, hidden, max_slots,
            freeze_language_backbone=True, bounded_protocol_gain=True,
        )
        self.compressor = OwnershipCompressor(
            dim, max_slots, frozen_source.decoder.depth_embedding.num_embeddings,
            ownership_mode=ownership_mode, ownership_seed=ownership_seed,
        )


@torch.no_grad()
def evaluate(model, rows, pad, bos, device, batch_size, depth, intervention="native"):
    model.eval()
    loss_sum, tokens = 0.0, 0
    slot_sum, slot_square_sum, slot_values = 0.0, 0.0, 0
    between_slot_sum, between_slot_rows = 0.0, 0
    budgets, route_rows = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        source, lengths, target = c10.collate_rows(batch, pad, device)
        logits, _, local_budgets, slots, _ = model.teacher(
            source, lengths, target, bos, depth, intervention,
        )
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        local_tokens = int(target.ne(pad).sum())
        tokens += local_tokens
        active_mask = torch.arange(slots.shape[1], device=slots.device)[None] < local_budgets[:, None]
        active = slots[active_mask]
        slot_sum += float(active.sum())
        slot_square_sum += float(active.square().sum())
        slot_values += active.numel()
        for batch_index, budget in enumerate(local_budgets.tolist()):
            if budget > 1:
                between_slot_sum += float(
                    slots[batch_index, :budget].var(dim=0, unbiased=False).mean()
                )
                between_slot_rows += 1
        budgets.extend(int(value) for value in local_budgets.cpu())
        route_rows.append((len(batch), dict(model.compressor.last_route_statistics)))
    mean = slot_sum / max(1, slot_values)
    route = {}
    for key in ("route_pair_overlap", "route_pair_cosine", "argmax_coverage", "owner_leaf_coverage"):
        available = [(weight, row[key]) for weight, row in route_rows if row[key] is not None]
        route[key] = (
            sum(weight * value for weight, value in available) / sum(weight for weight, _ in available)
            if available else None
        )
    return {
        "nll": loss_sum / max(1, tokens),
        "ppl": math.exp(min(20.0, loss_sum / max(1, tokens))),
        "tokens": tokens,
        "mean_budget": sum(budgets) / max(1, len(budgets)),
        "slot_variance": max(0.0, slot_square_sum / max(1, slot_values) - mean * mean),
        "between_slot_variance": between_slot_sum / max(1, between_slot_rows),
        "route": route,
    }


def trainable_state(model):
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def run_arm(
    arm: str, frozen_cpu, config, source_hash: str, checkpoint: str,
    train_rows, valid_rows, test_rows, args, sp, pieces: int, pad: int, bos: int, eos: int,
):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frozen = copy.deepcopy(frozen_cpu).to(args.device)
    model = OwnershipPressureProtocolModel(
        frozen, config.dim, config.hidden, args.max_slots,
        ownership_mode=arm, ownership_seed=args.seed + 91,
    ).to(args.device)
    language_before = c10.state_sha256(d07.language_backbone_state(model))
    initial_trainable_hash = c10.state_sha256(trainable_state(model))
    initial = {
        str(depth): evaluate(model, valid_rows, pad, bos, args.device, args.batch_size, depth)
        for depth in DEPTHS
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 1)
    depth_rng = random.Random(args.seed + 2)
    depth_schedule = []
    while len(depth_schedule) < args.steps:
        block = list(DEPTHS)
        depth_rng.shuffle(block)
        depth_schedule.extend(block)
    arm_dir = Path(args.evidence_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    trace_path = arm_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    started, first_grad = time.time(), None
    for step, batch in enumerate(schedule, 1):
        model.train()
        depth = depth_schedule[step - 1]
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        logits, route, budgets, slots, entropy = model.teacher(source, lengths, target, bos, depth)
        token_count = int(target.ne(pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / max(1, token_count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not d07.finite_trainable_gradients(model):
            raise RuntimeError(f"{arm}: non-finite gradient at step {step}")
        current_grad = {
            "compressor_read": d07.group_grad_norm(model.compressor.read_kernel),
            "compressor_slot": d07.group_grad_norm(model.compressor.slot_out),
            "reconstructor_read": d07.group_grad_norm(model.reconstructor.read_kernel),
            "protocol_gain": float(model.protocol_gain_logit.grad.detach().abs()),
        }
        if first_grad is None:
            first_grad = current_grad
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            row = {
                "arm": arm, "step": step, "depth": depth,
                "train_nll": float(loss.detach()), "grad_norm": grad_norm,
                "grad_groups": current_grad,
                "mean_budget": float(budgets.float().mean()),
                "slot_variance": float(slots.detach().var()),
                "compressor_entropy": [float(value) for value in entropy.detach().cpu()],
                "decoder_route": [float(value) for value in route.detach().cpu()],
                "slot_route": model.compressor.last_route_statistics,
                "elapsed_seconds": time.time() - started,
            }
            d07.append_jsonl(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    final_valid = {
        str(depth): evaluate(model, valid_rows, pad, bos, args.device, args.batch_size, depth)
        for depth in DEPTHS
    }
    final_test = {
        str(depth): {
            "native": evaluate(model, test_rows, pad, bos, args.device, args.batch_size, depth),
            "shuffle": evaluate(model, test_rows, pad, bos, args.device, args.batch_size, depth, "shuffle"),
            "zero": evaluate(model, test_rows, pad, bos, args.device, args.batch_size, depth, "zero"),
            "generation": d07.generation_metrics(
                model, test_rows, args, sp, pad, bos, eos, pieces, depth,
            ),
        }
        for depth in DEPTHS
    }
    source_after = c10.state_sha256(model.frozen_source.state_dict())
    language_after = c10.state_sha256(d07.language_backbone_state(model))
    causal = {
        str(depth): {
            "shuffle": final_test[str(depth)]["shuffle"]["nll"] - final_test[str(depth)]["native"]["nll"],
            "zero": final_test[str(depth)]["zero"]["nll"] - final_test[str(depth)]["native"]["nll"],
        }
        for depth in DEPTHS
    }
    summary = {
        "claim": CLAIM, "arm": arm, "checkpoint": checkpoint,
        "initial_trainable_sha256": initial_trainable_hash,
        "source_sha256_before": source_hash, "source_sha256_after": source_after,
        "language_sha256_before": language_before, "language_sha256_after": language_after,
        "protocol_gain_initial": float(torch.sigmoid(torch.tensor(-4.0))),
        "protocol_gain_final": float(torch.sigmoid(model.protocol_gain_logit).detach()),
        "initial_valid": initial, "final_valid": final_valid, "final_test": final_test,
        "causal_nll_deltas": causal, "first_step_gradients": first_grad,
        "seconds": time.time() - started,
    }
    write_json(arm_dir / "summary.json", summary)
    state = trainable_state(model)
    torch.save({
        "claim": CLAIM, "arm": arm, "source_checkpoint": checkpoint,
        "trainable_state_dict": state,
        "trainable_state_sha256": c10.state_sha256(state),
    }, arm_dir / "checkpoint_trainable.pt")
    del model, frozen, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def self_test(output: Path) -> None:
    valid = torch.ones(2, 8, dtype=torch.bool)
    budgets = torch.tensor([4, 3])
    structured = ownership_mask(valid, budgets, 8, "subheap", 7)
    randomized = ownership_mask(valid, budgets, 8, "random", 7)
    assert structured is not None and randomized is not None
    for owner, budget in ((structured, budgets), (randomized, budgets)):
        for batch_index in range(2):
            active = int(budget[batch_index])
            assert bool(owner[batch_index, :active].any(0).all())
            assert not bool((owner[batch_index, :active].sum(0) > 1).any())
    structured_groups = [
        torch.nonzero(structured[0, slot], as_tuple=False).flatten().tolist()
        for slot in range(4)
    ]
    random_groups = [
        torch.nonzero(randomized[0, slot], as_tuple=False).flatten().tolist()
        for slot in range(4)
    ]
    assert structured_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert random_groups != structured_groups
    write_json(output / "self_test.json", {
        "claim": CLAIM, "passed": True,
        "structured_groups": structured_groups, "random_groups": random_groups,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10801)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(output)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.max_slots < 8 or args.max_slots & (args.max_slots - 1):
        parser.error("--max-slots must be a power of two >= 8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad, vocab = pieces, pieces + 3
    frozen_cpu, _, config, source_hash, parent_hash = d03.load_model(
        Path(args.checkpoint), args, sp, pad, vocab,
    )
    train_rows, valid_rows, test_rows = d07.collect_rows(config, sp, pieces, eos, args)
    row_hash = {
        "train": d07.rows_sha256(train_rows),
        "valid": d07.rows_sha256(valid_rows),
        "test": d07.rows_sha256(test_rows),
    }
    summaries = {}
    for arm in ARMS:
        summaries[arm] = run_arm(
            arm, frozen_cpu, config, source_hash, args.checkpoint,
            train_rows, valid_rows, test_rows, args, sp, pieces, pad, bos, eos,
        )

    initialization_match = len({row["initial_trainable_sha256"] for row in summaries.values()}) == 1
    data_contract = all(
        row["source_sha256_before"] == row["source_sha256_after"] == source_hash
        and row["language_sha256_before"] == row["language_sha256_after"]
        for row in summaries.values()
    )
    structured = summaries["subheap"]
    p1_rows = [structured["final_test"][str(depth)]["native"]["route"] for depth in DEPTHS]
    random_rows = [summaries["random"]["final_test"][str(depth)]["native"]["route"] for depth in DEPTHS]
    p1 = all(
        row["owner_leaf_coverage"] is not None
        and abs(row["owner_leaf_coverage"] - 1.0) <= 1e-6
        and row["route_pair_overlap"] < 0.05
        and row["argmax_coverage"] >= 0.90
        for row in p1_rows
    ) and all(
        row["owner_leaf_coverage"] is not None
        and abs(row["owner_leaf_coverage"] - 1.0) <= 1e-6
        and row["route_pair_overlap"] < 0.05
        for row in random_rows
    )
    quality_margins = {}
    for depth in DEPTHS:
        key = str(depth)
        subheap_nll = structured["final_test"][key]["native"]["nll"]
        quality_margins[key] = {
            "vs_free": summaries["free"]["final_test"][key]["native"]["nll"] - subheap_nll,
            "vs_random": summaries["random"]["final_test"][key]["native"]["nll"] - subheap_nll,
        }
    p2 = sum(
        row["vs_free"] >= 0.02 and row["vs_random"] >= 0.02
        for row in quality_margins.values()
    ) >= 2
    p3 = sum(
        row["shuffle"] >= 0.10 and row["zero"] >= 0.10
        for row in structured["causal_nll_deltas"].values()
    ) >= 2
    nll = [structured["final_valid"][str(depth)]["nll"] for depth in DEPTHS]
    p4 = nll[2] <= nll[1] + 0.05 and nll[1] <= nll[0] + 0.05
    gradients = structured["first_step_gradients"]
    p5 = (
        gradients is not None and min(gradients.values()) > 0
        and min(
            structured["final_test"][str(depth)]["native"]["slot_variance"]
            for depth in DEPTHS
        ) > 1e-4
        and min(
            structured["final_test"][str(depth)]["native"]["between_slot_variance"]
            for depth in DEPTHS
        ) > 1e-8
    )
    gates = {
        "P0_contracts": initialization_match and data_contract,
        "P1_ownership": p1,
        "P2_quality": p2,
        "P3_input_causality": p3,
        "P4_rate_direction": p4,
        "P5_trainability": p5,
    }
    decision = "smoke_supports_multiseed" if all(gates.values()) else "smoke_blocks_multiseed"
    summary = {
        "claim": CLAIM, "decision": decision, "host": socket.gethostname(),
        "config": vars(args), "checkpoint_parent_sha256": parent_hash,
        "source_state_sha256": source_hash,
        "rows": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "row_sha256": row_hash, "initialization_match": initialization_match,
        "quality_nll_margins": quality_margins, "gates": gates,
        "arms": summaries,
        "contracts": {
            "target_enters_compressor": False,
            "target_length_enters_budget": False,
            "source_frozen": True,
            "language_backbone_frozen": True,
            "bounded_protocol_gain": True,
            "transformer_or_self_attention": False,
            "flat_length_route_table": False,
            "matched_arms": list(ARMS),
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete", "claim": CLAIM, "decision": decision,
        "gates": gates, "quality_nll_margins": quality_margins,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
