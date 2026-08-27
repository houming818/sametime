#!/usr/bin/env python3
"""Train a finite-rate TreeHeap protocol under depth-derived capacity pressure."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_multilevel_read_ablation_c12 as c12  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_length_pressure as d06  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_wmt_treeheap_seq2seq as wmt_metrics  # noqa: E402


CLAIM = "S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07"
CLAIM_R1 = "S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R1"
CLAIM_R2 = "S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R2"
DEPTHS = (5, 6, 7)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    c10.update_stream_digest(digest, rows)
    return digest.hexdigest()


class ProtocolReadKernel(nn.Module):
    """One recursive query kernel shared by every slot and source depth."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, query, local, depth_state):
        return torch.tanh(self.net(torch.cat((query, local, depth_state), dim=-1)))


class DepthPressureCompressor(nn.Module):
    """Read a source TreeHeap into a depth-limited number of protocol slots."""

    def __init__(self, dim: int, max_slots: int, source_depths: int):
        super().__init__()
        self.dim = dim
        self.max_slots = max_slots
        self.slot_query = nn.Embedding(max_slots, dim)
        self.depth_embedding = nn.Embedding(source_depths, dim)
        self.read_kernel = ProtocolReadKernel(dim)
        self.branch = nn.Linear(dim, dim, bias=False)
        self.query_norm = nn.LayerNorm(dim)
        self.slot_out = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, tree, masks, max_depth: int, budgets):
        batch = tree[0].shape[0]
        slot_index = torch.arange(self.max_slots, device=tree[0].device)
        query = self.slot_query(slot_index)[None].expand(batch, -1, -1)
        query = query + self.depth_embedding.weight[max_depth][None, None]
        frontier = masks[0][:, None].expand(-1, self.max_slots, -1).to(query.dtype)
        entropy = []
        local = None
        last_depth = min(max_depth, len(tree) - 1)
        for depth in range(last_depth + 1):
            nodes = tree[depth]
            valid = masks[depth]
            frontier = frontier * valid[:, None].to(frontier.dtype)
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
            scores = scores.masked_fill(~child_valid[:, None], -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * child_valid[:, None].to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, :, None] * probability).flatten(2)
        slots = self.slot_out(torch.cat((query, local), dim=-1))
        slot_mask = slot_index[None] < budgets[:, None]
        slots = slots * slot_mask[:, :, None]
        return slots, slot_mask, torch.stack(entropy)


def fold_protocol(slots, slot_mask):
    """Build bounded root-to-leaf TreeHeap levels from protocol leaf slots."""
    levels = [slots]
    masks = [slot_mask]
    node, valid = slots, slot_mask
    scale = math.sqrt(0.5)
    while node.shape[1] > 1:
        left, right = node[:, 0::2], node[:, 1::2]
        left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
        both = left_valid & right_valid
        parent = torch.where(
            both[:, :, None],
            (left + right) * scale,
            torch.where(left_valid[:, :, None], left, right),
        )
        valid = left_valid | right_valid
        parent = parent * valid[:, :, None]
        levels.append(parent)
        masks.append(valid)
        node = parent
    return list(reversed(levels)), list(reversed(masks))


class PressureProtocolModel(nn.Module):
    def __init__(
        self, frozen_source, dim: int, hidden: int, max_slots: int,
        freeze_language_backbone: bool = False, bounded_protocol_gain: bool = False,
    ):
        super().__init__()
        self.frozen_source = frozen_source
        for parameter in self.frozen_source.parameters():
            parameter.requires_grad_(False)
        self.compressor = DepthPressureCompressor(
            dim, max_slots, frozen_source.decoder.depth_embedding.num_embeddings,
        )
        inherited = copy.deepcopy(frozen_source.decoder)
        self.reconstructor = c11.MultiLevelConvolutionDecoder(
            inherited, dim, hidden, int(math.log2(max_slots)) + 1, use_up=True,
        )
        for parameter in self.reconstructor.parameters():
            parameter.requires_grad_(True)
        if freeze_language_backbone:
            for module in (
                self.reconstructor.embedding,
                self.reconstructor.query,
                self.reconstructor.cell,
                self.reconstructor.output,
                self.reconstructor.branch,
                self.reconstructor.depth_embedding,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.max_slots = max_slots
        self.freeze_language_backbone = freeze_language_backbone
        self.bounded_protocol_gain = bounded_protocol_gain
        if bounded_protocol_gain:
            self.protocol_gain_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.register_buffer("protocol_gain_logit", torch.tensor(20.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_source.eval()
        return self

    def protocol(self, source, lengths, depth: int, intervention: str = "native"):
        with torch.no_grad():
            levels, masks = d03.condition_states(self.frozen_source, source, lengths, "native")
            source_tree = self.frozen_source.decoder.convolve(levels, masks)
        budgets = d06.depth_budgets(lengths, depth, 2, self.max_slots)
        slots, slot_mask, entropy = self.compressor(source_tree, masks, depth, budgets)
        slots = slots * torch.sigmoid(self.protocol_gain_logit)
        if intervention == "shuffle":
            if slots.shape[0] > 1:
                slots = slots.roll(1, dims=0)
                slot_mask = slot_mask.roll(1, dims=0)
        elif intervention == "zero":
            slots = torch.zeros_like(slots)
        elif intervention != "native":
            raise ValueError(intervention)
        tree, tree_masks = fold_protocol(slots, slot_mask)
        return tree, tree_masks, budgets, slots, entropy

    def teacher(self, source, lengths, target, bos: int, depth: int, intervention="native"):
        tree, masks, budgets, slots, entropy = self.protocol(
            source, lengths, depth, intervention,
        )
        logits, route = self.reconstructor.teacher(tree, masks, target, bos)
        return logits, route, budgets, slots, entropy

    @torch.no_grad()
    def greedy(self, source, lengths, bos: int, eos: int, max_len: int, depth: int):
        tree, masks, budgets, slots, entropy = self.protocol(source, lengths, depth)
        output, route = self.reconstructor.greedy(tree, masks, bos, eos, max_len)
        return output, route, budgets, slots, entropy


def finite_trainable_gradients(model) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters() if parameter.requires_grad
    )


def group_grad_norm(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def language_backbone_state(model):
    prefixes = (
        "reconstructor.embedding.", "reconstructor.query.",
        "reconstructor.cell.", "reconstructor.output.",
        "reconstructor.branch.", "reconstructor.depth_embedding.",
    )
    return {
        name: value for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


@torch.no_grad()
def evaluate(model, rows, pad, bos, device, batch_size, depth, intervention="native"):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    slot_sum = 0.0
    slot_square_sum = 0.0
    slot_values = 0
    budgets = []
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
        tokens += int(target.ne(pad).sum())
        active_mask = torch.arange(slots.shape[1], device=slots.device)[None] < local_budgets[:, None]
        active = slots[active_mask]
        slot_sum += float(active.sum())
        slot_square_sum += float(active.square().sum())
        slot_values += active.numel()
        budgets.extend(int(value) for value in local_budgets.cpu())
    nll = loss_sum / max(1, tokens)
    mean = slot_sum / max(1, slot_values)
    variance = slot_square_sum / max(1, slot_values) - mean * mean
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": tokens,
        "mean_budget": sum(budgets) / max(1, len(budgets)),
        "slot_variance": max(0.0, variance),
    }


@torch.no_grad()
def generation_metrics(model, rows, args, sp, pad, bos, eos, pieces, depth, limit=12):
    model.eval()
    hypotheses, references, examples = [], [], []
    adjacent_equal = 0
    adjacent_total = 0
    for row in rows[:limit]:
        source, lengths, target = c10.collate_rows([row], pad, args.device)
        generated, route, budgets, _, _ = model.greedy(
            source, lengths, bos, eos, args.max_generation, depth,
        )
        hypothesis = c10.wmt.clean(generated[0].tolist(), eos, pieces)
        reference = c10.wmt.clean(target[0].tolist(), eos, pieces)
        hypotheses.append(hypothesis)
        references.append(reference)
        adjacent_equal += sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
        adjacent_total += max(0, len(hypothesis) - 1)
        examples.append({
            "direction": row[2],
            "source": row[3][1] if row[2] == "en2zh" else row[3][0],
            "reference": sp.decode(reference),
            "generation": sp.decode(hypothesis),
            "budget": int(budgets[0]),
            "route": [float(value) for value in route.detach().cpu()],
        })
    return {
        "token_bleu4": wmt_metrics.bleu4(hypotheses, references),
        "adjacent_repetition_rate": adjacent_equal / max(1, adjacent_total),
        "nonempty_rate": sum(bool(row) for row in hypotheses) / max(1, len(hypotheses)),
        "examples": examples,
    }


def collect_rows(config, sp, pieces, eos, args):
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    config.task_train_rows = max(args.train_rows * 4, args.train_rows + 1024)
    config.task_eval_rows = max(args.eval_rows * 4, args.eval_rows + 256)
    config.max_wmt_scan_lines = 3_000_000
    train, valid, test = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    maximum = args.max_slots + 2
    train = [row for row in train if len(row[0]) <= maximum][:args.train_rows]
    valid = [row for row in valid if len(row[0]) <= maximum][:args.eval_rows]
    test = [row for row in test if len(row[0]) <= maximum][:args.eval_rows]
    if min(len(train), len(valid), len(test)) < min(16, args.eval_rows):
        raise RuntimeError(f"insufficient filtered rows: {len(train)}, {len(valid)}, {len(test)}")
    return train, valid, test


def self_test(output: Path) -> None:
    torch.manual_seed(7)
    slots = torch.randn(3, 8, 12)
    budgets = torch.tensor([2, 5, 8])
    mask = torch.arange(8)[None] < budgets[:, None]
    levels, masks = fold_protocol(slots * mask[:, :, None], mask)
    assert [row.shape[1] for row in levels] == [1, 2, 4, 8]
    assert [row.sum(1).tolist() for row in masks] == [[1, 1, 1], [1, 2, 2], [1, 3, 4], [2, 5, 8]]
    write_json(output / "self_test.json", {
        "claim": CLAIM, "passed": True,
        "level_widths": [row.shape[1] for row in levels],
        "leaf_counts": masks[-1].sum(1).tolist(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--seed", type=int, default=10701)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze-language-backbone", action="store_true")
    parser.add_argument("--bounded-protocol-gain", action="store_true")
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
    if args.mode == "smoke":
        args.steps = args.steps or 120
        args.batch_size = args.batch_size or 8
        args.train_rows = args.train_rows or 2048
        args.eval_rows = args.eval_rows or 128
        args.log_every = args.log_every or 30
    else:
        args.steps = args.steps or 5000
        args.batch_size = args.batch_size or 12
        args.train_rows = args.train_rows or 50000
        args.eval_rows = args.eval_rows or 1000
        args.log_every = args.log_every or 250

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad, vocab = pieces, pieces + 3
    frozen, saved, config, source_hash, parent_hash = d03.load_model(
        Path(args.checkpoint), args, sp, pad, vocab,
    )
    frozen = frozen.to(args.device)
    model = PressureProtocolModel(
        frozen, config.dim, config.hidden, args.max_slots,
        freeze_language_backbone=args.freeze_language_backbone,
        bounded_protocol_gain=args.bounded_protocol_gain,
    ).to(args.device)
    claim = (
        CLAIM_R2 if args.bounded_protocol_gain
        else CLAIM_R1 if args.freeze_language_backbone
        else CLAIM
    )
    protocol_gain_initial = float(torch.sigmoid(model.protocol_gain_logit).detach())
    language_hash_before = c10.state_sha256(language_backbone_state(model))
    train_rows, valid_rows, test_rows = collect_rows(config, sp, pieces, eos, args)
    row_hash = {
        "train": rows_sha256(train_rows),
        "valid": rows_sha256(valid_rows),
        "test": rows_sha256(test_rows),
    }
    schedule = c10.rows_schedule(train_rows, args.steps, args.batch_size, args.seed + 1)
    initial = {
        str(depth): evaluate(model, valid_rows, pad, bos, args.device, args.batch_size, depth)
        for depth in DEPTHS
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    depth_rng = random.Random(args.seed + 2)
    depth_schedule = []
    while len(depth_schedule) < args.steps:
        block = list(DEPTHS)
        depth_rng.shuffle(block)
        depth_schedule.extend(block)
    depth_schedule = depth_schedule[:args.steps]
    trace_path = output / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    started = time.time()
    grad_audit = None
    for step, batch in enumerate(schedule, 1):
        model.train()
        depth = depth_schedule[step - 1]
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        logits, route, budgets, slots, entropy = model.teacher(
            source, lengths, target, bos, depth,
        )
        tokens = int(target.ne(pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not finite_trainable_gradients(model):
            raise RuntimeError(f"non-finite gradient at step {step}")
        current_grad = {
            "compressor_read": group_grad_norm(model.compressor.read_kernel),
            "compressor_slot": group_grad_norm(model.compressor.slot_out),
            "reconstructor_read": group_grad_norm(model.reconstructor.read_kernel),
            "reconstructor_output": group_grad_norm(model.reconstructor.output),
            "protocol_gain": (
                float(model.protocol_gain_logit.grad.detach().abs())
                if model.protocol_gain_logit.grad is not None else 0.0
            ),
        }
        if grad_audit is None:
            grad_audit = current_grad
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            validation = {
                str(value): evaluate(
                    model, valid_rows, pad, bos, args.device, args.batch_size, value,
                ) for value in DEPTHS
            }
            row = {
                "step": step, "depth": depth, "train_nll": float(loss.detach()),
                "valid": validation, "grad_norm": grad_norm,
                "grad_groups": current_grad,
                "mean_budget": float(budgets.float().mean()),
                "slot_variance": float(slots.detach().var()),
                "source_entropy": [float(value) for value in entropy.detach().cpu()],
                "route": [float(value) for value in route.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
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
            "generation": generation_metrics(
                model, test_rows, args, sp, pad, bos, eos, pieces, depth,
            ),
        } for depth in DEPTHS
    }
    source_hash_after = c10.state_sha256(model.frozen_source.state_dict())
    language_hash_after = c10.state_sha256(language_backbone_state(model))
    improvements = {
        str(depth): initial[str(depth)]["nll"] - final_valid[str(depth)]["nll"]
        for depth in DEPTHS
    }
    causal_deltas = {
        str(depth): {
            "shuffle": final_test[str(depth)]["shuffle"]["nll"] - final_test[str(depth)]["native"]["nll"],
            "zero": final_test[str(depth)]["zero"]["nll"] - final_test[str(depth)]["native"]["nll"],
        } for depth in DEPTHS
    }
    p0 = (
        source_hash == source_hash_after
        and (
            not args.freeze_language_backbone
            or language_hash_before == language_hash_after
        )
    )
    p1 = sum(value >= 0.10 for value in improvements.values()) >= 2
    p2 = (
        sum(value["shuffle"] >= 0.10 for value in causal_deltas.values()) >= 2
        and sum(value["zero"] >= 0.10 for value in causal_deltas.values()) >= 2
    )
    nll = [final_valid[str(depth)]["nll"] for depth in DEPTHS]
    p3 = nll[2] <= nll[1] + 0.05 and nll[1] <= nll[0] + 0.05
    p4 = (
        min(final_test[str(depth)]["native"]["slot_variance"] for depth in DEPTHS) > 1e-4
        and grad_audit is not None
        and grad_audit["compressor_read"] > 0
        and grad_audit["reconstructor_read"] > 0
        and (
            not args.bounded_protocol_gain
            or grad_audit["protocol_gain"] > 0
        )
    )
    gates = {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4}
    decision = "smoke_supports_formal" if all(gates.values()) else "smoke_blocks_formal"
    protocol_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("frozen_source.")
    }
    protocol_hash = c10.state_sha256(protocol_state)
    summary = {
        "claim": claim, "mode": args.mode, "decision": decision,
        "host": socket.gethostname(), "config": vars(args),
        "checkpoint": args.checkpoint,
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": source_hash_after,
        "language_backbone_sha256_before": language_hash_before,
        "language_backbone_sha256_after": language_hash_after,
        "parent_state_sha256": parent_hash,
        "protocol_state_sha256": protocol_hash,
        "protocol_gain_initial": protocol_gain_initial,
        "protocol_gain_final": float(torch.sigmoid(model.protocol_gain_logit).detach()),
        "rows": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "row_sha256": row_hash,
        "initial_valid": initial, "final_valid": final_valid, "final_test": final_test,
        "nll_improvements": improvements, "causal_nll_deltas": causal_deltas,
        "first_step_gradients": grad_audit, "gates": gates,
        "contracts": {
            "target_enters_compressor": False,
            "target_length_enters_budget": False,
            "source_frozen": True,
            "language_backbone_frozen": args.freeze_language_backbone,
            "bounded_protocol_gain": args.bounded_protocol_gain,
            "transformer_or_self_attention": False,
            "flat_length_route_table": False,
            "protocol": "depth_limited_slots_to_recursive_fold_to_recursive_read",
        },
        "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    torch.save({
        "claim": claim, "protocol_state_dict": protocol_state,
        "protocol_state_sha256": protocol_hash,
        "source_checkpoint": args.checkpoint,
        "source_state_sha256": source_hash,
        "config": vars(args), "row_sha256": row_hash,
    }, output / "checkpoint_protocol.pt")
    print(json.dumps({
        "event": "complete", "decision": decision,
        "gates": gates, "improvements": improvements,
        "causal_nll_deltas": causal_deltas,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
