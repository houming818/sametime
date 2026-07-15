#!/usr/bin/env python3
"""No-mask S1 echo from TreeHeap internal states only.

The encoder sees complete token blocks.  The decoder never sees leaves and may
read only the ancestor chain of each output address.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


VARIANTS = ("mean_only", "single_learned", "multichannel")


def load_manifest(block_dir: Path, split: str) -> dict:
    return json.loads((block_dir / f"manifest-{split}.json").read_text(encoding="utf-8"))


def iter_blocks(
    block_dir: Path,
    manifest: dict,
    batch_size: int,
    context: int,
    seed: int,
    max_blocks: int,
) -> Iterator[torch.Tensor]:
    rng = np.random.default_rng(seed)
    shards = list(manifest["shards"])
    rng.shuffle(shards)
    seen = 0
    for shard in shards:
        array = np.load(block_dir / shard["path"], mmap_mode="r")
        if array.shape[1] < context:
            raise ValueError(f"shard width {array.shape[1]} is smaller than context {context}")
        order = np.arange(array.shape[0])
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            if max_blocks:
                index = index[: max(0, max_blocks - seen)]
            if len(index) == 0:
                return
            block = np.asarray(array[index, :context], dtype=np.int64)
            yield torch.from_numpy(block)
            seen += len(index)
            if max_blocks and seen >= max_blocks:
                return


class AlgebraicFold(nn.Module):
    """Equal-parameter FOLD variants with four fixed-width output channels."""

    def __init__(self, dim: int, variant: str):
        super().__init__()
        if dim % 4:
            raise ValueError("dim must be divisible by four")
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.dim = dim
        self.variant = variant
        channel = dim // 4
        self.joint = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.project = nn.ModuleList(nn.Linear(dim, channel) for _ in range(4))
        self.norm = nn.LayerNorm(dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        mean = (left + right) * math.sqrt(0.5)
        joint = self.joint(torch.cat((left, right), dim=-1))
        if self.variant == "mean_only":
            sources = (mean, mean, mean, mean)
        elif self.variant == "single_learned":
            sources = (joint, joint, joint, joint)
        else:
            diff = (left - right) * math.sqrt(0.5)
            product = torch.tanh(left) * torch.tanh(right)
            sources = (mean, diff, product, joint)
        return self.norm(torch.cat([layer(value) for layer, value in zip(self.project, sources)], dim=-1))


class InternalEchoTreeHeap(nn.Module):
    def __init__(self, vocab: int, context: int, dim: int, variant: str):
        super().__init__()
        if context < 2 or context & (context - 1):
            raise ValueError("context must be a power of two >= 2")
        self.vocab = vocab
        self.context = context
        self.dim = dim
        self.depths = int(math.log2(context))
        self.embedding = nn.Embedding(vocab, dim)
        self.fold = AlgebraicFold(dim, variant)
        self.position = nn.Embedding(context, dim)
        self.depth = nn.Embedding(self.depths, dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.read_norm = nn.LayerNorm(dim)
        self.readout = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab))

    def encode(self, tokens: torch.Tensor, destroy_addresses: bool = False) -> List[torch.Tensor]:
        node = self.embedding(tokens)
        internal: List[torch.Tensor] = []
        while node.shape[1] > 1:
            left = node[:, 0::2]
            right = node[:, 1::2]
            if destroy_addresses and right.shape[1] > 1:
                right = right.roll(1, dims=1)
            node = self.fold(left, right)
            internal.append(node)
        return internal

    def ancestor_tensor(
        self,
        levels: Sequence[torch.Tensor],
        drop_depth: int = -1,
        root_mode: str = "native",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = levels[0].shape[0]
        positions = torch.arange(self.context, device=levels[0].device)
        rows: List[torch.Tensor] = []
        available: List[bool] = []
        for depth_index, level in enumerate(levels):
            index = positions // (1 << (depth_index + 1))
            state = level[:, index]
            if depth_index == self.depths - 1:
                if root_mode == "zero":
                    state = torch.zeros_like(state)
                elif root_mode == "shuffle":
                    state = state.roll(1, dims=0) if batch > 1 else torch.zeros_like(state)
                elif root_mode != "native":
                    raise ValueError(root_mode)
            rows.append(state)
            available.append(depth_index != drop_depth)
        ancestors = torch.stack(rows, dim=2)
        mask = torch.tensor(available, device=ancestors.device, dtype=torch.bool)
        return ancestors, mask

    def forward(
        self,
        tokens: torch.Tensor,
        drop_depth: int = -1,
        root_mode: str = "native",
        destroy_addresses: bool = False,
        return_audit: bool = False,
    ):
        levels = self.encode(tokens, destroy_addresses=destroy_addresses)
        ancestors, available = self.ancestor_tensor(levels, drop_depth, root_mode)
        positions = torch.arange(self.context, device=tokens.device)
        query = self.position(positions)[None].expand(tokens.shape[0], -1, -1)
        depth = self.depth(torch.arange(self.depths, device=tokens.device))[None, None]
        keys = self.key(ancestors + depth)
        scores = (query[:, :, None] * keys).sum(-1) / math.sqrt(self.dim)
        scores = scores.masked_fill(~available[None, None], -torch.inf)
        weights = F.softmax(scores, dim=-1)
        read = (weights[..., None] * self.value(ancestors)).sum(2)
        state = self.read_norm(read)
        state = self.readout(torch.cat((state, query), dim=-1))
        logits = F.linear(state, self.embedding.weight, self.output_bias)
        if return_audit:
            return logits, weights.detach().mean((0, 1)), levels
        return logits


def batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, int, int, int, int]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    loss = float(F.cross_entropy(flat_logits, flat_target, reduction="sum"))
    top = flat_logits.topk(5, dim=-1).indices
    top1_rows = top[:, 0].eq(flat_target).reshape_as(target)
    top5 = int(top.eq(flat_target[:, None]).any(-1).sum())
    return loss, target.numel(), int(top1_rows.sum()), top5, int(top1_rows.all(-1).sum())


@torch.no_grad()
def evaluate(
    model: InternalEchoTreeHeap,
    batches: Sequence[torch.Tensor],
    device: torch.device,
    drop_depth: int = -1,
    root_mode: str = "native",
    destroy_addresses: bool = False,
    example_limit: int = 4,
) -> dict:
    model.eval()
    total_loss = total_tokens = top1 = top5 = exact = blocks = 0
    depth_mass = torch.zeros(model.depths, device=device)
    examples: List[dict] = []
    for tokens in batches:
        tokens = tokens.to(device)
        logits, weights, _ = model(
            tokens,
            drop_depth=drop_depth,
            root_mode=root_mode,
            destroy_addresses=destroy_addresses,
            return_audit=True,
        )
        loss, count, one, five, sequence = batch_metrics(logits, tokens)
        total_loss += loss
        total_tokens += count
        top1 += one
        top5 += five
        exact += sequence
        blocks += tokens.shape[0]
        depth_mass += weights
        if len(examples) < example_limit:
            predicted = logits.argmax(-1)
            for source, output in zip(tokens[: example_limit - len(examples)], predicted[: example_limit - len(examples)]):
                examples.append({"target_ids": source.cpu().tolist(), "predicted_ids": output.cpu().tolist(), "exact": bool(source.eq(output).all())})
    nll = total_loss / max(1, total_tokens)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "token_top1": top1 / max(1, total_tokens),
        "token_top5": top5 / max(1, total_tokens),
        "block_exact": exact / max(1, blocks),
        "blocks": blocks,
        "depth_read_mass": [float(x) for x in (depth_mass / max(1, len(batches))).cpu()],
        "examples": examples,
    }


def finite_gradients(model: nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def train_variant(
    variant: str,
    initial_state: dict,
    train_manifest: dict,
    valid_batches: Sequence[torch.Tensor],
    vocab: int,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict:
    model = InternalEchoTreeHeap(vocab, args.context, args.dim, variant).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace: List[dict] = []
    gradients_ok = True
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = steps = 0
        for tokens in iter_blocks(Path(args.block_dir), train_manifest, args.batch, args.context, args.seed + epoch, args.max_train_blocks):
            tokens = tokens.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                loss = F.cross_entropy(logits.reshape(-1, vocab), tokens.reshape(-1))
            loss.backward()
            gradients_ok = gradients_ok and finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        valid = evaluate(model, valid_batches, device, example_limit=0)
        row = {"variant": variant, "epoch": epoch, "train_nll": loss_sum / max(1, steps), "valid_nll": valid["nll"], "valid_top1": valid["token_top1"], "valid_exact": valid["block_exact"], "elapsed_sec": time.time() - started}
        trace.append(row)
        print(json.dumps(row), flush=True)
    native = evaluate(model, valid_batches, device)
    drop_levels = [evaluate(model, valid_batches, device, drop_depth=depth, example_limit=0) for depth in range(model.depths)]
    root_zero = evaluate(model, valid_batches, device, root_mode="zero", example_limit=0)
    root_shuffle = evaluate(model, valid_batches, device, root_mode="shuffle", example_limit=0)
    address_destroy = evaluate(model, valid_batches, device, destroy_addresses=True, example_limit=0)
    checkpoint = output / f"checkpoint_{variant}.pt"
    torch.save({"variant": variant, "state_dict": model.state_dict(), "args": vars(args)}, checkpoint)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": gradients_ok,
        "seconds": time.time() - started,
        "trace": trace,
        "native": native,
        "drop_levels": drop_levels,
        "drop_level_nll_increase": [row["nll"] - native["nll"] for row in drop_levels],
        "root_zero": root_zero,
        "root_zero_nll_increase": root_zero["nll"] - native["nll"],
        "root_shuffle": root_shuffle,
        "root_shuffle_nll_increase": root_shuffle["nll"] - native["nll"],
        "address_destroy": address_destroy,
        "address_destroy_nll_increase": address_destroy["nll"] - native["nll"],
        "checkpoint": checkpoint.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    parser.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_multiresolution_internal_echo")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=71501)
    parser.add_argument("--max-train-blocks", type=int, default=50000)
    parser.add_argument("--max-valid-blocks", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    block_dir = Path(args.block_dir)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_manifest = load_manifest(block_dir, "train")
    valid_manifest = load_manifest(block_dir, "valid")
    vocab = int(train_manifest["tokenizer"]["vocab"])
    valid_batches = list(iter_blocks(block_dir, valid_manifest, args.batch, args.context, args.seed + 9000, args.max_valid_blocks))
    prototype = InternalEchoTreeHeap(vocab, args.context, args.dim, "mean_only")
    initial_state = copy.deepcopy(prototype.state_dict())
    del prototype
    results: Dict[str, dict] = {}
    started = time.time()
    for variant in VARIANTS:
        torch.manual_seed(args.seed)
        results[variant] = train_variant(variant, initial_state, train_manifest, valid_batches, vocab, args, device, output)
    multi = results["multichannel"]
    mean = results["mean_only"]
    single = results["single_learned"]
    mean_gain = mean["native"]["nll"] - multi["native"]["nll"]
    single_gain = single["native"]["nll"] - multi["native"]["nll"]
    closest_parent_damage = multi["drop_level_nll_increase"][0]
    maximum_level_damage = max(
        multi["drop_level_nll_increase"]
        + [multi["root_zero_nll_increase"], multi["root_shuffle_nll_increase"]]
    )
    derived = {
        "multichannel_nll_gain_over_mean": mean_gain,
        "multichannel_nll_gain_over_single": single_gain,
        "closest_parent_nll_damage": closest_parent_damage,
        "maximum_level_nll_damage": maximum_level_damage,
        "root_zero_nll_damage": multi["root_zero_nll_increase"],
        "root_shuffle_nll_damage": multi["root_shuffle_nll_increase"],
        "address_destroy_nll_damage": multi["address_destroy_nll_increase"],
    }
    gates = {
        "P1_multichannel_over_mean": mean_gain >= 0.05,
        "P2_multichannel_over_single": single_gain >= 0.02,
        "P3_closest_parent_causal": closest_parent_damage >= 0.10,
        "P4_some_level_causal": maximum_level_damage >= 0.02,
        "P5_address_causal": multi["address_destroy_nll_increase"] >= 0.10,
        "P6_finite_nonempty": all(row["finite_gradients"] and row["native"]["blocks"] > 0 and math.isfinite(row["native"]["nll"]) for row in results.values()),
        "equal_parameter_count": len({row["parameters"] for row in results.values()}) == 1,
        "P7_no_decoder_leaf_access": True,
        "P8_no_training_mask_or_noise": True,
    }
    summary = {
        "claim": "S1-INTERNAL-ECHO-C01",
        "predict": "P-S1-INTERNAL-ECHO-01",
        "host": socket.gethostname(),
        "device": str(device),
        "elapsed_sec": time.time() - started,
        "config": vars(args),
        "data": {"train_manifest": str(block_dir / "manifest-train.json"), "valid_manifest": str(block_dir / "manifest-valid.json"), "vocab": vocab},
        "models": results,
        "derived": derived,
        "gates": gates,
        "decision": "supported_pilot" if all(gates.values()) else "partial" if gates["P3_closest_parent_causal"] and gates["P5_address_causal"] else "not_supported",
        "boundary": "No-mask, internal-state-only surface codec proof; not semantics, reasoning, world knowledge, or Transformer superiority.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for result in results.values() for row in result["trace"]) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# S1 Multiresolution Internal-State Echo\n\n```json\n" + json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
