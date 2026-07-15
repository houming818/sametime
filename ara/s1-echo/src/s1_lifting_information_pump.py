#!/usr/bin/env python3
"""TreeHeap lifting information-pump proof on real Chinese token blocks."""
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


def manifest(block_dir: Path, split: str) -> dict:
    return json.loads((block_dir / f"manifest-{split}.json").read_text(encoding="utf-8"))


def iter_blocks(
    block_dir: Path,
    rows: dict,
    batch_size: int,
    context: int,
    seed: int,
    max_blocks: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    shards = list(rows["shards"])
    rng.shuffle(shards)
    seen = 0
    for shard in shards:
        array = np.load(block_dir / shard["path"], mmap_mode="r")
        if array.shape[1] <= context:
            raise ValueError(f"shard width {array.shape[1]} cannot provide context+target {context + 1}")
        order = np.arange(array.shape[0])
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            if max_blocks:
                index = index[: max(0, max_blocks - seen)]
            if len(index) == 0:
                return
            block = np.asarray(array[index, : context + 1], dtype=np.int64)
            yield torch.from_numpy(block[:, :context]), torch.from_numpy(block[:, context])
            seen += len(index)
            if max_blocks and seen >= max_blocks:
                return


class SharedPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # A bounded predictor keeps repeated lifting numerically stable while
        # preserving the exact algebraic inverse.
        return 2.0 * torch.tanh(0.5 * self.net(state))


class LiftingTreeHeap(nn.Module):
    def __init__(self, vocab: int, context: int, dim: int):
        super().__init__()
        if context < 2 or context & (context - 1):
            raise ValueError("context must be a power of two")
        self.vocab = vocab
        self.context = context
        self.dim = dim
        self.depths = int(math.log2(context))
        self.embedding = nn.Embedding(vocab, dim)
        self.predictor = SharedPredictor(dim)
        self.root_norm = nn.LayerNorm(dim)
        self.next_decoder = nn.Linear(dim, vocab)

    def fold_states(
        self,
        leaves: torch.Tensor,
        break_depth: int = -1,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        node = leaves
        details: List[torch.Tensor] = []
        depth = 0
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            if depth == break_depth:
                right = right.roll(1, dims=0) if right.shape[0] > 1 else torch.zeros_like(right)
            detail = right - self.predictor(left)
            parent = left + 0.5 * detail
            details.append(detail)
            node = parent
            depth += 1
        return node[:, 0], details

    def encode(self, tokens: torch.Tensor, break_depth: int = -1):
        return self.fold_states(self.embedding(tokens), break_depth)

    def unfold_states(
        self,
        root: torch.Tensor,
        details: Sequence[torch.Tensor],
        root_mode: str = "native",
        shuffle_detail_depth: int = -1,
    ) -> torch.Tensor:
        if root_mode == "zero":
            node = torch.zeros_like(root)
        elif root_mode == "shuffle":
            node = root.roll(1, dims=0) if root.shape[0] > 1 else torch.zeros_like(root)
        elif root_mode == "native":
            node = root
        else:
            raise ValueError(root_mode)
        node = node[:, None]
        for depth in range(len(details) - 1, -1, -1):
            detail = details[depth]
            if depth == shuffle_detail_depth:
                detail = detail.roll(1, dims=0) if detail.shape[0] > 1 else torch.zeros_like(detail)
            left = node - 0.5 * detail
            right = detail + self.predictor(left)
            expanded = torch.empty(
                left.shape[0], left.shape[1] * 2, left.shape[2],
                device=left.device, dtype=left.dtype,
            )
            expanded[:, 0::2] = left
            expanded[:, 1::2] = right
            node = expanded
        return node

    def next_logits(self, tokens: torch.Tensor, root_mode: str = "native", break_depth: int = -1):
        root, details = self.encode(tokens, break_depth)
        if root_mode == "zero":
            root = torch.zeros_like(root)
        elif root_mode == "shuffle":
            root = root.roll(1, dims=0) if root.shape[0] > 1 else torch.zeros_like(root)
        elif root_mode != "native":
            raise ValueError(root_mode)
        logits = self.next_decoder(self.root_norm(root))
        return logits, root, details


def finite_gradients(module: nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


@torch.no_grad()
def next_metrics(
    model: LiftingTreeHeap,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    root_mode: str = "native",
    break_depth: int = -1,
) -> dict:
    model.eval()
    total_loss = count = top1 = top5 = 0
    roots: List[torch.Tensor] = []
    for tokens, target in batches:
        tokens, target = tokens.to(device), target.to(device)
        logits, root, _ = model.next_logits(tokens, root_mode=root_mode, break_depth=break_depth)
        total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
        count += target.numel()
        top1 += int(logits.argmax(-1).eq(target).sum())
        top5 += int(logits.topk(5, dim=-1).indices.eq(target[:, None]).any(-1).sum())
        roots.append(root.float().cpu())
    root = torch.cat(roots)
    nll = total_loss / max(1, count)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "top1": top1 / max(1, count),
        "top5": top5 / max(1, count),
        "root_variance": float(root.var(0, unbiased=False).mean()),
    }


def nearest_token_metrics(
    states: torch.Tensor,
    target: torch.Tensor,
    embedding: torch.Tensor,
    chunk: int = 256,
) -> Tuple[int, int]:
    table = F.normalize(embedding.float(), dim=-1)
    flat_state = F.normalize(states.reshape(-1, states.shape[-1]).float(), dim=-1)
    flat_target = target.reshape(-1)
    correct = 0
    rows: List[torch.Tensor] = []
    for start in range(0, flat_state.shape[0], chunk):
        prediction = (flat_state[start : start + chunk] @ table.T).argmax(-1)
        correct += int(prediction.eq(flat_target[start : start + chunk]).sum())
        rows.append(prediction.cpu())
    predicted = torch.cat(rows).reshape_as(target.cpu())
    exact = int(predicted.eq(target.cpu()).all(-1).sum())
    return correct, exact


@torch.no_grad()
def codec_metrics(
    model: LiftingTreeHeap,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    root_mode: str = "native",
    shuffle_detail_depth: int = -1,
) -> dict:
    model.eval()
    squared = maximum = 0.0
    values = tokens_total = token_correct = blocks = block_exact = 0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        leaves = model.embedding(tokens)
        root, details = model.fold_states(leaves)
        decoded = model.unfold_states(
            root, details, root_mode=root_mode,
            shuffle_detail_depth=shuffle_detail_depth,
        )
        difference = decoded - leaves
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        values += difference.numel()
        correct, exact = nearest_token_metrics(decoded, tokens, model.embedding.weight)
        token_correct += correct
        tokens_total += tokens.numel()
        block_exact += exact
        blocks += tokens.shape[0]
    return {
        "state_mse": squared / max(1, values),
        "state_max_abs": maximum,
        "token_top1": token_correct / max(1, tokens_total),
        "block_exact": block_exact / max(1, blocks),
        "blocks": blocks,
    }


@torch.no_grad()
def algebraic_closure(model: LiftingTreeHeap, device: torch.device, seed: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    by_depth = {}
    for depth in range(1, 7):
        leaves = torch.randn(17, 1 << depth, model.dim, generator=generator, device=device)
        root, details = model.fold_states(leaves)
        recovered = model.unfold_states(root, details)
        error = recovered - leaves
        by_depth[str(depth)] = {
            "mse": float(error.square().mean()),
            "max_abs": float(error.abs().max()),
        }
    return {
        "by_depth": by_depth,
        "maximum_abs": max(row["max_abs"] for row in by_depth.values()),
    }


def train_model(
    name: str,
    initial_state: dict,
    train_manifest: dict,
    valid_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    vocab: int,
    device: torch.device,
    output: Path,
) -> dict:
    model = LiftingTreeHeap(vocab, args.context, args.dim).to(device)
    model.load_state_dict(initial_state)
    initial_predictor = {key: value.detach().clone() for key, value in model.predictor.state_dict().items()}
    frozen = name == "frozen_predictor"
    if frozen:
        for parameter in model.predictor.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    trace: List[dict] = []
    gradients_ok = True
    predictor_grad_max = 0.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = steps = 0
        for tokens, target in iter_blocks(
            Path(args.block_dir), train_manifest, args.batch, args.context,
            args.seed + epoch, args.max_train_blocks,
        ):
            tokens, target = tokens.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _, _ = model.next_logits(tokens)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            gradients_ok = gradients_ok and finite_gradients(model)
            if not frozen:
                grad = math.sqrt(sum(float(parameter.grad.float().square().sum()) for parameter in model.predictor.parameters() if parameter.grad is not None))
                predictor_grad_max = max(predictor_grad_max, grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        valid = next_metrics(model, valid_batches, device)
        row = {"model": name, "epoch": epoch, "train_nll": total / max(1, steps), "valid_nll": valid["nll"], "valid_top1": valid["top1"], "elapsed_sec": time.time() - started}
        trace.append(row)
        print(json.dumps(row), flush=True)
    predictor_delta = math.sqrt(sum(float((value - initial_predictor[key].to(value.device)).square().sum()) for key, value in model.predictor.state_dict().items()))
    native = next_metrics(model, valid_batches, device)
    root_zero = next_metrics(model, valid_batches, device, root_mode="zero")
    root_shuffle = next_metrics(model, valid_batches, device, root_mode="shuffle")
    broken = [next_metrics(model, valid_batches, device, break_depth=depth) for depth in range(model.depths)]
    closure = algebraic_closure(model, device, args.seed + 700)
    codec_batches = list(valid_batches[: max(1, math.ceil(args.codec_blocks / args.batch))])
    codec = codec_metrics(model, codec_batches, device)
    codec_root_zero = codec_metrics(model, codec_batches, device, root_mode="zero")
    codec_root_shuffle = codec_metrics(model, codec_batches, device, root_mode="shuffle")
    detail = [codec_metrics(model, codec_batches, device, shuffle_detail_depth=depth) for depth in range(model.depths)]
    checkpoint = output / f"checkpoint_{name}.pt"
    torch.save({"name": name, "state_dict": model.state_dict(), "args": vars(args)}, checkpoint)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "finite_gradients": gradients_ok,
        "predictor_grad_max": predictor_grad_max,
        "predictor_parameter_delta": predictor_delta,
        "seconds": time.time() - started,
        "trace": trace,
        "next_native": native,
        "next_root_zero": root_zero,
        "next_root_zero_nll_increase": root_zero["nll"] - native["nll"],
        "next_root_shuffle": root_shuffle,
        "next_root_shuffle_nll_increase": root_shuffle["nll"] - native["nll"],
        "next_break_depth": broken,
        "next_break_depth_nll_increase": [row["nll"] - native["nll"] for row in broken],
        "algebraic_closure": closure,
        "codec_native": codec,
        "codec_root_zero": codec_root_zero,
        "codec_root_shuffle": codec_root_shuffle,
        "codec_detail_shuffle": detail,
        "codec_detail_shuffle_token_drop": [codec["token_top1"] - row["token_top1"] for row in detail],
        "checkpoint": checkpoint.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    parser.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_lifting_information_pump")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=71511)
    parser.add_argument("--max-train-blocks", type=int, default=100000)
    parser.add_argument("--max-valid-blocks", type=int, default=8192)
    parser.add_argument("--codec-blocks", type=int, default=128)
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
    train_manifest = manifest(block_dir, "train")
    valid_manifest = manifest(block_dir, "valid")
    vocab = int(train_manifest["tokenizer"]["vocab"])
    valid_batches = list(iter_blocks(block_dir, valid_manifest, args.batch, args.context, args.seed + 9000, args.max_valid_blocks))
    prototype = LiftingTreeHeap(vocab, args.context, args.dim)
    initial_state = copy.deepcopy(prototype.state_dict())
    del prototype
    started = time.time()
    results: Dict[str, dict] = {}
    for name in ("learned_predictor", "frozen_predictor"):
        torch.manual_seed(args.seed)
        results[name] = train_model(name, initial_state, train_manifest, valid_batches, args, vocab, device, output)
    learned, frozen = results["learned_predictor"], results["frozen_predictor"]
    codec = learned["codec_native"]
    root_echo_drop = codec["token_top1"] - learned["codec_root_zero"]["token_top1"]
    root_shuffle_echo_drop = codec["token_top1"] - learned["codec_root_shuffle"]["token_top1"]
    maximum_detail_drop = max(learned["codec_detail_shuffle_token_drop"])
    learned_gain = frozen["next_native"]["nll"] - learned["next_native"]["nll"]
    max_break_damage = max(learned["next_break_depth_nll_increase"])
    derived = {
        "learned_nll_gain_over_frozen": learned_gain,
        "root_zero_echo_token_drop": root_echo_drop,
        "root_shuffle_echo_token_drop": root_shuffle_echo_drop,
        "maximum_detail_shuffle_token_drop": maximum_detail_drop,
        "root_shuffle_next_nll_increase": learned["next_root_shuffle_nll_increase"],
        "maximum_break_depth_next_nll_increase": max_break_damage,
    }
    gates = {
        "P1_algebraic_closure": learned["algebraic_closure"]["maximum_abs"] < 1e-5,
        "P2_native_codec": codec["state_mse"] < 1e-10 and codec["token_top1"] >= 0.999 and codec["block_exact"] >= 0.99,
        "P3_root_required_for_echo": max(root_echo_drop, root_shuffle_echo_drop) >= 0.10,
        "P4_details_addressed": maximum_detail_drop >= 0.10,
        "P5_predictor_learns": learned["finite_gradients"] and learned["predictor_grad_max"] > 0 and learned["predictor_parameter_delta"] >= 1e-4,
        "P6_learned_over_frozen": learned_gain >= 0.02,
        "P7_root_and_recursive_address_causal": learned["next_root_shuffle_nll_increase"] >= 0.10 and max_break_damage >= 0.02,
        "no_mask_or_noise": True,
        "echo_not_in_training_loss": True,
    }
    summary = {
        "claim": "S1-LIFT-PUMP-C01",
        "predict": "P-S1-LIFT-PUMP-01",
        "host": socket.gethostname(),
        "device": str(device),
        "elapsed_sec": time.time() - started,
        "config": vars(args),
        "data": {"vocab": vocab, "train_manifest": str(block_dir / "manifest-train.json"), "valid_manifest": str(block_dir / "manifest-valid.json")},
        "models": results,
        "derived": derived,
        "gates": gates,
        "decision": "supported_pilot" if all(gates.values()) else "partial" if all(gates[key] for key in ("P1_algebraic_closure", "P2_native_codec", "P3_root_required_for_echo", "P4_details_addressed")) else "not_supported",
        "boundary": "Lifting algebra and root next-token learning proof; not semantics, world knowledge, optimal compression, or architecture superiority.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for result in results.values() for row in result["trace"]) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# TreeHeap Lifting Information Pump\n\n```json\n" + json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
