#!/usr/bin/env python3
"""Real-corpus smoke for path-aware QK routing in TreeHeap."""
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
from types import SimpleNamespace
from typing import Dict, Iterable, Sequence

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SRC = Path.cwd() / "ara" / "s3-generation" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_SRC))
import s2_adaptive_lifting_wmt as adaptive  # noqa: E402
import s2_lifting_pump_wmt as prior  # noqa: E402
import s3_wmt_treeheap_seq2seq as base  # noqa: E402


ARMS = ("uniform", "qk_no_path", "qk_fixed_path", "qk_learned_path")
INTERVENTIONS = (
    "uniform",
    "query_shuffle",
    "key_shuffle",
    "path_zero",
    "path_shuffle",
    "left_right_swap",
    "source_shuffle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def selected_rows_hash(rows: Sequence[tuple[list[int], list[int]]]) -> str:
    digest = hashlib.sha256()
    for source, target in rows:
        digest.update(json.dumps([source, target], separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def fixed_path_table(heap_width: int, dim: int) -> torch.Tensor:
    """Deterministic unique depth and heap-address code with bounded norm."""
    total = 2 * heap_width - 1
    position = torch.arange(total, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dim))
    )
    table = torch.zeros(total, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    if dim > 1:
        odd_width = table[:, 1::2].shape[1]
        table[:, 1::2] = torch.cos(position * frequency[:odd_width])
    return table / math.sqrt(max(1, dim))


class TreeEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int, heap_width: int, pad: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.fold = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(dim)
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad = pad

    def forward(self, source: torch.Tensor, length: torch.Tensor):
        padded = torch.full(
            (source.shape[0], self.heap_width),
            self.pad,
            dtype=source.dtype,
            device=source.device,
        )
        padded[:, : source.shape[1]] = source
        mask = torch.arange(self.heap_width, device=source.device)[None] < length[:, None]
        node = self.embedding(padded) * mask[:, :, None]
        levels = [(node, mask)]
        for _ in range(self.depths):
            left, right = node[:, 0::2], node[:, 1::2]
            left_mask, right_mask = mask[:, 0::2], mask[:, 1::2]
            paired = left_mask & right_mask
            merged = self.norm(self.fold(torch.cat((left, right), dim=-1)))
            node = torch.where(
                paired[:, :, None],
                merged,
                torch.where(left_mask[:, :, None], left, right),
            )
            mask = left_mask | right_mask
            node = node * mask[:, :, None]
            levels.append((node, mask))
        return list(reversed(levels))


class TreeQKDecoder(nn.Module):
    def __init__(self, arm: str, vocab: int, dim: int, hidden: int, heap_width: int):
        super().__init__()
        self.target_embedding = nn.Embedding(vocab, dim)
        self.query_content = nn.Linear(hidden, dim, bias=False)
        self.query_parent = nn.Linear(dim, dim, bias=False)
        self.key_content = nn.Linear(dim, dim, bias=False)
        self.query_path = nn.Linear(dim, dim, bias=False)
        self.key_path = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.cell = nn.GRUCell(2 * dim, hidden)
        self.output = nn.Linear(hidden + dim, vocab)
        self.arm = arm
        self.dim = dim
        self.hidden = hidden
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.register_buffer("fixed_paths", fixed_path_table(heap_width, dim))
        if arm == "qk_learned_path":
            self.learned_paths = nn.Embedding(2 * heap_width - 1, dim)
            nn.init.zeros_(self.learned_paths.weight)
        else:
            self.learned_paths = None

    def path_codes(self, depth: int, intervention: str, device: torch.device):
        count = 1 << depth
        start = count - 1
        indexes = torch.arange(start, start + count, device=device)
        if self.arm in ("uniform", "qk_no_path"):
            code = torch.zeros(count, self.dim, device=device)
        elif self.arm == "qk_fixed_path":
            code = self.fixed_paths[indexes]
        elif self.arm == "qk_learned_path":
            code = self.learned_paths(indexes)
        else:
            raise ValueError(self.arm)
        if intervention == "path_zero":
            code = torch.zeros_like(code)
        elif intervention == "path_shuffle" and count > 1:
            code = code.roll(1, dims=0)
        return code

    def route(self, state: torch.Tensor, levels, intervention: str):
        batch = state.shape[0]
        base_query = self.query_content(state)
        if intervention == "query_shuffle" and batch > 1:
            base_query = base_query.roll(1, dims=0)
        parent_probability = state.new_ones((batch, 1))
        depth_entropy = []
        for depth in range(self.depths):
            parent_state, _ = levels[depth]
            child_state, child_mask_flat = levels[depth + 1]
            parent_count = parent_state.shape[1]
            child_state = child_state.reshape(batch, parent_count, 2, self.dim)
            child_mask = child_mask_flat.reshape(batch, parent_count, 2)
            parent_path = self.path_codes(depth, intervention, state.device)
            child_path = self.path_codes(depth + 1, intervention, state.device)
            child_path = child_path.reshape(parent_count, 2, self.dim)
            query = (
                base_query[:, None, :]
                + self.query_parent(parent_state)
                + self.query_path(parent_path)[None, :, :]
            )
            key_state = child_state
            if intervention == "key_shuffle" and batch > 1:
                key_state = key_state.roll(1, dims=0)
            key = (
                self.key_content(key_state)
                + self.key_path(child_path)[None, :, :, :]
            )
            scores = (query[:, :, None, :] * key).sum(-1) / math.sqrt(self.dim)
            if self.arm == "uniform" or intervention == "uniform":
                scores = torch.zeros_like(scores)
            if intervention == "left_right_swap":
                scores = scores.flip(-1)
            local = F.softmax(scores.masked_fill(~child_mask, -1e9), dim=-1)
            child_probability = parent_probability[:, :, None] * local
            child_probability = child_probability * child_mask
            parent_probability = child_probability.reshape(batch, -1)
            parent_probability = parent_probability / parent_probability.sum(
                -1, keepdim=True
            ).clamp_min(1e-12)
            depth_entropy.append(
                -(parent_probability.clamp_min(1e-12).log() * parent_probability)
                .sum(-1)
                .mean()
            )
        leaf_state, _ = levels[-1]
        context = (
            parent_probability[:, :, None] * self.value(leaf_state)
        ).sum(1)
        return context, parent_probability, torch.stack(depth_entropy)

    def step(self, previous, state, levels, intervention):
        context, leaf_probability, entropy = self.route(state, levels, intervention)
        state = self.cell(
            torch.cat((self.target_embedding(previous), context), dim=-1),
            state,
        )
        logits = self.output(torch.cat((state, context), dim=-1))
        return logits, state, leaf_probability, entropy

    def teacher(
        self,
        levels,
        target: torch.Tensor,
        bos: int,
        intervention: str,
        capture_leaf: bool = False,
    ):
        state = levels[0][0].new_zeros((target.shape[0], self.hidden))
        previous = torch.full(
            (target.shape[0],), bos, dtype=torch.long, device=target.device
        )
        outputs = []
        entropies = []
        leaf_rows = []
        for step in range(target.shape[1]):
            logits, state, leaf_probability, entropy = self.step(
                previous, state, levels, intervention
            )
            outputs.append(logits)
            entropies.append(entropy)
            if capture_leaf:
                leaf_rows.append(leaf_probability)
            previous = target[:, step]
        route_entropy = torch.stack(entropies).mean(0)
        leaf_trace = torch.stack(leaf_rows, dim=1) if capture_leaf else None
        return torch.stack(outputs, dim=1), route_entropy, leaf_trace

    def greedy(self, levels, bos: int, eos: int, max_len: int, intervention: str):
        batch = levels[0][0].shape[0]
        state = levels[0][0].new_zeros((batch, self.hidden))
        previous = torch.full(
            (batch,), bos, dtype=torch.long, device=state.device
        )
        done = torch.zeros(batch, dtype=torch.bool, device=state.device)
        outputs = []
        entropies = []
        for _ in range(max_len):
            logits, state, _, entropy = self.step(
                previous, state, levels, intervention
            )
            previous = logits.argmax(-1)
            outputs.append(previous)
            entropies.append(entropy)
            done |= previous.eq(eos)
            if bool(done.all()):
                break
        tokens = torch.stack(outputs, dim=1) if outputs else previous[:, None]
        route_entropy = torch.stack(entropies).mean(0)
        return tokens, route_entropy


class TreeQKModel(nn.Module):
    def __init__(
        self,
        arm: str,
        vocab: int,
        dim: int,
        hidden: int,
        heap_width: int,
        pad: int,
    ):
        super().__init__()
        self.encoder = TreeEncoder(vocab, dim, heap_width, pad)
        self.decoder = TreeQKDecoder(arm, vocab, dim, hidden, heap_width)
        self.arm = arm

    def levels(self, source, length, intervention):
        levels = self.encoder(source, length)
        if intervention == "source_shuffle" and source.shape[0] > 1:
            levels = [
                (state.roll(1, dims=0), mask.roll(1, dims=0))
                for state, mask in levels
            ]
        return levels

    def teacher(
        self,
        source,
        length,
        target,
        bos,
        intervention="native",
        route_mode="native",
        pair_break_depth=-1,
        capture_leaf=False,
    ):
        del route_mode, pair_break_depth
        levels = self.levels(source, length, intervention)
        logits, entropy, leaf = self.decoder.teacher(
            levels, target, bos, intervention, capture_leaf
        )
        if capture_leaf:
            return logits, entropy, leaf
        return logits, entropy

    def greedy(
        self,
        source,
        length,
        bos,
        eos,
        max_len,
        intervention="native",
        route_mode="native",
    ):
        del route_mode
        levels = self.levels(source, length, intervention)
        return self.decoder.greedy(levels, bos, eos, max_len, intervention)


def parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def make_loaders(splits, args, pad: int):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 991)
    return [
        DataLoader(
            base.ParallelDataset(split),
            batch_size=args.batch_size,
            shuffle=index == 0,
            generator=generator if index == 0 else None,
            num_workers=0,
            collate_fn=base.collate(pad),
            pin_memory=args.device.startswith("cuda"),
        )
        for index, split in enumerate(splits)
    ]


@torch.no_grad()
def initial_batch_nll(model, batch, args, pad, bos):
    source, length, target, _ = batch
    source = source.to(args.device)
    length = length.to(args.device)
    target = target.to(args.device)
    logits, _ = model.teacher(source, length, target, bos)
    return float(base.ce(logits, target, pad))


def js_divergence(first: torch.Tensor, second: torch.Tensor):
    first = first.clamp_min(1e-12)
    second = second.clamp_min(1e-12)
    middle = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - middle.log())).sum(-1)
        + (second * (second.log() - middle.log())).sum(-1)
    )


@torch.no_grad()
def route_audit(model, loader, args, bos, pad):
    model.eval()
    totals = {
        name: {"js_sum": 0.0, "count": 0}
        for name in ("query_shuffle", "key_shuffle", "path_zero", "path_shuffle")
    }
    entropy_sum = max_sum = effective_sum = 0.0
    native_count = 0
    for batch_index, (source, length, target, _) in enumerate(loader):
        if batch_index >= args.audit_batches:
            break
        source = source.to(args.device)
        length = length.to(args.device)
        target = target.to(args.device)
        _, _, native = model.teacher(
            source, length, target, bos, capture_leaf=True
        )
        valid = target.ne(pad)
        entropy = -(native.clamp_min(1e-12).log() * native).sum(-1)
        entropy_sum += float(entropy[valid].sum())
        max_sum += float(native.max(-1).values[valid].sum())
        effective_sum += float(entropy[valid].exp().sum())
        native_count += int(valid.sum())
        for name in totals:
            _, _, changed = model.teacher(
                source,
                length,
                target,
                bos,
                intervention=name,
                capture_leaf=True,
            )
            divergence = js_divergence(native, changed)
            totals[name]["js_sum"] += float(divergence[valid].sum())
            totals[name]["count"] += int(valid.sum())
    return {
        "native_leaf_entropy": entropy_sum / max(1, native_count),
        "native_effective_leaves": effective_sum / max(1, native_count),
        "native_max_leaf_mass": max_sum / max(1, native_count),
        "route_jsd": {
            name: values["js_sum"] / max(1, values["count"])
            for name, values in totals.items()
        },
        "tokens": native_count,
    }


@torch.no_grad()
def repetition_audit(model, rows, args, pad, bos, eos):
    loader = DataLoader(
        base.ParallelDataset(rows[: args.generation_samples]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=base.collate(pad),
    )
    repeated = adjacent = tokens = sequences = 0
    model.eval()
    for source, length, target, _ in loader:
        source = source.to(args.device)
        length = length.to(args.device)
        predicted, _ = model.greedy(
            source, length, bos, eos, target.shape[1]
        )
        for row in predicted.cpu().tolist():
            cleaned = base.clean(row, eos, pad)
            if not cleaned:
                continue
            tokens += len(cleaned)
            repeated += len(cleaned) - len(set(cleaned))
            adjacent += sum(a == b for a, b in zip(cleaned, cleaned[1:]))
            sequences += 1
    return {
        "repeated_token_fraction": repeated / max(1, tokens),
        "adjacent_repeat_fraction": adjacent / max(1, tokens - sequences),
        "nonempty_sequences": sequences,
    }


def train_arm(arm, splits, args, vocab, pad, bos, eos, processor):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TreeQKModel(
        arm, vocab, args.dim, args.hidden, args.heap_width, pad
    ).to(args.device)
    shared_hash = state_hash(
        (name, value)
        for name, value in model.state_dict().items()
        if "learned_paths" not in name and "fixed_paths" not in name
    )
    train_loader, valid_loader, test_loader = make_loaders(splits, args, pad)
    first_batch = next(iter(test_loader))
    step_zero_nll = initial_batch_nll(model, first_batch, args, pad, bos)
    cfg = SimpleNamespace(device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_nll = float("inf")
    best = None
    trace = []
    finite = True
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = steps = 0
        for source, length, target, _ in train_loader:
            source = source.to(args.device, non_blocking=True)
            length = length.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits, _ = model.teacher(source, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite = finite and prior.finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        valid = prior.evaluate(model, valid_loader, cfg, pad, bos, eos, processor)
        row = {
            "arm": arm,
            "epoch": epoch,
            "train_nll": total / max(1, steps),
            "valid_nll": valid["nll"],
            "route_depth_entropy": valid.get("route_depth_mass"),
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
    if best is None:
        raise RuntimeError(f"no finite checkpoint for {arm}")
    model.load_state_dict(best)
    test = prior.evaluate(model, test_loader, cfg, pad, bos, eos, processor)
    generation_loader = DataLoader(
        base.ParallelDataset(splits[2][: args.generation_samples]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=base.collate(pad),
    )
    generation = prior.evaluate(
        model,
        generation_loader,
        cfg,
        pad,
        bos,
        eos,
        processor,
        generate=True,
    )
    interventions = {}
    for name in INTERVENTIONS:
        changed = prior.evaluate(
            model,
            test_loader,
            cfg,
            pad,
            bos,
            eos,
            processor,
            intervention=name,
        )
        interventions[name] = {
            "nll": changed["nll"],
            "damage": changed["nll"] - test["nll"],
        }
    audit = route_audit(model, test_loader, args, bos, pad)
    repetition = repetition_audit(model, splits[2], args, pad, bos, eos)
    learned_parameters = (
        parameter_count(model.decoder.learned_paths)
        if model.decoder.learned_paths is not None
        else 0
    )
    return {
        "arm": arm,
        "parameters": parameter_count(model),
        "learned_path_parameters": learned_parameters,
        "finite_gradients": finite,
        "initial_shared_sha256": shared_hash,
        "step_zero_nll": step_zero_nll,
        "final_model_sha256": state_hash(model.state_dict().items()),
        "seconds": time.time() - started,
        "trace": trace,
        "test": test,
        "generation": generation,
        "repetition": repetition,
        "interventions": interventions,
        "route_audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv"
    )
    parser.add_argument(
        "--spm-model", default="/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model"
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--valid-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--generation-samples", type=int, default=128)
    parser.add_argument("--max-scan", type=int, default=120000)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=31)
    parser.add_argument("--heap-width", type=int, default=32)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--audit-batches", type=int, default=4)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, sampling = adaptive.load_rows(args, processor)
    pieces = processor.get_piece_size()
    pad = pieces
    bos = processor.bos_id()
    eos = processor.eos_id()
    vocab = pieces + 1
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    started = time.time()
    for arm in arms:
        results[arm] = train_arm(
            arm, splits, args, vocab, pad, bos, eos, processor
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    shared_hashes = {
        result["initial_shared_sha256"] for result in results.values()
    }
    step_zero_pair_delta = None
    if "qk_no_path" in results and "qk_learned_path" in results:
        step_zero_pair_delta = abs(
            results["qk_no_path"]["step_zero_nll"]
            - results["qk_learned_path"]["step_zero_nll"]
        )
    summary = {
        "claim": "S3-TREE-QK-PATH-C01",
        "claim_scope": "one-seed real-corpus wiring and mechanism smoke",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args) | {
            "evidence_dir": str(args.evidence_dir),
            "arms": list(arms),
        },
        "data": {
            "direction": "en_to_zh",
            "selected_rows_sha256": selected_rows_hash(rows),
            "spm_sha256": sha256_file(Path(args.spm_model)),
            **sampling,
        },
        "contracts": {
            "shared_initial_state_equal": len(shared_hashes) == 1,
            "shared_initial_sha256": next(iter(shared_hashes)),
            "qk_no_path_vs_learned_step_zero_nll_delta": step_zero_pair_delta,
            "same_selected_rows_and_batch_seed": True,
            "balanced_heap_width": args.heap_width,
            "relation_is_qk": True,
            "read_is_separate_value_sum": True,
        },
        "results": results,
        "boundary": (
            "This smoke tests path-aware QK wiring and causal use. "
            "It cannot establish architecture superiority, sparse-compute "
            "benefit, power savings, or final language quality."
        ),
    }
    summary_path = args.evidence_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    examples_path = args.evidence_dir / "examples.json"
    examples_path.write_text(
        json.dumps(
            {
                arm: result["generation"]["examples"]
                for arm, result in results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = args.evidence_dir / "trace.jsonl"
    trace_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for result in results.values()
            for row in result["trace"]
        ),
        encoding="utf-8",
    )
    readme_path = args.evidence_dir / "README.md"
    readme_path.write_text(
        "# Tree QK path-encoding real-corpus smoke\n\n"
        "A deterministic WMT English-Chinese slice compares uniform routing, "
        "content-only QK, fixed path-aware QK and learned path-aware QK. "
        "R is QK; Value/READ is evaluated as a separate leaf-content sum. "
        "See summary.json for contracts, interventions and evidence boundary.\n",
        encoding="utf-8",
    )
    files = (summary_path, examples_path, trace_path, readme_path)
    (args.evidence_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    compact = {
        arm: {
            "parameters": result["parameters"],
            "step_zero_nll": result["step_zero_nll"],
            "test_nll": result["test"]["nll"],
            "bleu4": result["generation"]["token_bleu4"],
            "effective_leaves": result["route_audit"]["native_effective_leaves"],
            "path_zero_damage": result["interventions"]["path_zero"]["damage"],
            "path_shuffle_damage": result["interventions"]["path_shuffle"]["damage"],
            "route_jsd": result["route_audit"]["route_jsd"],
        }
        for arm, result in results.items()
    }
    print(
        json.dumps(
            {"contracts": summary["contracts"], "results": compact},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
