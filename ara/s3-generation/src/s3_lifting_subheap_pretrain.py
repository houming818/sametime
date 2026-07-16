#!/usr/bin/env python3
"""Real-text multi-scale subheap pretraining for the reversible lifting model."""
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
from typing import Dict, Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as lifting
import s3_wmt_treeheap_seq2seq as base
from s3_conditional_denoising_seq2seq import MixedDocuments


WIDTHS = (1, 2, 4, 8)
VARIANTS = ("token_only", "random_span", "subheap")


class SubheapBlocks(IterableDataset):
    def __init__(
        self,
        root: Path,
        spm_path: str,
        split: str,
        seed: int,
        block_length: int,
        variant: str,
        forced_width: int = 0,
    ):
        self.root = root
        self.spm_path = spm_path
        self.split = split
        self.seed = seed
        self.block_length = block_length
        self.variant = variant
        self.forced_width = forced_width

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        sp = spm.SentencePieceProcessor(model_file=self.spm_path)
        mask_id = sp.get_piece_size()
        rng = random.Random(self.seed)
        for text in MixedDocuments(self.root, self.split, self.seed):
            ids = sp.encode(text, out_type=int)
            for offset in range(0, len(ids) - self.block_length + 1, self.block_length):
                clean = ids[offset : offset + self.block_length]
                if self.forced_width:
                    width = self.forced_width
                elif self.variant == "token_only":
                    width = 1
                else:
                    width = rng.choice(WIDTHS)
                if self.variant == "random_span" and not self.forced_width:
                    start = rng.randrange(self.block_length - width + 1)
                else:
                    slot = rng.randrange(self.block_length // width)
                    start = slot * width
                damaged = list(clean)
                damaged[start : start + width] = [mask_id] * width
                target = clean[start : start + width] + [sp.eos_id()]
                yield (
                    torch.tensor(damaged, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                    torch.tensor(start, dtype=torch.long),
                    torch.tensor(width, dtype=torch.long),
                )


def collate(pad: int):
    def fn(batch: Sequence[Tuple[torch.Tensor, ...]]):
        source = torch.stack([row[0] for row in batch])
        target_length = max(row[1].numel() for row in batch)
        target = torch.full((len(batch), target_length), pad, dtype=torch.long)
        for index, row in enumerate(batch):
            target[index, : row[1].numel()] = row[1]
        start = torch.stack([row[2] for row in batch])
        width = torch.stack([row[3] for row in batch])
        return source, target, start, width
    return fn


def render_masked(sp: spm.SentencePieceProcessor, ids: Sequence[int], mask_id: int) -> str:
    pieces: List[str] = []
    pending: List[int] = []
    for token in ids:
        if int(token) == mask_id:
            if pending:
                pieces.append(sp.decode(pending))
                pending = []
            pieces.append("[MASK]")
        else:
            pending.append(int(token))
    if pending:
        pieces.append(sp.decode(pending))
    return "".join(pieces)


def clean(ids: Sequence[int], eos: int, pad: int) -> List[int]:
    answer: List[int] = []
    for token in ids:
        if int(token) in (eos, pad):
            break
        answer.append(int(token))
    return answer


def transformed_levels(model, source, length, intervention: str):
    encoder_intervention = intervention if intervention.startswith("detail_shuffle_") else "native"
    working_source = source
    if intervention == "sibling_swap":
        working_source = source.reshape(source.shape[0], -1, 2).flip(2).reshape_as(source)
    _, _, _, levels, masks = model.states(working_source, length, encoder_intervention)
    levels = [row.clone() for row in levels]
    masks = [row.clone() for row in masks]
    if intervention == "native" or intervention.startswith("detail_shuffle_"):
        pass
    elif intervention == "source_shuffle":
        levels = [row.roll(1, dims=0) for row in levels]
        masks = [row.roll(1, dims=0) for row in masks]
    elif intervention == "root_zero":
        levels[0].zero_()
    elif intervention == "no_source":
        levels = [torch.zeros_like(row) for row in levels]
    elif intervention == "sibling_swap":
        pass
    else:
        raise ValueError(intervention)
    return levels, masks


def teacher(model, source, target, bos: int, intervention: str = "native"):
    length = torch.full(
        (source.shape[0],), source.shape[1], device=source.device, dtype=torch.long,
    )
    levels, masks = transformed_levels(model, source, length, intervention)
    return model.decoder.teacher(levels, masks, target, bos)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    pad: int,
    bos: int,
    eos: int,
    sp,
    limit: int,
    intervention: str = "native",
    generate: bool = False,
) -> dict:
    model.eval()
    loss_sum = tokens = token_hit = greedy_hit = greedy_tokens = exact = samples = 0
    repeat_sum = 0.0
    generated_rows: List[Tuple[int, ...]] = []
    route_sum = None
    examples: List[dict] = []
    for batch_no, (source, target, start, width) in enumerate(loader, 1):
        source, target = source.to(device), target.to(device)
        logits, route = teacher(model, source, target, bos, intervention)
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(valid.sum())
        prediction = logits.argmax(-1)
        token_hit += int((prediction.eq(target) & valid).sum())
        route_sum = route.detach().double() if route_sum is None else route_sum + route.detach().double()
        if generate:
            levels, masks = transformed_levels(
                model, source,
                torch.full((source.shape[0],), source.shape[1], device=device, dtype=torch.long),
                intervention,
            )
            generated, _ = model.decoder.greedy(levels, masks, bos, eos, target.shape[1])
            if generated.shape[1] < target.shape[1]:
                generated = F.pad(generated, (0, target.shape[1] - generated.shape[1]), value=pad)
            generated = generated[:, : target.shape[1]]
            greedy_hit += int((generated.eq(target) & valid).sum())
            greedy_tokens += int(valid.sum())
            for row in range(source.shape[0]):
                reference = clean(target[row].cpu().tolist(), eos, pad)
                hypothesis = clean(generated[row].cpu().tolist(), eos, pad)
                exact += int(reference == hypothesis)
                samples += 1
                generated_rows.append(tuple(hypothesis))
                if len(hypothesis) > 1:
                    repeat_sum += sum(
                        hypothesis[index] == hypothesis[index - 1]
                        for index in range(1, len(hypothesis))
                    ) / (len(hypothesis) - 1)
                if len(examples) < 8:
                    examples.append({
                        "masked": render_masked(sp, source[row].cpu().tolist(), pad),
                        "address": int(start[row]),
                        "width": int(width[row]),
                        "reference": sp.decode(reference),
                        "generated": sp.decode(hypothesis),
                    })
        if batch_no >= limit:
            break
    nll = loss_sum / max(1, tokens)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "teacher_token_accuracy": token_hit / max(1, tokens),
        "greedy_token_accuracy": greedy_hit / max(1, greedy_tokens) if generate else None,
        "greedy_exact": exact / max(1, samples) if generate else None,
        "nonempty_fraction": (
            sum(bool(row) for row in generated_rows) / max(1, samples)
            if generate else None
        ),
        "adjacent_repeat_fraction": repeat_sum / max(1, samples) if generate else None,
        "unique_output_fraction": len(set(generated_rows)) / max(1, samples) if generate else None,
        "route_depth_mass": [float(x) for x in (route_sum / max(1, batch_no)).cpu()],
        "examples": examples,
    }


@torch.no_grad()
def closure(model, loader, device: str) -> dict:
    source, _, _, _ = next(iter(loader))
    source = source.to(device)
    length = torch.full((source.shape[0],), source.shape[1], device=device, dtype=torch.long)
    leaf, _, _, levels, _ = model.states(source, length)
    difference = levels[-1] - leaf
    return {
        "state_mse": float(difference.square().mean()),
        "state_max_abs": float(difference.abs().max()),
    }


def make_loader(args, split: str, variant: str, seed: int, forced_width: int = 0):
    dataset = SubheapBlocks(
        Path(args.root), args.spm_model, split, seed, args.block_length,
        variant, forced_width,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch,
        collate_fn=collate(args.pad),
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def train_variant(name: str, args, sp, output: Path) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = lifting.make_model(
        "learned_update", args.vocab, args.dim, args.hidden,
        args.heap_width, args.pad,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = make_loader(args, "train", name, args.seed)
    train_iter = iter(train_loader)
    common_loaders = {
        width: make_loader(args, "valid", name, args.seed + 1000 + width, width)
        for width in WIDTHS
    }
    initial = {
        str(width): evaluate(
            model, common_loaders[width], args.device, args.pad, args.bos,
            args.eos, sp, args.valid_batches,
        )["nll"]
        for width in (4, 8)
    }
    trace: List[dict] = []
    started = time.time()
    finite = True
    best_nll = float("inf")
    best = None
    for step in range(1, args.steps + 1):
        model.train()
        source, target, _, _ = next(train_iter)
        source, target = source.to(args.device), target.to(args.device)
        logits, _ = teacher(model, source, target, args.bos)
        loss = base.ce(logits, target, args.pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.valid_every == 0 or step == args.steps:
            valid = {
                str(width): evaluate(
                    model, common_loaders[width], args.device, args.pad,
                    args.bos, args.eos, sp, args.valid_batches,
                )["nll"]
                for width in WIDTHS
            }
            mean_deep = (valid["4"] + valid["8"]) / 2
            row = {
                "variant": name,
                "step": step,
                "train_nll": float(loss.detach()),
                "valid_nll_by_width": valid,
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
            if mean_deep < best_nll:
                best_nll = mean_deep
                best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best)
    test = {}
    for width in WIDTHS:
        loader = make_loader(args, "test", name, args.seed + 2000 + width, width)
        test[str(width)] = evaluate(
            model, loader, args.device, args.pad, args.bos, args.eos, sp,
            args.test_batches, generate=True,
        )
    audit_loader = make_loader(args, "test", name, args.seed + 3008, 8)
    native = evaluate(
        model, audit_loader, args.device, args.pad, args.bos, args.eos,
        sp, args.test_batches,
    )["nll"]
    interventions = {}
    for intervention in ["source_shuffle", "root_zero", "no_source", "sibling_swap"]:
        result = evaluate(
            model, audit_loader, args.device, args.pad, args.bos, args.eos,
            sp, args.test_batches, intervention,
        )
        interventions[intervention] = {
            "nll": result["nll"],
            "damage": result["nll"] - native,
        }
    interventions["detail_shuffle"] = []
    for depth in range(model.encoder.depths):
        result = evaluate(
            model, audit_loader, args.device, args.pad, args.bos, args.eos,
            sp, args.test_batches, f"detail_shuffle_{depth}",
        )
        interventions["detail_shuffle"].append({
            "depth": depth,
            "nll": result["nll"],
            "damage": result["nll"] - native,
        })
    checkpoint = output / f"checkpoint_{name}.pt"
    torch.save({
        "name": name,
        "state_dict": best,
        "config": vars(args),
        "trace": trace,
    }, checkpoint)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": finite,
        "seconds": time.time() - started,
        "initial_nll": initial,
        "trace": trace,
        "test": test,
        "interventions_width8": interventions,
        "closure": closure(model, audit_loader, args.device),
        "checkpoint": checkpoint.name,
    }


def build_gates(results: Dict[str, dict]) -> Tuple[dict, dict]:
    token = results["token_only"]
    random_span = results["random_span"]
    subheap = results["subheap"]
    token_gains = {
        width: token["test"][str(width)]["nll"] - subheap["test"][str(width)]["nll"]
        for width in WIDTHS
    }
    matched_gains = {
        width: random_span["test"][str(width)]["nll"] - subheap["test"][str(width)]["nll"]
        for width in WIDTHS
    }
    accuracy_gains = {
        width: (
            subheap["test"][str(width)]["greedy_token_accuracy"]
            - random_span["test"][str(width)]["greedy_token_accuracy"]
        )
        for width in WIDTHS
    }
    interventions = subheap["interventions_width8"]
    detail_damage = [row["damage"] for row in interventions["detail_shuffle"]]
    derived = {
        "subheap_nll_gain_over_token_by_width": token_gains,
        "subheap_nll_gain_over_random_span_by_width": matched_gains,
        "subheap_greedy_accuracy_gain_over_random_span_by_width": accuracy_gains,
        "subheap_source_shuffle_damage_width8": interventions["source_shuffle"]["damage"],
        "subheap_root_zero_damage_width8": interventions["root_zero"]["damage"],
        "subheap_no_source_damage_width8": interventions["no_source"]["damage"],
        "subheap_sibling_swap_damage_width8": interventions["sibling_swap"]["damage"],
        "subheap_detail_shuffle_damage_width8": detail_damage,
    }
    gates = {
        "P1_finite_and_learning": all(
            row["finite_gradients"]
            and row["test"]["8"]["nll"] < row["initial_nll"]["8"]
            for row in results.values()
        ),
        "P2_subheap_beats_matched_span_on_deep": (
            (matched_gains[4] + matched_gains[8]) / 2 >= 0.03
            and not (token_gains[4] < 0 and token_gains[8] < 0)
        ),
        "P3_generation_nonempty_and_accuracy_gain": (
            subheap["test"]["8"]["nonempty_fraction"] >= 0.75
            and subheap["test"]["8"]["adjacent_repeat_fraction"] <= 0.40
            and subheap["test"]["8"]["unique_output_fraction"] >= 0.10
            and sum(value > 0 for value in accuracy_gains.values()) >= 3
        ),
        "P4_source_causal": interventions["source_shuffle"]["damage"] >= 0.10,
        "P5_root_and_depth_causal": (
            interventions["root_zero"]["damage"] >= 0.03
            and sum(value >= 0.03 for value in detail_damage) >= 2
        ),
        "P6_addresses_causal": interventions["sibling_swap"]["damage"] >= 0.03,
        "P7_closed": subheap["closure"]["state_mse"] < 1e-10,
    }
    return derived, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_lifting_subheap_pretrain_smoke")
    parser.add_argument("--seed", type=int, default=71903)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--valid-every", type=int, default=100)
    parser.add_argument("--valid-batches", type=int, default=8)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.block_length != args.heap_width or args.heap_width & (args.heap_width - 1):
        raise ValueError("this proof requires block_length == power-of-two heap_width")
    if set(args.variants) != set(VARIANTS):
        raise ValueError("the registered proof requires token_only and subheap")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    args.pad = sp.get_piece_size()
    args.vocab = args.pad + 1
    args.bos = sp.bos_id()
    args.eos = sp.eos_id()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: Dict[str, dict] = {}
    for name in args.variants:
        results[name] = train_variant(name, args, sp, output)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    derived, gates = build_gates(results)
    summary = {
        "claim": "S3-LIFT-SUBHEAP-PRETRAIN-C01",
        "status": "supported pilot" if all(gates.values()) else "partial or rejected; inspect gates",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "results": results,
        "derived": derived,
        "gates": gates,
        "boundary": "Real-text self-supervised generation proof, not WMT transfer, world knowledge, or architecture superiority.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Lifting Subheap Pretraining Evidence\n\n"
        "Preregistered token-only versus address-aligned multi-scale subheap masking.\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": derived, "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
