#!/usr/bin/env python3
"""Real-Chinese conditional denoising gate for the TreeHeap seq2seq stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_treeheap_p0_pretrain import SOURCES, files, text_from
from s3_wmt_treeheap_seq2seq import BowSeq2Seq, FlatSeq2Seq, TreeHeapSeq2Seq, clean


MODELS = {"treeheap": TreeHeapSeq2Seq, "flat_seq": FlatSeq2Seq, "bow": BowSeq2Seq}


def document_bucket(source: str, text: str) -> int:
    payload = (source + "\0" + text[:2048]).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big") % 100


def in_split(bucket: int, split: str) -> bool:
    if split == "train":
        return bucket < 96
    if split == "valid":
        return 96 <= bucket < 98
    if split == "test":
        return bucket >= 98
    raise ValueError(split)


def documents(root: Path, source: str, split: str) -> Iterator[str]:
    while True:
        for path in files(root, source):
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = text_from(row, source)
                    if len(text) >= 64 and in_split(document_bucket(source, text), split):
                        yield text


class MixedDocuments:
    def __init__(self, root: Path, split: str, seed: int):
        self.rng = random.Random(seed)
        self.names = list(SOURCES)
        self.weights = [SOURCES[name] for name in self.names]
        self.streams = {name: documents(root, name, split) for name in self.names}

    def __iter__(self) -> Iterator[str]:
        while True:
            name = self.rng.choices(self.names, self.weights)[0]
            yield next(self.streams[name])


def span_mask(ids: List[int], mask_id: int, rng: random.Random, rate: float, max_span: int) -> Tuple[List[int], List[bool]]:
    n = len(ids)
    wanted = max(1, round(n * rate))
    selected = [False] * n
    count = 0
    while count < wanted:
        start = rng.randrange(n)
        span = rng.randint(1, max_span)
        for pos in range(start, min(n, start + span)):
            if not selected[pos]:
                selected[pos] = True
                count += 1
                if count >= wanted:
                    break
    damaged = [mask_id if selected[i] else token for i, token in enumerate(ids)]
    return damaged, selected


class DenoisingBlocks(IterableDataset):
    def __init__(self, root: Path, spm_path: str, split: str, seed: int, length: int, mask_rate: float, max_span: int, target_mode: str, target_length: int):
        self.root = root
        self.spm_path = spm_path
        self.split = split
        self.seed = seed
        self.length = length
        self.mask_rate = mask_rate
        self.max_span = max_span
        self.target_mode = target_mode
        self.target_length = target_length

    def __iter__(self):
        sp = spm.SentencePieceProcessor(model_file=self.spm_path)
        mask_id = sp.get_piece_size()
        rng = random.Random(self.seed)
        for text in MixedDocuments(self.root, self.split, self.seed):
            ids = sp.encode(text, out_type=int)
            for start in range(0, len(ids) - self.length + 1, self.length):
                clean_ids = ids[start:start + self.length]
                if self.target_mode == "full":
                    damaged, selected = span_mask(clean_ids, mask_id, rng, self.mask_rate, self.max_span)
                    target = clean_ids
                else:
                    gap_start = rng.randrange(0, self.length - self.target_length + 1)
                    gap_end = gap_start + self.target_length
                    damaged = clean_ids.copy()
                    damaged[gap_start:gap_end] = [mask_id] * self.target_length
                    target = clean_ids[gap_start:gap_end]
                    selected = [True] * self.target_length
                yield (
                    torch.tensor(damaged, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                    torch.tensor(selected, dtype=torch.bool),
                )


def collate(batch):
    return tuple(torch.stack(items) for items in zip(*batch))


def render_source(sp: spm.SentencePieceProcessor, ids: List[int], mask_id: int) -> str:
    pieces: List[str] = []
    run: List[int] = []
    for token in ids:
        if token == mask_id:
            if run:
                pieces.append(sp.decode(run))
                run = []
            pieces.append("[MASK]")
        else:
            run.append(token)
    if run:
        pieces.append(sp.decode(run))
    return "".join(pieces)


def repetition_fraction(ids: List[int]) -> float:
    if len(ids) < 2:
        return 0.0
    return sum(ids[i] == ids[i - 1] for i in range(1, len(ids))) / (len(ids) - 1)


@torch.no_grad()
def evaluate(model, loader, device: str, bos: int, eos: int, pad: int, sp, limit: int, mode: str = "full") -> Dict[str, object]:
    model.eval()
    nll_sum = token_count = masked_hit = masked_count = greedy_hit = greedy_count = exact = sample_count = 0
    repetitions: List[float] = []
    examples: List[dict] = []
    for batch_no, (src, target, selected) in enumerate(loader, 1):
        src, target, selected = src.to(device), target.to(device), selected.to(device)
        length = torch.full((src.shape[0],), src.shape[1], device=device, dtype=torch.long)
        memory, memory_mask = model.encode(src, length, mode)
        logits = model.decoder.teacher(memory, memory_mask, target, bos)
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").reshape_as(target)
        nll_sum += float(losses.sum().item())
        token_count += int(target.numel())
        teacher_pred = logits.argmax(-1)
        masked_hit += int((teacher_pred.eq(target) & selected).sum().item())
        masked_count += int(selected.sum().item())
        generated = model.decoder.greedy(memory, memory_mask, bos, eos, target.shape[1])
        if generated.shape[1] < target.shape[1]:
            generated = F.pad(generated, (0, target.shape[1] - generated.shape[1]), value=pad)
        generated = generated[:, :target.shape[1]]
        greedy_hit += int(generated.eq(target).sum().item())
        greedy_count += int(target.numel())
        exact += int(generated.eq(target).all(-1).sum().item())
        sample_count += int(target.shape[0])
        for row in range(src.shape[0]):
            out_ids = clean(generated[row].cpu().tolist(), eos, pad)
            repetitions.append(repetition_fraction(out_ids))
            if len(examples) < 8:
                examples.append({
                    "damaged": render_source(sp, src[row].cpu().tolist(), pad),
                    "reference": sp.decode(target[row].cpu().tolist()),
                    "generated": sp.decode(out_ids),
                })
        if batch_no >= limit:
            break
    nll = nll_sum / max(1, token_count)
    return {
        "nll": nll,
        "ppl": math.exp(min(20, nll)),
        "masked_teacher_accuracy": masked_hit / max(1, masked_count),
        "greedy_token_accuracy": greedy_hit / max(1, greedy_count),
        "greedy_exact": exact / max(1, sample_count),
        "adjacent_repeat_fraction": sum(repetitions) / max(1, len(repetitions)),
        "examples": examples,
    }


def train_one(name: str, args, sp, vocab: int, pad: int, bos: int, eos: int, out: Path) -> Dict[str, object]:
    torch.manual_seed(args.seed + sum(map(ord, name)))
    model = MODELS[name](vocab, args.dim, args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        DenoisingBlocks(Path(args.root), args.spm_model, "train", args.seed, args.length, args.mask_rate, args.max_span, args.target_mode, args.target_length),
        batch_size=args.batch,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        DenoisingBlocks(Path(args.root), args.spm_model, "valid", args.seed + 1, args.length, args.mask_rate, args.max_span, args.target_mode, args.target_length),
        batch_size=args.batch,
        collate_fn=collate,
        num_workers=0,
    )
    train_iter = iter(train_loader)
    trace = []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        src, target, _ = next(train_iter)
        src, target = src.to(args.device), target.to(args.device)
        length = torch.full((src.shape[0],), src.shape[1], device=args.device, dtype=torch.long)
        logits = model(src, length, target, bos)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.valid_every == 0 or step == args.steps:
            metrics = evaluate(model, valid_loader, args.device, bos, eos, pad, sp, args.valid_batches)
            row = {"step": step, "train_nll": float(loss.item()), **{k: v for k, v in metrics.items() if k != "examples"}}
            trace.append(row)
            print(json.dumps({"model": name, **row}), flush=True)
    test_loader = DataLoader(
        DenoisingBlocks(Path(args.root), args.spm_model, "test", args.seed + 2, args.length, args.mask_rate, args.max_span, args.target_mode, args.target_length),
        batch_size=args.batch,
        collate_fn=collate,
        num_workers=0,
    )
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.time() - started,
        "trace": trace,
        "test": evaluate(model, test_loader, args.device, bos, eos, pad, sp, args.test_batches),
    }
    if name == "treeheap":
        result["test_leaf_only"] = evaluate(model, test_loader, args.device, bos, eos, pad, sp, args.test_batches, "leaf_only")
        result["test_root_only"] = evaluate(model, test_loader, args.device, bos, eos, pad, sp, args.test_batches, "root_only")
    torch.save({"model": name, "state_dict": model.state_dict(), "config": vars(args)}, out / f"checkpoint_{name}.pt")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nio/datasets/pretrain")
    ap.add_argument("--spm-model", required=True)
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_conditional_denoising_smoke")
    ap.add_argument("--model", choices=["all", *MODELS], default="all")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--target-mode", choices=["full", "gap"], default="gap")
    ap.add_argument("--target-length", type=int, default=16)
    ap.add_argument("--mask-rate", type=float, default=0.30)
    ap.add_argument("--max-span", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--valid-every", type=int, default=250)
    ap.add_argument("--valid-batches", type=int, default=16)
    ap.add_argument("--test-batches", type=int, default=32)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0.0 < args.mask_rate < 1.0:
        raise ValueError("--mask-rate must be between zero and one")
    if not 0 < args.target_length <= args.length:
        raise ValueError("--target-length must be in [1, --length]")
    out = Path(args.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pad = sp.get_piece_size()
    vocab = pad + 1
    bos, eos = sp.bos_id(), sp.eos_id()
    names = list(MODELS) if args.model == "all" else [args.model]
    results = {name: train_one(name, args, sp, vocab, pad, bos, eos, out) for name in names}
    summary = {
        "claim": "S3-DENOISE-SEQ2SEQ-C01",
        "host": socket.gethostname(),
        "config": vars(args),
        "models": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "examples.json").write_text(json.dumps({name: result["test"]["examples"] for name, result in results.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps({"model": name, **row}) for name, result in results.items() for row in result["trace"]) + "\n", encoding="utf-8")
    (out / "README.md").write_text("# Conditional Denoising Seq2Seq\n\nSee `summary.json` and `examples.json`.\n", encoding="utf-8")
    print(json.dumps({name: result["test"] for name, result in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
