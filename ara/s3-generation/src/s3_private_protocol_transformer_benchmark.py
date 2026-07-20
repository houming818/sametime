#!/usr/bin/env python3
"""Parameter-matched small-Transformer benchmark for SPR-064 Stage A."""
from __future__ import annotations

import argparse
import copy
import json
import math
import socket
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
import s2_adaptive_lifting_wmt as adaptive
import s3_private_protocol_battle as battle


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        pad: int,
        model_dim: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        feedforward: int,
        dropout: float,
        max_positions: int,
    ):
        super().__init__()
        self.pad = pad
        self.model_dim = model_dim
        self.max_positions = max_positions
        self.source_embedding = nn.Embedding(vocab, model_dim, padding_idx=pad)
        self.target_embedding = nn.Embedding(vocab, model_dim, padding_idx=pad)
        self.position_embedding = nn.Embedding(max_positions, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, encoder_layers, norm=nn.LayerNorm(model_dim),
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, decoder_layers, norm=nn.LayerNorm(model_dim),
        )
        self.output = nn.Linear(model_dim, vocab)

    def positions(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.max_positions:
            raise ValueError(f"sequence length {length} exceeds {self.max_positions}")
        return torch.arange(length, device=device)

    def embed(self, tokens: torch.Tensor, table: nn.Embedding) -> torch.Tensor:
        pos = self.position_embedding(self.positions(tokens.shape[1], tokens.device))
        return table(tokens) * math.sqrt(self.model_dim) + pos[None]

    def encode(self, src: torch.Tensor, length: torch.Tensor):
        valid = torch.arange(src.shape[1], device=src.device)[None] < length[:, None]
        memory = self.encoder(
            self.embed(src, self.source_embedding),
            src_key_padding_mask=~valid,
        )
        return memory, valid

    def decode(self, previous: torch.Tensor, memory: torch.Tensor, source_valid: torch.Tensor):
        size = previous.shape[1]
        causal = torch.triu(
            torch.ones((size, size), device=previous.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding = previous.eq(self.pad)
        state = self.decoder(
            self.embed(previous, self.target_embedding),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=~source_valid,
        )
        return self.output(state)

    def teacher(self, src, length, target, bos, **kwargs):
        memory, source_valid = self.encode(src, length)
        previous = torch.full_like(target, self.pad)
        previous[:, 0] = bos
        if target.shape[1] > 1:
            previous[:, 1:] = target[:, :-1]
        return self.decode(previous, memory, source_valid), None

    @torch.no_grad()
    def greedy(self, src, length, bos, eos, max_len, **kwargs):
        memory, source_valid = self.encode(src, length)
        tokens = torch.full(
            (src.shape[0], 1), bos, device=src.device, dtype=torch.long,
        )
        output = []
        done = torch.zeros(src.shape[0], device=src.device, dtype=torch.bool)
        for _ in range(max_len):
            next_token = self.decode(tokens, memory, source_valid)[:, -1].argmax(-1)
            output.append(next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            done |= next_token.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(output, dim=1), None


def make_model(args, vocab: int, pad: int, recipe: str) -> SmallTransformer:
    return SmallTransformer(
        vocab=vocab,
        pad=pad,
        model_dim=args.tf_dim,
        heads=args.tf_heads,
        encoder_layers=args.tf_encoder_layers,
        decoder_layers=args.tf_decoder_layers,
        feedforward=args.tf_feedforward,
        dropout=0.0 if recipe == "same_recipe" else args.tf_dropout,
        max_positions=args.max_positions,
    )


def make_optimizer(model, recipe: str, args, total_steps: int):
    if recipe == "same_recipe":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.same_lr)
        return optimizer, None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.standard_lr, betas=(0.9, 0.98), eps=1e-9,
    )
    warmup = max(1, int(total_steps * args.warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one(recipe, seed, loaders, args, vocab, pad, bos, eos, sp, output):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(args, vocab, pad, recipe).to(args.device)
    epochs = args.same_epochs if recipe == "same_recipe" else args.standard_epochs
    total_steps = epochs * len(loaders[0])
    optimizer, scheduler = make_optimizer(model, recipe, args, total_steps)
    best_nll, best = float("inf"), None
    trace = []
    finite = True
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = steps = 0
        for src, length, target, _ in loaders[0]:
            src = src.to(args.device, non_blocking=True)
            length = length.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits, _ = model.teacher(src, length, target, bos)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=pad,
                label_smoothing=0.0 if recipe == "same_recipe" else args.label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite = finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_sum += float(loss.detach())
            steps += 1
        valid = battle.evaluate(model, loaders[1], args, pad, bos, eos, sp)
        row = {
            "recipe": recipe,
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(1, steps),
            "valid_nll": valid["nll"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
    if best is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best)
    test = battle.evaluate(
        model, loaders[2], args, pad, bos, eos, sp, generate=True,
    )
    checkpoint = output / f"checkpoint_{recipe}_seed{seed}.pt"
    torch.save({
        "recipe": recipe,
        "seed": seed,
        "state_dict": best,
        "config": vars(args),
    }, checkpoint)
    row = {
        "recipe": recipe,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
        "finite_gradients": finite,
        "trace": trace,
        "test": test,
        "checkpoint": checkpoint.name,
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def verify_frozen_contract(args, baseline: dict):
    config = baseline["config"]
    expected = {
        "data": args.data,
        "spm_model": args.spm_model,
        "seeds": args.seeds,
        "train_samples": args.train_samples,
        "valid_samples": args.valid_samples,
        "test_samples": args.test_samples,
        "max_scan": args.max_scan,
        "min_len": args.min_len,
        "max_len": args.max_len,
    }
    mismatch = {
        key: {"baseline": config.get(key), "benchmark": value}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatch:
        raise ValueError(f"frozen Stage A contract mismatch: {mismatch}")


def aggregate(rows: List[dict]) -> Dict[str, dict]:
    result = {}
    for recipe in sorted({row["recipe"] for row in rows}):
        selected = [row for row in rows if row["recipe"] == recipe]
        nll = [row["test"]["nll"] for row in selected]
        bleu = [row["test"]["token_bleu4"] for row in selected]
        result[recipe] = {
            "nll_mean": mean(nll),
            "nll_stdev": stdev(nll) if len(nll) > 1 else 0.0,
            "bleu4_mean": mean(bleu),
            "bleu4_stdev": stdev(bleu) if len(bleu) > 1 else 0.0,
            "parameters": selected[0]["parameters"],
            "seconds_mean": mean(row["seconds"] for row in selected),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--recipes", nargs="+", default=["same_recipe", "standard_recipe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[71901, 71902, 71903])
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--valid-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--max-scan", type=int, default=300000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tf-dim", type=int, default=256)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--tf-encoder-layers", type=int, default=2)
    parser.add_argument("--tf-decoder-layers", type=int, default=2)
    parser.add_argument("--tf-feedforward", type=int, default=512)
    parser.add_argument("--tf-dropout", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--same-epochs", type=int, default=4)
    parser.add_argument("--standard-epochs", type=int, default=8)
    parser.add_argument("--same-lr", type=float, default=2e-3)
    parser.add_argument("--standard-lr", type=float, default=5e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--smoke", action="store_true",
        help="allow a reduced split for code-path validation; never use for formal evidence",
    )
    args = parser.parse_args()
    if set(args.recipes) - {"same_recipe", "standard_recipe"}:
        raise ValueError("unknown recipe")

    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    if not args.smoke:
        verify_frozen_contract(args, baseline)
    random_seed = 71900
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    sampling_args = copy.copy(args)
    sampling_args.seed = random_seed
    rows, sampling = adaptive.load_rows(sampling_args, sp)
    if not args.smoke and sampling != baseline["data"]["sampling"]:
        raise ValueError("deterministic sampling no longer matches Stage A")
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    if vocab != baseline["data"]["vocab"]:
        raise ValueError("vocabulary no longer matches Stage A")
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    loaders = [
        DataLoader(
            base.ParallelDataset(split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=base.collate(pad),
            pin_memory=args.device.startswith("cuda"),
        )
        for split in splits
    ]
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe = make_model(args, vocab, pad, "standard_recipe")
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    del probe
    h1_parameters = baseline["aggregate"]["h1"]["parameters"]
    parameter_gap = abs(parameter_count - h1_parameters) / h1_parameters
    if parameter_gap > 0.05:
        raise ValueError(f"parameter mismatch {parameter_gap:.3%} exceeds 5%")

    started = time.time()
    results = []
    for recipe in args.recipes:
        for seed in args.seeds:
            results.append(train_one(
                recipe, seed, loaders, args, vocab, pad, bos, eos, sp, output,
            ))
    scores = aggregate(results)
    tree_nll = baseline["aggregate"]["h1"]["nll_mean"]
    standard_nll = scores.get("standard_recipe", scores[args.recipes[-1]])["nll_mean"]
    tree_gap = tree_nll - standard_nll
    decision = (
        "smoke_only" if args.smoke else
        "competitive" if tree_gap <= 0.02 else
        "weak_parity" if tree_gap <= 0.10 else "behind"
    )
    summary = {
        "claim": "S3-PRIVATE-PROTOCOL-TF-C02",
        "predict": "P-S3-PRIVATE-PROTOCOL-TF-02",
        "status": decision,
        "smoke": args.smoke,
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "sampling": sampling,
        "transformer": scores,
        "registered_baselines": baseline["aggregate"],
        "treeheap_h1_minus_standard_transformer_nll": tree_gap,
        "parameter_gap_fraction_vs_h1": parameter_gap,
        "boundary": "Small matched Transformer benchmark; not an industry-top or global-optimum result.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "runs.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    trace = [item for row in results for item in row["trace"]]
    (output / "trace.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in trace) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Small Transformer benchmark\n\n```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
