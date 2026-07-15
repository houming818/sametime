#!/usr/bin/env python3
"""Streaming full-data smoke for S1 WMT canonical echo.

This is a stability probe, not a final metric run. It never materializes the
full 14M corpus in memory. It streams TSV lines, tokenizes a batch, hashes
tokens into fixed vocab IDs, moves only the batch to GPU, and trains selected
canonical encoders for a bounded number of steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0
UNK = 1
EN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,!?;:()\"-]")
ZH_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[，。！？；：（）“”《》、,.!?;:()\"-]")


def tok_en(text: str) -> list[str]:
    return [x.lower() for x in EN_RE.findall(text)]


def tok_zh(text: str) -> list[str]:
    return ZH_RE.findall(text)


def hid(token: str, vocab: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=4).digest()
    return 2 + (int.from_bytes(h, "little") % max(1, vocab - 2))


def encode(tokens: list[str], vocab: int, max_len: int) -> list[int]:
    ids = [hid(t, vocab) for t in tokens[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


def iter_batches(paths: list[str], args):
    en_buf, zh_buf = [], []
    seen = accepted = 0
    while True:
        for path in paths:
            with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    seen += 1
                    if "\t" not in line:
                        continue
                    a, b = line.rstrip("\n").split("\t", 1)
                    if sum("\u4e00" <= ch <= "\u9fff" for ch in a) > sum("\u4e00" <= ch <= "\u9fff" for ch in b):
                        zh_raw, en_raw = a, b
                    else:
                        en_raw, zh_raw = a, b
                    en_t = tok_en(en_raw)
                    zh_t = tok_zh(zh_raw)
                    if not (args.min_len <= len(en_t) <= args.max_len and args.min_len <= len(zh_t) <= args.max_len):
                        continue
                    accepted += 1
                    en_buf.append(encode(en_t, args.en_vocab, args.max_len))
                    zh_buf.append(encode(zh_t, args.zh_vocab, args.max_len))
                    if len(en_buf) >= args.batch:
                        yield torch.tensor(en_buf, dtype=torch.long), torch.tensor(zh_buf, dtype=torch.long), seen, accepted
                        en_buf, zh_buf = [], []


class ComposeKernel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.LayerNorm(dim))

    def forward(self, left, right):
        return self.net(torch.cat([left, right, left * right, left - right], dim=-1))


class TreeHeapCanonical(nn.Module):
    kind = "treeheap"

    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.en_emb = nn.Embedding(en_vocab, dim, padding_idx=PAD)
        self.zh_emb = nn.Embedding(zh_vocab, dim, padding_idx=PAD)
        self.path_emb = nn.Embedding(max_len, dim)
        self.en_leaf = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.zh_leaf = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.compose = ComposeKernel(dim)
        self.mean_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.root = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.en_dec = nn.Linear(dim, en_vocab)
        self.zh_dec = nn.Linear(dim, zh_vocab)

    def encode_side(self, ids, lang):
        pos = torch.arange(self.max_len, device=ids.device).unsqueeze(0)
        leaf = (self.en_emb(ids) if lang == "en" else self.zh_emb(ids)) + self.path_emb(pos)
        leaf = self.en_leaf(leaf) if lang == "en" else self.zh_leaf(leaf)
        cur = leaf
        while cur.shape[1] > 1:
            if cur.shape[1] % 2 == 1:
                cur = torch.cat([cur, torch.zeros_like(cur[:, -1:, :])], dim=1)
            cur = self.compose(cur[:, 0::2, :], cur[:, 1::2, :])
        mask = (ids != PAD).float().unsqueeze(-1)
        mean = (leaf * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.root(cur[:, 0, :]) + self.mean_proj(mean), dim=-1), leaf

    def forward(self, en, zh):
        er, el = self.encode_side(en, "en")
        zr, zl = self.encode_side(zh, "zh")
        return er, zr, self.en_dec(el), self.zh_dec(zl)


class BoWCanonical(nn.Module):
    kind = "bow"

    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.en_emb = nn.Embedding(en_vocab, dim, padding_idx=PAD)
        self.zh_emb = nn.Embedding(zh_vocab, dim, padding_idx=PAD)
        self.en_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.zh_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.en_dec = nn.Linear(dim, en_vocab)
        self.zh_dec = nn.Linear(dim, zh_vocab)

    @staticmethod
    def pool(x, ids):
        m = (ids != PAD).float().unsqueeze(-1)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, en, zh):
        el, zl = self.en_emb(en), self.zh_emb(zh)
        return F.normalize(self.en_proj(self.pool(el, en)), dim=-1), F.normalize(self.zh_proj(self.pool(zl, zh)), dim=-1), self.en_dec(el), self.zh_dec(zl)


class LSTMCanonical(BoWCanonical):
    kind = "lstm"

    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__(en_vocab, zh_vocab, max_len, dim)
        self.en_lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.zh_lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)

    def forward(self, en, zh):
        el, _ = self.en_lstm(self.en_emb(en))
        zl, _ = self.zh_lstm(self.zh_emb(zh))
        return F.normalize(self.en_proj(self.pool(el, en)), dim=-1), F.normalize(self.zh_proj(self.pool(zl, zh)), dim=-1), self.en_dec(el), self.zh_dec(zl)


class TransformerCanonical(BoWCanonical):
    kind = "transformer"

    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__(en_vocab, zh_vocab, max_len, dim)
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation="gelu")
        self.en_tx = nn.TransformerEncoder(layer, 2)
        zlayer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation="gelu")
        self.zh_tx = nn.TransformerEncoder(zlayer, 2)

    def forward(self, en, zh):
        p = torch.arange(self.max_len, device=en.device).unsqueeze(0)
        el = self.en_tx(self.en_emb(en) + self.pos(p), src_key_padding_mask=(en == PAD))
        zl = self.zh_tx(self.zh_emb(zh) + self.pos(p), src_key_padding_mask=(zh == PAD))
        return F.normalize(self.en_proj(self.pool(el, en)), dim=-1), F.normalize(self.zh_proj(self.pool(zl, zh)), dim=-1), self.en_dec(el), self.zh_dec(zl)


MODELS = {"treeheap": TreeHeapCanonical, "bow": BoWCanonical, "lstm": LSTMCanonical, "transformer": TransformerCanonical}


def losses(er, zr, el, zl, en, zh, temp):
    logits = er @ zr.t() / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    align = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
    echo = (F.cross_entropy(el.reshape(-1, el.shape[-1]), en.reshape(-1), ignore_index=PAD) + F.cross_entropy(zl.reshape(-1, zl.shape[-1]), zh.reshape(-1), ignore_index=PAD)) * 0.5
    margin = float((1 - logits.detach().diag().mul(temp)).mean().cpu())
    return align, echo, margin


def train_model(name, args, paths, device):
    model = MODELS[name](args.en_vocab, args.zh_vocab, args.max_len, args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    gen = iter_batches(paths, args)
    rows = []
    started = time.time()
    for step in range(1, args.steps + 1):
        en, zh, seen, accepted = next(gen)
        en = en.to(device)
        zh = zh.to(device)
        opt.zero_grad(set_to_none=True)
        er, zr, el, zl = model(en, zh)
        align, echo, pos_dist = losses(er, zr, el, zl, en, zh, args.temperature)
        loss = align + args.echo_weight * echo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            rows.append({
                "model": name,
                "step": step,
                "seen_lines": seen,
                "accepted_pairs": accepted,
                "loss": float(loss.detach().cpu()),
                "align": float(align.detach().cpu()),
                "echo": float(echo.detach().cpu()),
                "batch_positive_distance_proxy": pos_dist,
                "elapsed_sec": time.time() - started,
                "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_wmt_streaming_full_smoke")
    ap.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv")
    ap.add_argument("--models", default="transformer")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--en-vocab", type=int, default=8192)
    ap.add_argument("--zh-vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--echo-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    paths = [p.strip() for p in args.wmt_path.split(",") if p.strip()]
    all_rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        all_rows.extend(train_model(name, args, paths, device))
        torch.cuda.empty_cache()
    summary = {
        "purpose": "full-data streaming GPU stability smoke",
        "paths": paths,
        "models": args.models,
        "steps": args.steps,
        "batch": args.batch,
        "max_len": args.max_len,
        "rows": all_rows,
        "last_rows": [r for r in all_rows if r["step"] == args.steps],
        "pilot_pass": all(r["cuda_memory_allocated"] > 0 for r in all_rows if r["step"] == args.steps),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["last_rows"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
