#!/usr/bin/env python3
"""S1 WMT canonical echo probe.

This probe tests the S1 idea that parallel surface forms can be mapped toward a
shared low-entropy canonical state while preserving enough leaf state to echo
the original surface tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
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


def detok_en(tokens: list[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([.,!?;:)])", r"\1", out)
    out = re.sub(r"([(])\s+", r"\1", out)
    return out


def detok_zh(tokens: list[str]) -> str:
    return "".join(tokens)


def read_pairs(args):
    pairs = []
    en_counter, zh_counter = Counter(), Counter()
    target = None if args.samples <= 0 else args.samples
    scan_limit = None if args.scan_lines <= 0 else args.scan_lines
    for raw_path in args.wmt_path.split(","):
        path = Path(raw_path.strip())
        if not path:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f):
                if scan_limit is not None and line_no >= scan_limit:
                    break
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                a, b = line.split("\t", 1)
                if sum("\u4e00" <= ch <= "\u9fff" for ch in a) > sum("\u4e00" <= ch <= "\u9fff" for ch in b):
                    zh, en = a, b
                else:
                    en, zh = a, b
                en_t = tok_en(en)
                zh_t = tok_zh(zh)
                if not (args.min_len <= len(en_t) <= args.max_len and args.min_len <= len(zh_t) <= args.max_len):
                    continue
                pairs.append((en_t, zh_t, en, zh))
                en_counter.update(en_t)
                zh_counter.update(zh_t)
                if target is not None and len(pairs) >= target * 2:
                    break
        if target is not None and len(pairs) >= target * 2:
            break
    if target is not None and len(pairs) < target:
        raise RuntimeError(f"only collected {len(pairs)} pairs, need {target}")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if target is not None:
        pairs = pairs[:target]
    en_vocab = ["<pad>", "<unk>"] + [w for w, _ in en_counter.most_common(args.en_vocab - 2)]
    zh_vocab = ["<pad>", "<unk>"] + [w for w, _ in zh_counter.most_common(args.zh_vocab - 2)]
    en_stoi = {w: i for i, w in enumerate(en_vocab)}
    zh_stoi = {w: i for i, w in enumerate(zh_vocab)}
    return pairs, en_stoi, zh_stoi, en_vocab, zh_vocab


def encode(tokens, stoi, max_len):
    ids = [stoi.get(t, UNK) for t in tokens[:max_len]]
    length = len(ids)
    ids += [PAD] * (max_len - len(ids))
    return ids, length


def build_arrays(args):
    pairs, en_stoi, zh_stoi, en_vocab, zh_vocab = read_pairs(args)
    en_ids, zh_ids, en_len, zh_len = [], [], [], []
    raw = []
    for en_t, zh_t, en_raw, zh_raw in pairs:
        ei, el = encode(en_t, en_stoi, args.max_len)
        zi, zl = encode(zh_t, zh_stoi, args.max_len)
        en_ids.append(ei)
        zh_ids.append(zi)
        en_len.append(el)
        zh_len.append(zl)
        raw.append({"en": en_raw, "zh": zh_raw, "en_tok": en_t[: args.max_len], "zh_tok": zh_t[: args.max_len]})
    arrays = {
        "en": np.array(en_ids, dtype=np.int64),
        "zh": np.array(zh_ids, dtype=np.int64),
        "en_len": np.array(en_len, dtype=np.int64),
        "zh_len": np.array(zh_len, dtype=np.int64),
        "raw": raw,
        "en_vocab": en_vocab,
        "zh_vocab": zh_vocab,
    }
    n = len(en_ids)
    n_train = int(n * args.train_frac)
    n_test = int(n * args.test_frac)
    idx = np.arange(n)
    split = {
        "train": idx[:n_train],
        "test": idx[n_train : n_train + n_test],
        "ood": idx[n_train + n_test :],
    }
    return arrays, split


class ComposeKernel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, left, right):
        return self.net(torch.cat([left, right, left * right, left - right], dim=-1))


class TreeHeapCanonical(nn.Module):
    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.kind = "treeheap"
        self.max_len = max_len
        self.dim = dim
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
        if lang == "en":
            leaf = self.en_leaf(self.en_emb(ids) + self.path_emb(pos))
        else:
            leaf = self.zh_leaf(self.zh_emb(ids) + self.path_emb(pos))
        cur = leaf
        while cur.shape[1] > 1:
            if cur.shape[1] % 2 == 1:
                cur = torch.cat([cur, torch.zeros_like(cur[:, -1:, :])], dim=1)
            cur = self.compose(cur[:, 0::2, :], cur[:, 1::2, :])
        mask = (ids != PAD).float().unsqueeze(-1)
        mean = (leaf * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        root = F.normalize(self.root(cur[:, 0, :]) + self.mean_proj(mean), dim=-1)
        return root, leaf

    def forward(self, en_ids, zh_ids):
        en_root, en_leaf = self.encode_side(en_ids, "en")
        zh_root, zh_leaf = self.encode_side(zh_ids, "zh")
        return en_root, zh_root, self.en_dec(en_leaf), self.zh_dec(zh_leaf)


class BoWCanonical(nn.Module):
    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.kind = "bow"
        self.max_len = max_len
        self.en_emb = nn.Embedding(en_vocab, dim, padding_idx=PAD)
        self.zh_emb = nn.Embedding(zh_vocab, dim, padding_idx=PAD)
        self.en_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.zh_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.en_dec = nn.Linear(dim, en_vocab)
        self.zh_dec = nn.Linear(dim, zh_vocab)

    @staticmethod
    def mean_pool(leaf, ids):
        mask = (ids != PAD).float().unsqueeze(-1)
        return (leaf * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, en_ids, zh_ids):
        en_leaf = self.en_emb(en_ids)
        zh_leaf = self.zh_emb(zh_ids)
        en_root = F.normalize(self.en_proj(self.mean_pool(en_leaf, en_ids)), dim=-1)
        zh_root = F.normalize(self.zh_proj(self.mean_pool(zh_leaf, zh_ids)), dim=-1)
        return en_root, zh_root, self.en_dec(en_leaf), self.zh_dec(zh_leaf)


class LSTMCanonical(nn.Module):
    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.kind = "lstm"
        self.max_len = max_len
        self.en_emb = nn.Embedding(en_vocab, dim, padding_idx=PAD)
        self.zh_emb = nn.Embedding(zh_vocab, dim, padding_idx=PAD)
        self.en_lstm = nn.LSTM(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.zh_lstm = nn.LSTM(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.en_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.zh_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.en_dec = nn.Linear(dim, en_vocab)
        self.zh_dec = nn.Linear(dim, zh_vocab)

    @staticmethod
    def mean_pool(leaf, ids):
        mask = (ids != PAD).float().unsqueeze(-1)
        return (leaf * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, en_ids, zh_ids):
        en_leaf, _ = self.en_lstm(self.en_emb(en_ids))
        zh_leaf, _ = self.zh_lstm(self.zh_emb(zh_ids))
        en_root = F.normalize(self.en_proj(self.mean_pool(en_leaf, en_ids)), dim=-1)
        zh_root = F.normalize(self.zh_proj(self.mean_pool(zh_leaf, zh_ids)), dim=-1)
        return en_root, zh_root, self.en_dec(en_leaf), self.zh_dec(zh_leaf)


class TransformerCanonical(nn.Module):
    def __init__(self, en_vocab, zh_vocab, max_len, dim):
        super().__init__()
        self.kind = "transformer"
        self.max_len = max_len
        self.en_emb = nn.Embedding(en_vocab, dim, padding_idx=PAD)
        self.zh_emb = nn.Embedding(zh_vocab, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.en_tx = nn.TransformerEncoder(enc_layer, num_layers=2)
        zh_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.zh_tx = nn.TransformerEncoder(zh_layer, num_layers=2)
        self.en_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.zh_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.en_dec = nn.Linear(dim, en_vocab)
        self.zh_dec = nn.Linear(dim, zh_vocab)

    @staticmethod
    def mean_pool(leaf, ids):
        mask = (ids != PAD).float().unsqueeze(-1)
        return (leaf * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, en_ids, zh_ids):
        pos = torch.arange(self.max_len, device=en_ids.device).unsqueeze(0)
        en_leaf = self.en_tx(self.en_emb(en_ids) + self.pos_emb(pos), src_key_padding_mask=(en_ids == PAD))
        zh_leaf = self.zh_tx(self.zh_emb(zh_ids) + self.pos_emb(pos), src_key_padding_mask=(zh_ids == PAD))
        en_root = F.normalize(self.en_proj(self.mean_pool(en_leaf, en_ids)), dim=-1)
        zh_root = F.normalize(self.zh_proj(self.mean_pool(zh_leaf, zh_ids)), dim=-1)
        return en_root, zh_root, self.en_dec(en_leaf), self.zh_dec(zh_leaf)


MODEL_CLASSES = {
    "treeheap": TreeHeapCanonical,
    "bow": BoWCanonical,
    "lstm": LSTMCanonical,
    "transformer": TransformerCanonical,
}


def alignment_loss(en_root, zh_root, temp):
    logits = en_root @ zh_root.t() / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


def echo_loss(en_logits, zh_logits, en_ids, zh_ids):
    return (
        F.cross_entropy(en_logits.reshape(-1, en_logits.shape[-1]), en_ids.reshape(-1), ignore_index=PAD)
        + F.cross_entropy(zh_logits.reshape(-1, zh_logits.shape[-1]), zh_ids.reshape(-1), ignore_index=PAD)
    ) * 0.5


def variance_loss(en_root, zh_root, target_std):
    roots = torch.cat([en_root, zh_root], dim=0)
    std = torch.sqrt(roots.var(dim=0) + 1e-4)
    return F.relu(target_std - std).mean()


def batch_iter(indices, batch, rng):
    order = indices.copy()
    rng.shuffle(order)
    for i in range(0, len(order), batch):
        yield order[i : i + batch]


def train_one(model, arrays, split, args, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    en = torch.tensor(arrays["en"], dtype=torch.long)
    zh = torch.tensor(arrays["zh"], dtype=torch.long)
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in model.kind))
    trace = []
    for epoch in range(args.epochs):
        total = {"loss": 0.0, "align": 0.0, "echo": 0.0, "n": 0}
        for sel_np in batch_iter(split["train"], args.batch, rng):
            sel = torch.tensor(sel_np, dtype=torch.long)
            en_batch = en[sel].to(device, non_blocking=True)
            zh_batch = zh[sel].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            en_root, zh_root, en_logits, zh_logits = model(en_batch, zh_batch)
            l_align = alignment_loss(en_root, zh_root, args.temperature)
            l_echo = echo_loss(en_logits, zh_logits, en_batch, zh_batch)
            l_var = variance_loss(en_root, zh_root, args.target_std)
            loss = args.align_weight * l_align + args.echo_weight * l_echo + args.var_weight * l_var
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            b = len(sel_np)
            total["loss"] += float(loss.detach().cpu()) * b
            total["align"] += float(l_align.detach().cpu()) * b
            total["echo"] += float(l_echo.detach().cpu()) * b
            total.setdefault("var", 0.0)
            total["var"] += float(l_var.detach().cpu()) * b
            total["n"] += b
        if epoch in {0, 1, 2, args.epochs - 1}:
            trace.append({k: (v / total["n"] if k != "n" else v) for k, v in total.items()} | {"epoch": epoch})
    model.cpu()
    return model, trace


def compute_roots(model, arrays, indices, args, device):
    model.to(device)
    en_all, zh_all, en_pred, zh_pred = [], [], [], []
    en = torch.tensor(arrays["en"], dtype=torch.long)
    zh = torch.tensor(arrays["zh"], dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(indices), args.eval_batch):
            sel = torch.tensor(indices[i : i + args.eval_batch], dtype=torch.long)
            en_batch = en[sel].to(device, non_blocking=True)
            zh_batch = zh[sel].to(device, non_blocking=True)
            er, zr, elog, zlog = model(en_batch, zh_batch)
            en_all.append(er.cpu())
            zh_all.append(zr.cpu())
            en_pred.append(elog.argmax(-1).cpu())
            zh_pred.append(zlog.argmax(-1).cpu())
    model.cpu()
    return torch.cat(en_all), torch.cat(zh_all), torch.cat(en_pred).numpy(), torch.cat(zh_pred).numpy()


def metric_for_split(model, arrays, indices, args, device):
    en_root, zh_root, en_pred, zh_pred = compute_roots(model, arrays, indices, args, device)
    sim = en_root @ zh_root.t()
    diag = sim.diag()
    n = sim.shape[0]
    roll = torch.arange(n)
    neg = sim[roll, (roll + 1) % n]
    dist_pos = float((1.0 - diag).mean())
    dist_neg = float((1.0 - neg).mean())
    ranks = (sim.argsort(dim=1, descending=True) == torch.arange(n).unsqueeze(1)).nonzero()[:, 1] + 1
    retrieval1 = float((ranks <= 1).float().mean())
    retrieval5 = float((ranks <= 5).float().mean())
    retrieval10 = float((ranks <= 10).float().mean())
    probs = F.softmax(sim / args.temperature, dim=1)
    entropy = float((-(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=1)).mean())
    positive_prob = float(probs.diag().mean())
    uniform_entropy = float(math.log(max(1, n)))

    en_true = arrays["en"][indices]
    zh_true = arrays["zh"][indices]
    en_mask = en_true != PAD
    zh_mask = zh_true != PAD
    en_tok = float((en_pred[en_mask] == en_true[en_mask]).mean()) if en_mask.any() else 0.0
    zh_tok = float((zh_pred[zh_mask] == zh_true[zh_mask]).mean()) if zh_mask.any() else 0.0
    en_exact = float((((en_pred == en_true) | ~en_mask).all(axis=1)).mean())
    zh_exact = float((((zh_pred == zh_true) | ~zh_mask).all(axis=1)).mean())
    return {
        "positive_distance": dist_pos,
        "negative_distance": dist_neg,
        "distance_margin_neg_minus_pos": dist_neg - dist_pos,
        "retrieval_at_1": retrieval1,
        "retrieval_at_5": retrieval5,
        "retrieval_at_10": retrieval10,
        "alignment_entropy": entropy,
        "uniform_entropy": uniform_entropy,
        "positive_pair_probability": positive_prob,
        "random_retrieval_at_1": 1.0 / max(1, n),
        "en_echo_token_acc": en_tok,
        "zh_echo_token_acc": zh_tok,
        "en_echo_exact": en_exact,
        "zh_echo_exact": zh_exact,
    }


def readable_examples(model, arrays, split_indices, args, device, count):
    en_root, zh_root, en_pred, zh_pred = compute_roots(model, arrays, split_indices[: min(len(split_indices), args.eval_batch)], args, device)
    sim = en_root @ zh_root.t()
    examples = []
    en_vocab, zh_vocab = arrays["en_vocab"], arrays["zh_vocab"]
    for local_i in range(min(count, sim.shape[0])):
        global_i = int(split_indices[local_i])
        top = int(sim[local_i].argmax())
        paired_score = float(sim[local_i, local_i])
        top_score = float(sim[local_i, top])
        en_len = int((arrays["en"][global_i] != PAD).sum())
        zh_len = int((arrays["zh"][global_i] != PAD).sum())
        ep = [en_vocab[x] for x in en_pred[local_i, :en_len].tolist()]
        zp = [zh_vocab[x] for x in zh_pred[local_i, :zh_len].tolist()]
        raw = arrays["raw"][global_i]
        examples.append(
            {
                "en": raw["en"],
                "zh": raw["zh"],
                "en_echo": detok_en(ep),
                "zh_echo": detok_zh(zp),
                "paired_cosine": paired_score,
                "top_match_rank1": top == local_i,
                "top_score": top_score,
                "top_index_in_eval_window": top,
            }
        )
    return examples


def evaluate_untrained(model_cls, arrays, split, args, device):
    torch.manual_seed(args.seed + 999)
    model = model_cls(len(arrays["en_vocab"]), len(arrays["zh_vocab"]), args.max_len, args.dim)
    return {name: metric_for_split(model, arrays, idx[: args.max_eval], args, device) for name, idx in split.items() if name != "train"}


def instantiate_model(name, arrays, args):
    return MODEL_CLASSES[name](len(arrays["en_vocab"]), len(arrays["zh_vocab"]), args.max_len, args.dim)


def run(args):
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    arrays, split = build_arrays(args)
    device = torch.device(args.device)

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [x for x in model_names if x not in MODEL_CLASSES]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; available={sorted(MODEL_CLASSES)}")
    untrained_tree = evaluate_untrained(TreeHeapCanonical, arrays, split, args, device)
    metrics = {"untrained_treeheap": untrained_tree}
    traces = {}
    params = {}
    examples = {}
    for model_name in model_names:
        model = instantiate_model(model_name, arrays, args)
        model, trace = train_one(model, arrays, split, args, device)
        traces[model_name] = trace
        params[model_name] = sum(p.numel() for p in model.parameters())
        metrics[model_name] = {name: metric_for_split(model, arrays, idx[: args.max_eval], args, device) for name, idx in split.items() if name != "train"}
        examples[f"{model_name}_ood"] = readable_examples(model, arrays, split["ood"], args, device, args.examples if model_name == "treeheap" else min(3, args.examples))

    primary = args.primary_model
    baseline = args.baseline_model
    if primary not in metrics or baseline not in metrics:
        raise ValueError(f"primary/baseline missing from trained models: primary={primary}, baseline={baseline}, models={model_names}")
    ood = metrics[primary]["ood"]
    bow_ood = metrics[baseline]["ood"]
    raw_ood = metrics["untrained_treeheap"]["ood"]
    pass_checks = {
        "positive_distance_below_negative": ood["positive_distance"] < ood["negative_distance"],
        "retrieval_beats_random": ood["retrieval_at_1"] > ood["random_retrieval_at_1"] * args.min_random_gain,
        "positive_probability_beats_random": ood["positive_pair_probability"] > ood["random_retrieval_at_1"] * args.min_prob_gain,
        "entropy_below_uniform": ood["alignment_entropy"] < ood["uniform_entropy"],
        "echo_nontrivial": min(ood["en_echo_token_acc"], ood["zh_echo_token_acc"]) >= args.min_echo_token_acc,
        "primary_beats_baseline_margin": ood["distance_margin_neg_minus_pos"] > bow_ood["distance_margin_neg_minus_pos"],
    }
    model_summaries = {
        name: {"trace": traces[name], "metrics": metrics[name], "parameters": params[name]}
        for name in model_names
    }
    model_summaries["untrained_treeheap"] = {"metrics": metrics["untrained_treeheap"]}
    return {
        "claim": "S1-CANON-WMT-C01",
        "predict": "P-S1-CANON-WMT01",
        "host": args.host_label,
        "device": args.device,
        "config": vars(args),
        "dataset": {
            "pairs": int(len(arrays["en"])),
            "train": int(len(split["train"])),
            "test": int(len(split["test"])),
            "ood": int(len(split["ood"])),
            "en_vocab": int(len(arrays["en_vocab"])),
            "zh_vocab": int(len(arrays["zh_vocab"])),
        },
        "models": model_summaries,
        "examples": examples,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported_if_pass": "WMT parallel forms can be pulled into a closer canonical TreeHeap state while preserving echo.",
            "baseline_warning": "If BoW matches or beats TreeHeap, the evidence supports canonicalization but not a TreeHeap-specific advantage.",
            "not_proved": ["translation BLEU", "semantic understanding", "unsupervised world-model grounding", "final S2 decoder"],
        },
        "elapsed_sec": time.time() - started,
    }


def write_outputs(summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for model_name, model_row in summary["models"].items():
            for row in model_row.get("trace", []):
                f.write(json.dumps({"model": model_name, **row}, ensure_ascii=False) + "\n")
    primary = summary["config"]["primary_model"]
    baseline = summary["config"]["baseline_model"]
    t = summary["models"][primary]["metrics"]["ood"]
    b = summary["models"][baseline]["metrics"]["ood"]
    u = summary["models"]["untrained_treeheap"]["metrics"]["ood"]
    model_lines = []
    for name, model_row in summary["models"].items():
        if "metrics" not in model_row or "ood" not in model_row["metrics"]:
            continue
        row = model_row["metrics"]["ood"]
        model_lines.append(
            f"{name:20s} margin={row['distance_margin_neg_minus_pos']:.6f} "
            f"ret@1={row['retrieval_at_1']:.6f} ret@5={row['retrieval_at_5']:.6f} "
            f"entropy={row['alignment_entropy']:.6f} en_echo={row['en_echo_token_acc']:.6f} "
            f"zh_echo={row['zh_echo_token_acc']:.6f}"
        )
    ex_lines = []
    for ex in summary["examples"].get(f"{primary}_ood", []):
        ex_lines.append(
            f"- paired_cos={ex['paired_cosine']:.4f}, top1={ex['top_match_rank1']}\n"
            f"  - EN: {ex['en']}\n"
            f"  - ZH: {ex['zh']}\n"
            f"  - EN echo: {ex['en_echo']}\n"
            f"  - ZH echo: {ex['zh_echo']}"
        )
    readme = f"""# S1 WMT Canonical Echo Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
{primary}_ood_positive_distance = {t['positive_distance']:.6f}
{primary}_ood_negative_distance = {t['negative_distance']:.6f}
{primary}_ood_margin            = {t['distance_margin_neg_minus_pos']:.6f}
{primary}_ood_retrieval@1       = {t['retrieval_at_1']:.6f}
{primary}_ood_retrieval@5       = {t['retrieval_at_5']:.6f}
{primary}_ood_entropy           = {t['alignment_entropy']:.6f}
{primary}_ood_en_echo_token     = {t['en_echo_token_acc']:.6f}
{primary}_ood_zh_echo_token     = {t['zh_echo_token_acc']:.6f}

{baseline}_ood_margin           = {b['distance_margin_neg_minus_pos']:.6f}
{baseline}_ood_retrieval@1      = {b['retrieval_at_1']:.6f}
untrained_treeheap_entropy     = {u['alignment_entropy']:.6f}
```

## OOD Model Table

```text
{chr(10).join(model_lines)}
```

## Readable {primary} OOD Examples

{chr(10).join(ex_lines)}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_wmt_canonical_echo_probe")
    parser.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--scan-lines", type=int, default=500000)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--models", default="treeheap,bow")
    parser.add_argument("--primary-model", default="treeheap")
    parser.add_argument("--baseline-model", default="bow")
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--en-vocab", type=int, default=8192)
    parser.add_argument("--zh-vocab", type=int, default=8192)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--max-eval", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--echo-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--target-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-label", default="unknown")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--min-random-gain", type=float, default=10.0)
    parser.add_argument("--min-prob-gain", type=float, default=2.0)
    parser.add_argument("--min-echo-token-acc", type=float, default=0.55)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
