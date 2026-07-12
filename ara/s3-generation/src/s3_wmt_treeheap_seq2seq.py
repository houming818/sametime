#!/usr/bin/env python3
"""Minimal real-data WMT English -> Chinese TreeHeap seq2seq proof."""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    data: str; spm_model: str; evidence_dir: str; model: str; seed: int
    train_samples: int; valid_samples: int; test_samples: int; max_scan: int
    min_len: int; max_len: int; dim: int; hidden: int; batch_size: int
    epochs: int; lr: float; device: str; num_workers: int


class ParallelDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[List[int], List[int]]]): self.rows = rows
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, i: int) -> Tuple[List[int], List[int]]: return self.rows[i]


def load_rows(cfg: Config, sp: spm.SentencePieceProcessor) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    rows: List[Tuple[List[int], List[int]]] = []
    required = cfg.train_samples + cfg.valid_samples + cfg.test_samples
    with open(cfg.data, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= cfg.max_scan or len(rows) >= required: break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            src, tgt = sp.encode(parts[0], out_type=int), sp.encode(parts[1], out_type=int)
            if cfg.min_len <= len(src) <= cfg.max_len and cfg.min_len <= len(tgt) <= cfg.max_len:
                rows.append((src + [sp.eos_id()], tgt + [sp.eos_id()]))
    if len(rows) < required:
        raise RuntimeError(f"only {len(rows)} usable rows; need {required}; raise --max-scan or lower samples")
    return rows, sp.get_piece_size()


def collate(pad: int):
    def fn(batch: Sequence[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_len = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
        tgt_len = torch.tensor([len(y) for _, y in batch], dtype=torch.long)
        src = torch.full((len(batch), int(src_len.max())), pad, dtype=torch.long)
        tgt = torch.full((len(batch), int(tgt_len.max())), pad, dtype=torch.long)
        for i, (x, y) in enumerate(batch): src[i, :len(x)], tgt[i, :len(y)] = torch.tensor(x), torch.tensor(y)
        return src, src_len, tgt, tgt_len
    return fn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)


class Decoder(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__()
        self.emb, self.q = nn.Embedding(vocab, dim), nn.Linear(hidden, dim, bias=False)
        self.cell = nn.GRUCell(dim * 2, hidden)
        self.out = nn.Linear(hidden + dim, vocab)
        self.hidden = hidden

    def step(self, prev: torch.Tensor, state: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = (self.q(state)[:, None, :] * memory).sum(-1) / math.sqrt(memory.shape[-1])
        attn = masked_softmax(score, mask)
        context = (attn[:, :, None] * memory).sum(1)
        state = self.cell(torch.cat([self.emb(prev), context], -1), state)
        return self.out(torch.cat([state, context], -1)), state

    def teacher(self, memory: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, bos: int) -> torch.Tensor:
        state = memory.new_zeros((memory.shape[0], self.hidden)); prev = torch.full((memory.shape[0],), bos, device=memory.device, dtype=torch.long)
        out = []
        for t in range(target.shape[1]):
            logits, state = self.step(prev, state, memory, mask); out.append(logits); prev = target[:, t]
        return torch.stack(out, 1)

    def greedy(self, memory: torch.Tensor, mask: torch.Tensor, bos: int, eos: int, max_len: int) -> torch.Tensor:
        state = memory.new_zeros((memory.shape[0], self.hidden)); prev = torch.full((memory.shape[0],), bos, device=memory.device, dtype=torch.long); out = []
        done = torch.zeros(memory.shape[0], device=memory.device, dtype=torch.bool)
        for _ in range(max_len):
            logits, state = self.step(prev, state, memory, mask); prev = logits.argmax(-1); out.append(prev); done |= prev.eq(eos)
            if bool(done.all()): break
        return torch.stack(out, 1) if out else prev[:, None]


class Seq2SeqBase(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int): super().__init__(); self.decoder = Decoder(vocab, dim, hidden)
    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError
    def forward(self, src: torch.Tensor, length: torch.Tensor, target: torch.Tensor, bos: int) -> torch.Tensor:
        memory, mask = self.encode(src, length); return self.decoder.teacher(memory, mask, target, bos)
    def generate(self, src: torch.Tensor, length: torch.Tensor, bos: int, eos: int, max_len: int, mode: str = "full") -> torch.Tensor:
        memory, mask = self.encode(src, length, mode); return self.decoder.greedy(memory, mask, bos, eos, max_len)


class BowSeq2Seq(Seq2SeqBase):
    def __init__(self, vocab: int, dim: int, hidden: int): super().__init__(vocab, dim, hidden); self.emb = nn.Embedding(vocab, dim)
    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.arange(src.shape[1], device=src.device)[None] < length[:, None]
        h = (self.emb(src) * mask[:, :, None]).sum(1) / length[:, None].clamp_min(1)
        return h[:, None], torch.ones((src.shape[0], 1), device=src.device, dtype=torch.bool)


class FlatSeq2Seq(Seq2SeqBase):
    def __init__(self, vocab: int, dim: int, hidden: int): super().__init__(vocab, dim, hidden); self.emb = nn.Embedding(vocab, dim); self.rnn = nn.GRU(dim, dim, batch_first=True)
    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), length.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed); out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])
        mask = torch.arange(src.shape[1], device=src.device)[None] < length[:, None]
        return out, mask


class TreeHeapSeq2Seq(Seq2SeqBase):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__(vocab, dim, hidden); self.emb = nn.Embedding(vocab, dim); self.left = nn.Parameter(torch.randn(dim) * .02); self.right = nn.Parameter(torch.randn(dim) * .02)
        self.compose = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Tanh())
    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.arange(src.shape[1], device=src.device)[None] < length[:, None]; node, node_mask = self.emb(src), mask; levels = [(node, node_mask)]
        while node.shape[1] > 1:
            if node.shape[1] % 2: node, node_mask = torch.cat([node, torch.zeros_like(node[:, :1])], 1), torch.cat([node_mask, torch.zeros_like(node_mask[:, :1])], 1)
            left, right, lm, rm = node[:, 0::2], node[:, 1::2], node_mask[:, 0::2], node_mask[:, 1::2]
            merged = self.compose(torch.cat([left + self.left, right + self.right], -1)); node = torch.where((lm & rm)[:, :, None], merged, torch.where(lm[:, :, None], left, right)); node_mask = lm | rm; levels.append((node, node_mask))
        if mode == "root_only": return levels[-1]
        if mode == "leaf_only": return levels[0]
        return torch.cat([x for x, _ in levels], 1), torch.cat([m for _, m in levels], 1)


MODELS = {"treeheap": TreeHeapSeq2Seq, "bow": BowSeq2Seq, "flat_seq": FlatSeq2Seq}


def ce(logits: torch.Tensor, target: torch.Tensor, pad: int) -> torch.Tensor: return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad)
def clean(ids: Sequence[int], eos: int, pad: int) -> List[int]:
    out = []
    for x in ids:
        if x == eos or x == pad: break
        out.append(int(x))
    return out
def bleu4(hyps: Sequence[List[int]], refs: Sequence[List[int]]) -> float:
    precisions = []
    for n in range(1, 5):
        hit = total = 0
        for hyp, ref in zip(hyps, refs):
            hc: Dict[Tuple[int, ...], int] = {}; rc: Dict[Tuple[int, ...], int] = {}
            for i in range(max(0, len(hyp)-n+1)): hc[tuple(hyp[i:i+n])] = hc.get(tuple(hyp[i:i+n]), 0) + 1
            for i in range(max(0, len(ref)-n+1)): rc[tuple(ref[i:i+n])] = rc.get(tuple(ref[i:i+n]), 0) + 1
            total += sum(hc.values()); hit += sum(min(v, rc.get(k, 0)) for k, v in hc.items())
        precisions.append((hit + 1) / (total + 1))
    hyp_len, ref_len = sum(map(len, hyps)), sum(map(len, refs)); bp = math.exp(min(0.0, 1 - ref_len / max(1, hyp_len)))
    return 100 * bp * math.exp(sum(math.log(p) for p in precisions) / 4)


def evaluate(model: Seq2SeqBase, loader: DataLoader, cfg: Config, pad: int, bos: int, eos: int, sp: spm.SentencePieceProcessor, mode: str = "full") -> Dict[str, object]:
    model.eval(); loss_sum = tokens = exact = 0; hyps: List[List[int]] = []; refs: List[List[int]] = []; examples = []
    with torch.no_grad():
        for src, length, target, tgt_len in loader:
            src, length, target = src.to(cfg.device), length.to(cfg.device), target.to(cfg.device)
            memory, memory_mask = model.encode(src, length, mode)
            logits = model.decoder.teacher(memory, memory_mask, target, bos); valid = target.ne(pad); loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad, reduction="sum").item()); tokens += int(valid.sum().item())
            pred = model.decoder.greedy(memory, memory_mask, bos, eos, target.shape[1]).cpu(); src_cpu, target_cpu = src.cpu(), target.cpu()
            for i in range(src.shape[0]):
                h, r = clean(pred[i].tolist(), eos, pad), clean(target_cpu[i].tolist(), eos, pad); hyps.append(h); refs.append(r); exact += int(h == r)
                if len(examples) < 12: examples.append({"en": sp.decode(clean(src_cpu[i].tolist(), eos, pad)), "reference_zh": sp.decode(r), "hypothesis_zh": sp.decode(h)})
    return {"nll": loss_sum / max(1, tokens), "ppl": math.exp(min(20, loss_sum / max(1, tokens))), "exact": exact / max(1, len(refs)), "token_bleu4": bleu4(hyps, refs), "examples": examples}


def train(name: str, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, cfg: Config, vocab: int, pad: int, bos: int, eos: int, sp: spm.SentencePieceProcessor, checkpoint: Path) -> Dict[str, object]:
    torch.manual_seed(cfg.seed + sum(map(ord, name))); model = MODELS[name](vocab, cfg.dim, cfg.hidden).to(cfg.device); opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr); trace = []; best = None; best_nll = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train(); total = count = 0
        for src, length, target, _ in train_loader:
            src, length, target = src.to(cfg.device), length.to(cfg.device), target.to(cfg.device); logits = model(src, length, target, bos); loss = ce(logits, target, pad); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss.item()); count += 1
        valid = evaluate(model, valid_loader, cfg, pad, bos, eos, sp); trace.append({"epoch": epoch, "train_nll": total/max(1,count), **{k:v for k,v in valid.items() if k != "examples"}}); print(f"[{name}] epoch={epoch} train_nll={total/max(1,count):.4f} valid_nll={valid['nll']:.4f} bleu={valid['token_bleu4']:.2f}", flush=True)
        if valid["nll"] < best_nll: best_nll, best = float(valid["nll"]), {k:v.cpu() for k,v in model.state_dict().items()}
    assert best is not None; model.load_state_dict(best); torch.save({"model": name, "config": asdict(cfg), "state_dict": best}, checkpoint); result = {"parameters": sum(p.numel() for p in model.parameters()), "checkpoint": checkpoint.name, "trace": trace, "test": evaluate(model, test_loader, cfg, pad, bos, eos, sp)}
    if name == "treeheap": result["test_leaf_only"] = evaluate(model, test_loader, cfg, pad, bos, eos, sp, "leaf_only"); result["test_root_only"] = evaluate(model, test_loader, cfg, pad, bos, eos, sp, "root_only")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en"); ap.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model"); ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_wmt_treeheap_seq2seq"); ap.add_argument("--model", default="all", choices=["all", *MODELS]); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--train-samples", type=int, default=30000); ap.add_argument("--valid-samples", type=int, default=2000); ap.add_argument("--test-samples", type=int, default=2000); ap.add_argument("--max-scan", type=int, default=120000); ap.add_argument("--min-len", type=int, default=3); ap.add_argument("--max-len", type=int, default=24); ap.add_argument("--dim", type=int, default=256); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--num-workers", type=int, default=2); args = ap.parse_args(); cfg = Config(**vars(args))
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model); rows, pieces = load_rows(cfg, sp); pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1; splits = [rows[:cfg.train_samples], rows[cfg.train_samples:cfg.train_samples+cfg.valid_samples], rows[cfg.train_samples+cfg.valid_samples:]]; loaders = [DataLoader(ParallelDataset(x), batch_size=cfg.batch_size, shuffle=i==0, num_workers=cfg.num_workers, collate_fn=collate(pad), pin_memory=cfg.device.startswith("cuda")) for i,x in enumerate(splits)]
    started=time.time(); names=list(MODELS) if cfg.model=="all" else [cfg.model]; out=Path(cfg.evidence_dir); out.mkdir(parents=True,exist_ok=True); results={name:train(name,*loaders,cfg,vocab,pad,bos,eos,sp,out/f"checkpoint_best_{name}.pt") for name in names}; summary={"claim":"S3-WMT-SEQ2SEQ-C01","host":socket.gethostname(),"seconds":time.time()-started,"config":asdict(cfg),"data":{"direction":"en_to_zh","rows":len(rows),"vocab":vocab,"pad":pad},"models":results}; (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); (out/"examples.json").write_text(json.dumps({n:x["test"]["examples"] for n,x in results.items()},indent=2,ensure_ascii=False),encoding="utf-8"); (out/"trace.jsonl").write_text("\n".join(json.dumps({"model":n,**r},ensure_ascii=False) for n,x in results.items() for r in x["trace"])+"\n",encoding="utf-8"); (out/"README.md").write_text("# Real WMT TreeHeap Seq2Seq\n\n```json\n"+json.dumps({n:{"parameters":x["parameters"],"test":x["test"]} for n,x in results.items()},indent=2,ensure_ascii=False)+"\n```\n",encoding="utf-8"); print(json.dumps({n:x["test"] for n,x in results.items()},indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
