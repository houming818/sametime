#!/usr/bin/env python3
"""Full-corpus residual TreeHeap forest pretraining proof.

The script has two durable phases:

  prepare: tokenize raw Chinese documents once into fixed NAS shards;
  train:   compare residual and matched no-residual TreeHeap forests.

The decoder sees only the recursively folded root. There is no leaf bypass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F


SOURCE_WEIGHTS = {"news": 0.50, "wiki": 0.30, "web": 0.20}


def source_files(root: Path, split: str) -> Dict[str, List[Path]]:
    base = root / "Chinese-Train-Datasets"
    if split == "train":
        rows = {
            "news": [base / "new2016zh" / "news2016zh_train.json"],
            "wiki": sorted((base / "wiki_zh").rglob("wiki_*")),
            "web": [base / "webtext2019zh" / "webtext_zh_train.json"],
        }
    else:
        rows = {
            "news": [base / "new2016zh" / "news2016zh_valid.json"],
            "wiki": [],
            "web": [base / "webtext2019zh" / "webtext_zh_valid.json"],
        }
    return {name: [p for p in paths if p.is_file()] for name, paths in rows.items()}


def row_text(row: dict, source: str) -> str:
    if source == "wiki":
        return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
    return (str(row.get("title", "")) + "\n" + str(row.get("content", ""))).strip()


def document_batches(paths: Dict[str, List[Path]], batch_docs: int) -> Iterator[Tuple[str, List[str]]]:
    for source in ("news", "wiki", "web"):
        pending: List[str] = []
        for path in paths[source]:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        text = row_text(json.loads(line), source)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if len(text) < 16:
                        continue
                    pending.append(text)
                    if len(pending) >= batch_docs:
                        yield source, pending
                        pending = []
        if pending:
            yield source, pending


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    out = Path(args.block_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    if sp.get_piece_size() >= 65535:
        raise ValueError("uint16 block format requires vocab < 65535")
    width = args.context + 1
    shard_rows: List[np.ndarray] = []
    shards: List[dict] = []
    source_blocks = {name: 0 for name in SOURCE_WEIGHTS}
    source_docs = {name: 0 for name in SOURCE_WEIGHTS}
    started = time.time()

    def flush() -> None:
        if not shard_rows:
            return
        idx = len(shards)
        path = out / f"{args.split}-{idx:05d}.npy"
        arr = np.stack(shard_rows).astype(np.uint16, copy=False)
        np.save(path, arr, allow_pickle=False)
        shards.append({"path": path.name, "rows": int(arr.shape[0]), "sha256": sha256(path)})
        shard_rows.clear()
        print(f"[prepare] shard={idx} rows={arr.shape[0]} total={sum(x['rows'] for x in shards)}", flush=True)

    paths = source_files(Path(args.root), args.split)
    for source, docs in document_batches(paths, args.tokenize_batch_docs):
        encoded = sp.encode(docs, out_type=int, num_threads=args.tokenize_threads)
        source_docs[source] += len(docs)
        for ids in encoded:
            ids.append(sp.eos_id())
            usable = len(ids) - (len(ids) % width)
            for start in range(0, usable, width):
                shard_rows.append(np.asarray(ids[start : start + width], dtype=np.uint16))
                source_blocks[source] += 1
                if len(shard_rows) >= args.shard_rows:
                    flush()
        if args.max_blocks and sum(source_blocks.values()) >= args.max_blocks:
            break
    flush()
    manifest = {
        "format": "npy uint16 [rows, context+1]",
        "split": args.split,
        "context": args.context,
        "tokenizer": {"path": args.spm_model, "sha256": sha256(Path(args.spm_model)), "vocab": sp.get_piece_size()},
        "source_files": {
            name: [{"path": str(p), "bytes": p.stat().st_size} for p in rows]
            for name, rows in paths.items()
        },
        "source_docs": source_docs,
        "source_blocks": source_blocks,
        "total_blocks": sum(x["rows"] for x in shards),
        "shards": shards,
        "complete_source_pass": args.max_blocks == 0,
        "elapsed_sec": time.time() - started,
    }
    (out / f"manifest-{args.split}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("split", "source_docs", "source_blocks", "total_blocks", "complete_source_pass", "elapsed_sec")}, indent=2), flush=True)


class ParameterTreeHeapKernel(nn.Module):
    """A bank of shared three-slot [root,left,right] parameter TreeHeaps."""

    def __init__(self, dim: int, rank: int, heads: int):
        super().__init__()
        self.heads = heads
        self.down = nn.Parameter(torch.empty(heads, 3, dim, rank))
        self.bias = nn.Parameter(torch.zeros(heads, rank))
        self.up = nn.Parameter(torch.empty(heads, rank, dim))
        self.gate = nn.Linear(3 * dim, heads)
        nn.init.xavier_uniform_(self.down.flatten(0, 1))
        nn.init.xavier_uniform_(self.up)

    def forward(self, state: torch.Tensor, left: torch.Tensor, right: torch.Tensor, ablate: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = torch.stack((state, left, right), dim=2)
        hidden = torch.einsum("bnsd,hsdr->bnhr", slots, self.down) + self.bias[None, None]
        head_out = torch.einsum("bnhr,hrd->bnhd", F.gelu(hidden), self.up)
        gate = F.softmax(self.gate(torch.cat((state, left, right), dim=-1)), dim=-1)
        if 0 <= ablate < self.heads:
            keep = torch.ones_like(gate)
            keep[..., ablate] = 0
            gate = gate * keep
            gate = gate / gate.sum(-1, keepdim=True).clamp_min(1e-8)
        return (gate[..., None] * head_out).sum(2), gate, head_out


class TreeHeapForestLM(nn.Module):
    def __init__(self, vocab: int, dim: int, rank: int, heads: int, refinements: int, residual: bool):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.refinements = refinements
        self.residual = residual
        self.embedding = nn.Embedding(vocab, dim)
        self.kernel = ParameterTreeHeapKernel(dim, rank, heads)
        self.norm = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab)
        self.residual_scale = nn.Parameter(torch.tensor(0.2))

    def encode(self, tokens: torch.Tensor, destroy_addresses: bool = False, ablate: int = -1, audit: bool = False):
        node = self.embedding(tokens)
        gate_sum = node.new_zeros(self.kernel.heads)
        gate_count = 0
        head_cos_sum = node.new_tensor(0.0)
        head_cos_count = 0
        while node.shape[1] > 1:
            if node.shape[1] % 2:
                node = torch.cat((node, torch.zeros_like(node[:, :1])), dim=1)
            left, right = node[:, 0::2], node[:, 1::2]
            if destroy_addresses:
                right = right.flip(1)
            state = (left + right) * math.sqrt(0.5)
            for _ in range(self.refinements):
                delta, gate, head_out = self.kernel(state, left, right, ablate)
                state = self.norm(state + self.residual_scale * delta) if self.residual else self.norm(delta)
                if audit:
                    gate_sum += gate.detach().sum((0, 1))
                    gate_count += gate.shape[0] * gate.shape[1]
                    if self.kernel.heads > 1:
                        normed = F.normalize(head_out.detach(), dim=-1)
                        cos = torch.einsum("bnhd,bnkd->bnhk", normed, normed)
                        mask = ~torch.eye(self.kernel.heads, device=cos.device, dtype=torch.bool)
                        head_cos_sum += cos[..., mask].sum()
                        head_cos_count += cos[..., mask].numel()
            node = state
        root = node[:, 0]
        extras = {
            "gate_mean": gate_sum / max(1, gate_count),
            "head_pair_cosine": head_cos_sum / max(1, head_cos_count),
        }
        return root, extras

    def forward(self, tokens: torch.Tensor, **kwargs):
        root, extras = self.encode(tokens, **kwargs)
        return self.decoder(root), root, extras


@dataclass
class TrainConfig:
    block_dir: str
    evidence_dir: str
    context: int
    dim: int
    rank: int
    heads: int
    refinements: int
    batch: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    max_train_blocks: int
    max_valid_blocks: int
    log_every: int
    valid_every: int
    checkpoint_every: int
    device: str


def manifest(block_dir: Path, split: str) -> dict:
    return json.loads((block_dir / f"manifest-{split}.json").read_text(encoding="utf-8"))


def iter_batches(block_dir: Path, man: dict, batch: int, seed: int, max_blocks: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    shard_rows = list(man["shards"])
    rng.shuffle(shard_rows)
    seen = 0
    for row in shard_rows:
        arr = np.load(block_dir / row["path"], mmap_mode="r")
        order = np.arange(arr.shape[0])
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            if max_blocks:
                idx = idx[: max(0, max_blocks - seen)]
            if len(idx) == 0:
                return
            block = np.asarray(arr[idx], dtype=np.int64)
            yield torch.from_numpy(block[:, :-1]), torch.from_numpy(block[:, -1])
            seen += len(idx)
            if max_blocks and seen >= max_blocks:
                return


def model_metrics(model: TreeHeapForestLM, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], device: torch.device, ablate: int = -1, destroy: bool = False) -> dict:
    model.eval()
    total_loss = total = top1 = top5 = 0
    roots: List[torch.Tensor] = []
    gate = torch.zeros(model.kernel.heads, device=device)
    head_cos = 0.0
    with torch.no_grad():
        for tokens, target in batches:
            tokens, target = tokens.to(device), target.to(device)
            logits, root, extras = model(tokens, destroy_addresses=destroy, ablate=ablate, audit=True)
            total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
            total += target.numel()
            top1 += int(logits.argmax(-1).eq(target).sum())
            top5 += int(logits.topk(5, dim=-1).indices.eq(target[:, None]).any(-1).sum())
            roots.append(root.float().cpu())
            gate += extras["gate_mean"]
            head_cos += float(extras["head_pair_cosine"])
    root = torch.cat(roots)
    nll = total_loss / max(1, total)
    gate = gate / max(1, len(batches))
    return {
        "nll": nll,
        "ppl": math.exp(min(20, nll)),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "root_variance": float(root.var(0, unbiased=False).mean()),
        "root_mean_pair_cosine": float((F.normalize(root, dim=-1) @ F.normalize(root, dim=-1).T).mean()),
        "gate_mean": [float(x) for x in gate.cpu()],
        "gate_entropy": float(-(gate.clamp_min(1e-9) * gate.clamp_min(1e-9).log()).sum().cpu()),
        "head_pair_cosine": head_cos / max(1, len(batches)),
    }


def gradient_audit(model: TreeHeapForestLM, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> dict:
    model.train()
    tokens, target = (x.to(device) for x in batch)
    model.zero_grad(set_to_none=True)
    logits, _, _ = model(tokens)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "embedding_grad_norm": float(model.embedding.weight.grad.norm()),
        "kernel_down_grad_norm": float(model.kernel.down.grad.norm()),
        "kernel_up_grad_norm": float(model.kernel.up.grad.norm()),
        "decoder_grad_norm": float(model.decoder.weight.grad.norm()),
    }


def save_checkpoint(path: Path, models: Dict[str, TreeHeapForestLM], opts: Dict[str, torch.optim.Optimizer], step: int, epoch: int, cfg: TrainConfig) -> None:
    torch.save({
        "step": step,
        "epoch": epoch,
        "config": asdict(cfg),
        "models": {k: v.state_dict() for k, v in models.items()},
        "optimizers": {k: v.state_dict() for k, v in opts.items()},
    }, path)


def train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__annotations__})
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    block_dir, out = Path(cfg.block_dir), Path(cfg.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.jsonl").unlink(missing_ok=True)
    train_man, valid_man = manifest(block_dir, "train"), manifest(block_dir, "valid")
    vocab = int(train_man["tokenizer"]["vocab"])
    if train_man["context"] != cfg.context or valid_man["context"] != cfg.context:
        raise ValueError("manifest context does not match --context")
    models = {
        "residual_forest": TreeHeapForestLM(vocab, cfg.dim, cfg.rank, cfg.heads, cfg.refinements, True).to(device),
        "noresidual_forest": TreeHeapForestLM(vocab, cfg.dim, cfg.rank, cfg.heads, cfg.refinements, False).to(device),
        "single_residual": TreeHeapForestLM(vocab, cfg.dim, cfg.rank, 1, cfg.refinements, True).to(device),
    }
    opts = {name: torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for name, model in models.items()}
    valid_batches = list(iter_batches(block_dir, valid_man, cfg.batch, cfg.seed + 9000, cfg.max_valid_blocks))
    trace: List[dict] = []
    step = 0
    started = time.time()
    amp = device.type == "cuda"
    for epoch in range(1, cfg.epochs + 1):
        for tokens, target in iter_batches(block_dir, train_man, cfg.batch, cfg.seed + epoch, cfg.max_train_blocks):
            tokens, target = tokens.to(device, non_blocking=True), target.to(device, non_blocking=True)
            row = {"step": step + 1, "epoch": epoch}
            for name, model in models.items():
                model.train()
                opt = opts[name]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits, _, _ = model(tokens)
                    loss = F.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                row[f"{name}_train_nll"] = float(loss.detach())
            step += 1
            if step % cfg.log_every == 0:
                row["elapsed_sec"] = time.time() - started
                trace.append(row)
                print(json.dumps(row), flush=True)
                with (out / "trace.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            if step % cfg.valid_every == 0:
                for name, model in models.items():
                    ev = model_metrics(model, valid_batches, device)
                    print(f"[valid] step={step} model={name} nll={ev['nll']:.5f} top5={ev['top5']:.4f}", flush=True)
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(out / "checkpoint_latest.pt", models, opts, step, epoch, cfg)
        save_checkpoint(out / "checkpoint_latest.pt", models, opts, step, epoch, cfg)

    results: Dict[str, dict] = {}
    for name, model in models.items():
        base = model_metrics(model, valid_batches, device)
        destroyed = model_metrics(model, valid_batches, device, destroy=True)
        ablations = [model_metrics(model, valid_batches, device, ablate=h) for h in range(model.kernel.heads)] if model.kernel.heads > 1 else []
        grad = gradient_audit(model, valid_batches[0], device)
        results[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "valid": base,
            "destroy_addresses": destroyed,
            "address_destroy_nll_increase": destroyed["nll"] - base["nll"],
            "head_ablations": ablations,
            "head_ablation_nll_increase": [x["nll"] - base["nll"] for x in ablations],
            "gradient_audit": grad,
        }
    residual, nores = results["residual_forest"], results["noresidual_forest"]
    derived = {
        "residual_minus_nores_parameter_count": residual["parameters"] - nores["parameters"],
        "nores_minus_residual_valid_nll": nores["valid"]["nll"] - residual["valid"]["nll"],
        "residual_address_destroy_nll_increase": residual["address_destroy_nll_increase"],
        "residual_max_head_ablation_nll_increase": max(residual["head_ablation_nll_increase"], default=0.0),
    }
    gates = {
        "same_parameter_budget": derived["residual_minus_nores_parameter_count"] == 0,
        "residual_nll_gain": derived["nores_minus_residual_valid_nll"] >= 0.02,
        "address_causal": derived["residual_address_destroy_nll_increase"] >= 0.02,
        "head_causal": derived["residual_max_head_ablation_nll_increase"] >= 0.01,
        "root_not_collapsed": residual["valid"]["root_variance"] > 1e-4,
    }
    summary = {
        "claim": "S3-RESIDUAL-FOREST-C01",
        "predict": "P-S3-RESIDUAL-FOREST-01",
        "host": socket.gethostname(),
        "device": str(device),
        "config": asdict(cfg),
        "data": {"train_manifest": train_man, "valid_manifest": valid_man},
        "models": results,
        "derived": derived,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "elapsed_sec": time.time() - started,
        "boundary": "Real-corpus next-token root-bottleneck proof; not consciousness, WMT superiority, or unsupervised semantic ontology evidence.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        "# Residual TreeHeap Forest Pretraining\n\n"
        f"Claim: `S3-RESIDUAL-FOREST-C01`\n\nPilot pass: `{summary['pilot_pass']}`\n\n"
        "See `summary.json`, `trace.jsonl`, and `checkpoint_latest.pt`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": derived, "gates": gates, "pilot_pass": summary["pilot_pass"]}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train"))
    ap.add_argument("--root", default="/home/nio/datasets/pretrain")
    ap.add_argument("--spm-model", default="ara/s3-generation/evidence/s3_p0_world_observation_tokenizer/p0_zh_16000.model")
    ap.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/blocks64")
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_residual_treeheap_forest")
    ap.add_argument("--split", choices=("train", "valid"), default="train")
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--tokenize-batch-docs", type=int, default=256)
    ap.add_argument("--tokenize-threads", type=int, default=8)
    ap.add_argument("--shard-rows", type=int, default=250000)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--refinements", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=71301)
    ap.add_argument("--max-train-blocks", type=int, default=0)
    ap.add_argument("--max-valid-blocks", type=int, default=8192)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--valid-every", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
