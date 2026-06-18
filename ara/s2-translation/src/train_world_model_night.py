"""
train_world_model_night.py - single-GPU overnight TreeHeap world-model training.

This job is designed for io.grepcode.cn's fragile, power-limited RTX 3090:

- one process, one GPU
- no power/clock changes
- streaming dataset reads
- incremental local evidence
- checkpoint and evidence mirror to NAS

Training objective
------------------
For BPE token pairs inside a local window, compute:

    state(center) = t_merge(CMul(L0(center), WorldPath(center)))

Then train state(center) to retrieve the true context token among in-batch
negatives. This is not a final language model; it is a fresh checkpoint for
P-FRAME01 and follow-up geometry probes.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_DATA = "/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv"
DEFAULT_BPE = "/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model"
DEFAULT_LOCAL_ROOT = "/data/homecicd/sametime/ara/s2-translation/evidence/world_model_night"
DEFAULT_NAS_ROOT = "/mnt/nas/datasets/wmt_massive/evidence_nio/world_model_night"


PROBES = [
    ("football", "ball", ["foot", "kick", "field", "goal", "soccer"], ["hand", "basket", "racket", "engine", "snow"]),
    ("basketball", "ball", ["hand", "throw", "court", "basket", "hoop"], ["foot", "kick", "bat", "engine", "snow"]),
    ("baseball", "ball", ["bat", "glove", "pitch", "base", "field"], ["foot", "basket", "racket", "engine", "snow"]),
    ("tennis", "ball", ["racket", "court", "net", "serve", "match"], ["foot", "basket", "bat", "engine", "snow"]),
    ("snowball", "ball", ["snow", "cold", "winter", "ice", "throw"], ["foot", "basket", "bat", "engine", "racket"]),
    ("fireball", "ball", ["fire", "hot", "flame", "burn", "magic"], ["foot", "basket", "bat", "ice", "racket"]),
    ("wheelchair", "chair", ["wheel", "move", "roll", "patient", "hospital"], ["foot", "basket", "bat", "snow", "racket"]),
    ("sailboat", "boat", ["sail", "wind", "water", "sea", "harbor"], ["wheel", "engine", "basket", "snow", "bat"]),
    ("motorboat", "boat", ["motor", "engine", "fuel", "water", "speed"], ["sail", "wind", "basket", "snow", "bat"]),
    ("racecar", "car", ["race", "speed", "engine", "track", "wheel"], ["sail", "basket", "snow", "bat", "racket"]),
    ("teacup", "cup", ["tea", "drink", "hot", "kettle", "saucer"], ["football", "engine", "snow", "bat", "racket"]),
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def auc_pairwise(pos, neg):
    total = 0
    good = 0.0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                good += 1.0
            elif p == n:
                good += 0.5
    return good / max(total, 1)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class TreeHeapWorldModel(nn.Module):
    def __init__(self, vocab=32000, dim=128, depth=5):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.depth = depth
        self.L0 = nn.Embedding(vocab, dim)
        self.context = nn.Embedding(vocab, dim)
        self.t_nodes = nn.ModuleList([nn.Embedding(2**level, dim) for level in range(depth)])
        self.t_merge = nn.Linear(dim, dim)
        nn.init.normal_(self.L0.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context.weight, mean=0.0, std=0.02)
        for emb in self.t_nodes:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        nn.init.eye_(self.t_merge.weight)
        nn.init.zeros_(self.t_merge.bias)

    def path(self, ids):
        w = torch.zeros((ids.numel(), self.dim), device=ids.device)
        for level, emb in enumerate(self.t_nodes):
            stride = max(self.vocab // (2**level), 1)
            if level == 0:
                node_idx = torch.zeros_like(ids)
            else:
                node_idx = torch.clamp(ids // stride, 0, (2**level) - 1)
            w = w + emb(node_idx)
        return F.normalize(w, dim=-1)

    def cmul_pre(self, ids):
        t = F.normalize(self.L0(ids), dim=-1)
        w = self.path(ids)
        left = t[:, :64] * w[:, :64] - t[:, 64:] * w[:, 64:]
        right = t[:, :64] * w[:, 64:] + t[:, 64:] * w[:, :64]
        return torch.cat([left, right], dim=-1)

    def state(self, ids, mode="tree"):
        if mode == "l0":
            return F.normalize(self.L0(ids), dim=-1)
        if mode == "path":
            return self.path(ids)
        if mode == "cmul":
            return self.cmul_pre(ids)
        if mode == "merge_no_bias":
            return self.cmul_pre(ids) @ self.t_merge.weight.T
        if mode == "tree":
            return self.t_merge(self.cmul_pre(ids))
        raise ValueError(mode)

    def forward(self, center, context):
        s = F.normalize(self.state(center, "tree"), dim=-1)
        c = F.normalize(self.context(context), dim=-1)
        return s @ c.T

    def checkpoint_dict(self):
        return {
            "L0": {"weight": self.L0.weight.detach().cpu()},
            "context": {"weight": self.context.weight.detach().cpu()},
            "t_nodes": [{"weight": emb.weight.detach().cpu()} for emb in self.t_nodes],
            "t_merge": {
                "weight": self.t_merge.weight.detach().cpu(),
                "bias": self.t_merge.bias.detach().cpu(),
            },
            "meta": {"vocab": self.vocab, "dim": self.dim, "depth": self.depth, "created_at": now()},
        }


def encode_line(sp, line, max_len):
    parts = line.rstrip("\n").split("\t")
    text = parts[1] if len(parts) >= 2 else parts[0]
    ids = sp.encode(text.lower())[:max_len]
    return [int(i) for i in ids if 0 <= int(i) < 32000]


def pair_stream(data_path, sp, window, max_len, seed, max_lines):
    rng = random.Random(seed)
    while True:
        with open(data_path, encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f):
                if max_lines and line_no >= max_lines:
                    break
                ids = encode_line(sp, line, max_len)
                if len(ids) < 2:
                    continue
                for i, center in enumerate(ids):
                    lo = max(0, i - window)
                    hi = min(len(ids), i + window + 1)
                    ctx = [ids[j] for j in range(lo, hi) if j != i]
                    if not ctx:
                        continue
                    yield center, rng.choice(ctx)


def gpu_snapshot():
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.limit,power.draw,clocks.gr,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi-error: {exc}"


def geometry_probe(model, ids, device):
    rows = []
    t = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for mode in ["l0", "path", "cmul", "merge_no_bias", "tree"]:
            x = F.normalize(model.state(t, mode), dim=-1).detach().cpu().numpy()
            sims = x @ x.T
            off = sims[~np.eye(len(ids), dtype=bool)]
            rows.append(
                {
                    "mode": mode,
                    "cos_mean": float(off.mean()),
                    "cos_std": float(off.std()),
                    "cos_p05": float(np.quantile(off, 0.05)),
                    "cos_p95": float(np.quantile(off, 0.95)),
                }
            )
    return rows


def frame_probe(model, sp, device):
    words = sorted({w for p in PROBES for w in ([p[0], p[1]] + p[2] + p[3])})
    ids = torch.tensor([sp.encode(w.lower())[0] if sp.encode(w.lower()) else 0 for w in words], dtype=torch.long, device=device)
    rows = []
    with torch.no_grad():
        for mode in ["l0", "path", "cmul", "merge_no_bias", "tree"]:
            arr = model.state(ids, mode).detach().cpu().numpy()
            table = {w: arr[i] for i, w in enumerate(words)}
            aucs, mrrs, hit1s, hit3s = [], [], [], []
            for composite, base, pos, neg in PROBES:
                delta = table[composite] - table[base]
                scored = []
                for label, anchors in [(1, pos), (0, neg)]:
                    for a in anchors:
                        scored.append((a, label, cosine(delta, table[a] - table[base])))
                scored.sort(key=lambda x: -x[2])
                pos_scores = [s for _, lab, s in scored if lab == 1]
                neg_scores = [s for _, lab, s in scored if lab == 0]
                first = next(i + 1 for i, (_, lab, _) in enumerate(scored) if lab == 1)
                aucs.append(auc_pairwise(pos_scores, neg_scores))
                mrrs.append(1.0 / first)
                hit1s.append(int(any(lab == 1 for _, lab, _ in scored[:1])))
                hit3s.append(int(any(lab == 1 for _, lab, _ in scored[:3])))
            rows.append(
                {
                    "mode": mode,
                    "auc": float(np.mean(aucs)),
                    "mrr": float(np.mean(mrrs)),
                    "hit1": float(np.mean(hit1s)),
                    "hit3": float(np.mean(hit3s)),
                }
            )
    return rows


def sync_to_nas(local_root, nas_root):
    nas = Path(nas_root)
    if not str(nas).startswith("/mnt/nas/"):
        return
    src = f"{Path(local_root)}/"
    try:
        nas.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", src, f"{nas}/"], check=True)
        return
    except PermissionError:
        pass
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    subprocess.run(["sudo", "mkdir", "-p", str(nas)], check=True)
    try:
        subprocess.run(["sudo", "rsync", "-a", "--delete", src, f"{nas}/"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(["sudo", "cp", "-a", f"{Path(local_root)}/.", str(nas)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--nas-root", default=DEFAULT_NAS_ROOT)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--max-lines", type=int, default=500000)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import sentencepiece as spm

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    local_root = Path(args.local_root)
    ckpt_dir = local_root / "checkpoints"
    local_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = local_root / "run.log"
    summary_path = local_root / "summary.json"
    stop = {"flag": False}

    def log(msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def handle_signal(signum, frame):
        stop["flag"] = True
        log(f"received signal {signum}, will stop after current step")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe)
    model = TreeHeapWorldModel(vocab=32000, dim=128, depth=5).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    stream = pair_stream(args.data, sp, args.window, args.max_len, args.seed, args.max_lines)
    deadline = time.time() + args.hours * 3600
    metrics_rows, geometry_rows, frame_rows, gpu_rows = [], [], [], []

    summary = {"started_at": now(), "config": vars(args), "events": [], "status": "running"}

    def save_summary():
        summary.update(
            {
                "updated_at": now(),
                "metrics_rows": len(metrics_rows),
                "geometry_rows": len(geometry_rows),
                "frame_rows": len(frame_rows),
                "gpu_rows": len(gpu_rows),
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("starting world-model night train")
    log(f"gpu snapshot: {gpu_snapshot()}")
    save_summary()

    probe_ids = list(range(1000, 3000, 2))[:512]
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            if stop["flag"] or time.time() > deadline:
                break
            centers, contexts = [], []
            while len(centers) < args.batch_size:
                c, t = next(stream)
                if c != t:
                    centers.append(c)
                    contexts.append(t)
            center = torch.tensor(centers, dtype=torch.long, device=args.device)
            ctx = torch.tensor(contexts, dtype=torch.long, device=args.device)
            logits = model(center, ctx) / args.temperature
            labels = torch.arange(logits.shape[0], dtype=torch.long, device=args.device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == labels).float().mean().item()
            global_step += 1
            epoch_loss += float(loss.item())
            epoch_acc += acc
            if global_step % 50 == 0:
                snap = gpu_snapshot()
                gpu_rows.append({"time": now(), "epoch": epoch, "step": global_step, "snapshot": snap})
                write_csv(local_root / "gpu_snapshots.csv", gpu_rows)
                log(f"epoch={epoch} step={global_step} loss={loss.item():.4f} acc={acc:.4f} gpu={snap}")
                save_summary()
        done_steps = step if step else 0
        if done_steps:
            metrics_rows.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": epoch_loss / done_steps,
                    "inbatch_acc": epoch_acc / done_steps,
                }
            )
        g_rows = geometry_probe(model, probe_ids, args.device)
        for row in g_rows:
            row.update({"epoch": epoch, "global_step": global_step})
        geometry_rows.extend(g_rows)
        f_rows = frame_probe(model, sp, args.device)
        for row in f_rows:
            row.update({"epoch": epoch, "global_step": global_step})
        frame_rows.extend(f_rows)
        write_csv(local_root / "train_metrics.csv", metrics_rows)
        write_csv(local_root / "geometry_probe.csv", geometry_rows)
        write_csv(local_root / "frame_probe.csv", frame_rows)
        ckpt_path = ckpt_dir / f"world_model_ep{epoch:03d}_step{global_step:06d}.pt"
        torch.save(model.checkpoint_dict(), ckpt_path)
        log(f"saved {ckpt_path}")
        sync_to_nas(local_root, args.nas_root)
        save_summary()
        if stop["flag"] or time.time() > deadline:
            break

    summary["status"] = "stopped" if stop["flag"] else "complete"
    summary["finished_at"] = now()
    save_summary()
    sync_to_nas(local_root, args.nas_root)
    log(f"finished status={summary['status']} gpu snapshot: {gpu_snapshot()}")


if __name__ == "__main__":
    main()
