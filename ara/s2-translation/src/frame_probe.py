"""
frame_probe.py - P-FRAME01 diagnostic for TreeHeap world-model geometry.

This is a pilot evidence gate, not final proof. It asks whether
`composite - base` directions point toward interpretable relation anchors
inside different TreeHeap readout modes.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_BPE = "/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model"
DEFAULT_CKPT_GLOB = "/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep*.pt"


PROBES = [
    {
        "frame": "football",
        "composite": "football",
        "base": "ball",
        "positive": ["foot", "kick", "field", "goal", "soccer", "team"],
        "negative": ["hand", "throw", "court", "basket", "bat", "racket", "net", "ice", "wheel", "engine"],
    },
    {
        "frame": "basketball",
        "composite": "basketball",
        "base": "ball",
        "positive": ["hand", "throw", "court", "basket", "hoop", "team"],
        "negative": ["foot", "kick", "goal", "bat", "racket", "net", "ice", "wheel", "engine", "grass"],
    },
    {
        "frame": "baseball",
        "composite": "baseball",
        "base": "ball",
        "positive": ["bat", "glove", "pitch", "base", "field", "team"],
        "negative": ["foot", "kick", "basket", "racket", "net", "ice", "wheel", "engine", "court", "throw"],
    },
    {
        "frame": "tennis",
        "composite": "tennis",
        "base": "ball",
        "positive": ["racket", "court", "net", "serve", "match"],
        "negative": ["foot", "kick", "basket", "bat", "glove", "ice", "wheel", "engine", "goal", "base"],
    },
    {
        "frame": "volleyball",
        "composite": "volleyball",
        "base": "ball",
        "positive": ["hand", "net", "serve", "court", "team"],
        "negative": ["foot", "kick", "basket", "bat", "racket", "ice", "wheel", "engine", "goal", "base"],
    },
    {
        "frame": "snowball",
        "composite": "snowball",
        "base": "ball",
        "positive": ["snow", "cold", "winter", "ice", "throw"],
        "negative": ["foot", "kick", "basket", "bat", "racket", "court", "engine", "wheel", "goal", "team"],
    },
    {
        "frame": "fireball",
        "composite": "fireball",
        "base": "ball",
        "positive": ["fire", "hot", "flame", "burn", "magic"],
        "negative": ["foot", "kick", "basket", "bat", "racket", "court", "ice", "wheel", "goal", "team"],
    },
    {
        "frame": "wheelchair",
        "composite": "wheelchair",
        "base": "chair",
        "positive": ["wheel", "move", "roll", "patient", "hospital"],
        "negative": ["foot", "kick", "basket", "bat", "racket", "fire", "snow", "goal", "field", "team"],
    },
    {
        "frame": "sailboat",
        "composite": "sailboat",
        "base": "boat",
        "positive": ["sail", "wind", "water", "sea", "harbor"],
        "negative": ["wheel", "engine", "road", "foot", "basket", "fire", "snow", "court", "goal", "bat"],
    },
    {
        "frame": "motorboat",
        "composite": "motorboat",
        "base": "boat",
        "positive": ["motor", "engine", "fuel", "water", "speed"],
        "negative": ["sail", "wind", "foot", "basket", "fire", "snow", "court", "goal", "bat", "racket"],
    },
    {
        "frame": "racecar",
        "composite": "racecar",
        "base": "car",
        "positive": ["race", "speed", "engine", "track", "wheel"],
        "negative": ["sail", "wind", "water", "foot", "basket", "fire", "snow", "court", "bat", "racket"],
    },
    {
        "frame": "policecar",
        "composite": "policecar",
        "base": "car",
        "positive": ["police", "sirens", "law", "officer", "street"],
        "negative": ["sail", "wind", "basket", "fire", "snow", "court", "bat", "racket", "goal", "team"],
    },
    {
        "frame": "teacup",
        "composite": "teacup",
        "base": "cup",
        "positive": ["tea", "drink", "hot", "kettle", "saucer"],
        "negative": ["football", "engine", "wheel", "police", "snow", "court", "bat", "racket", "goal", "team"],
    },
    {
        "frame": "coffeehouse",
        "composite": "coffeehouse",
        "base": "house",
        "positive": ["coffee", "drink", "shop", "table", "cup"],
        "negative": ["football", "engine", "wheel", "police", "snow", "court", "bat", "racket", "goal", "team"],
    },
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-12)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def auc_pairwise(pos_scores, neg_scores):
    total = 0
    good = 0.0
    for p in pos_scores:
        for n in neg_scores:
            total += 1
            if p > n:
                good += 1.0
            elif p == n:
                good += 0.5
    return good / max(total, 1)


def average_precision(labels):
    hits = 0
    vals = []
    for i, label in enumerate(labels, start=1):
        if label:
            hits += 1
            vals.append(hits / i)
    return float(np.mean(vals)) if vals else 0.0


class TreeHeapCkpt:
    def __init__(self, ckpt_path, bpe_path, device):
        import sentencepiece as spm

        self.ckpt_path = ckpt_path
        self.name = Path(ckpt_path).stem
        self.device = torch.device(device)
        self.d = 128
        self.td = 5
        self.vocab = 32000
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.l0 = ckpt["L0"]["weight"].float().to(self.device)
        self.t_nodes = [ckpt["t_nodes"][i]["weight"].float().to(self.device) for i in range(self.td)]
        self.tm_w = ckpt["t_merge"]["weight"].float().to(self.device)
        self.tm_b = ckpt["t_merge"]["bias"].float().to(self.device)

    def encode(self, word):
        pieces = self.sp.encode(word.lower())
        return int(pieces[0]) if pieces else 0

    def ids(self, words):
        return torch.tensor([self.encode(w) for w in words], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def path(self, ids):
        w = torch.zeros((ids.numel(), self.d), device=self.device)
        for level in range(self.td):
            stride = max(self.vocab // (2**level), 1)
            if level == 0:
                node_idx = torch.zeros_like(ids)
            else:
                node_idx = torch.clamp(ids // stride, 0, (2**level) - 1)
            w = w + self.t_nodes[level][node_idx]
        return F.normalize(w, dim=-1)

    @torch.no_grad()
    def cmul_pre(self, ids):
        t = F.normalize(self.l0[ids], dim=-1)
        w = self.path(ids)
        left = t[:, :64] * w[:, :64] - t[:, 64:] * w[:, 64:]
        right = t[:, :64] * w[:, 64:] + t[:, 64:] * w[:, :64]
        return torch.cat([left, right], dim=-1)

    @torch.no_grad()
    def vectors(self, words, mode):
        ids = self.ids(words)
        if mode == "l0":
            return F.normalize(self.l0[ids], dim=-1).cpu().numpy()
        if mode == "path":
            return self.path(ids).cpu().numpy()
        if mode == "cmul":
            return self.cmul_pre(ids).cpu().numpy()
        if mode == "merge_no_bias":
            return (self.cmul_pre(ids) @ self.tm_w.T).cpu().numpy()
        if mode == "tree":
            return (self.cmul_pre(ids) @ self.tm_w.T + self.tm_b).cpu().numpy()
        raise ValueError(mode)


def all_probe_words():
    words = set()
    for p in PROBES:
        words.add(p["composite"])
        words.add(p["base"])
        words.update(p["positive"])
        words.update(p["negative"])
    return sorted(words)


def random_vectors(words, dim, seed):
    rng = np.random.RandomState(seed)
    return {w: normalize_np(rng.randn(1, dim).astype(np.float32))[0] for w in words}


def evaluate_table(vecs, probes):
    rows = []
    for p in probes:
        comp = vecs[p["composite"]]
        base = vecs[p["base"]]
        delta = comp - base
        scored = []
        for label, anchors in [("positive", p["positive"]), ("negative", p["negative"])]:
            for a in anchors:
                if a not in vecs:
                    continue
                anchor_delta = vecs[a] - base
                scored.append({"anchor": a, "label": label, "score": cosine(delta, anchor_delta)})
        scored.sort(key=lambda x: -x["score"])
        labels = [1 if x["label"] == "positive" else 0 for x in scored]
        pos_scores = [x["score"] for x in scored if x["label"] == "positive"]
        neg_scores = [x["score"] for x in scored if x["label"] == "negative"]
        first_pos_rank = next((i + 1 for i, x in enumerate(scored) if x["label"] == "positive"), len(scored) + 1)
        rows.append(
            {
                "frame": p["frame"],
                "composite": p["composite"],
                "base": p["base"],
                "n_pos": len(pos_scores),
                "n_neg": len(neg_scores),
                "auc": auc_pairwise(pos_scores, neg_scores),
                "ap": average_precision(labels),
                "mrr": 1.0 / first_pos_rank,
                "hit1": int(any(x["label"] == "positive" for x in scored[:1])),
                "hit3": int(any(x["label"] == "positive" for x in scored[:3])),
                "top5": " ".join([f"{x['anchor']}:{x['label']}:{x['score']:.4f}" for x in scored[:5]]),
            }
        )
    return rows


def summarize(rows):
    keys = ["auc", "ap", "mrr", "hit1", "hit3"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--ckpt-glob", default=DEFAULT_CKPT_GLOB)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    words = all_probe_words()
    modes = ["random", "l0", "path", "cmul", "merge_no_bias", "tree", "centered_tree"]
    detail_rows = []
    summary_rows = []
    events = []

    def event(msg):
        print(f"[{now()}] {msg}", flush=True)
        events.append({"time": now(), "msg": msg})

    event("starting P-FRAME01 pilot")
    random_table = random_vectors(words, 128, args.seed)
    rows = evaluate_table(random_table, PROBES)
    for r in rows:
        r.update({"checkpoint": "random", "mode": "random"})
    detail_rows.extend(rows)
    summary_rows.append({"checkpoint": "random", "mode": "random", **summarize(rows)})

    ckpts = sorted(glob.glob(args.ckpt_glob))
    event(f"found checkpoints={ckpts}")
    for ckpt_path in ckpts:
        event(f"loading {ckpt_path}")
        ckpt = TreeHeapCkpt(ckpt_path, args.bpe, args.device)
        cache = {}
        for mode in ["l0", "path", "cmul", "merge_no_bias", "tree"]:
            arr = ckpt.vectors(words, mode)
            if mode == "tree":
                cache["centered_tree"] = {w: v for w, v in zip(words, arr - arr.mean(axis=0, keepdims=True))}
            cache[mode] = {w: v for w, v in zip(words, arr)}
        for mode in ["l0", "path", "cmul", "merge_no_bias", "tree", "centered_tree"]:
            rows = evaluate_table(cache[mode], PROBES)
            for r in rows:
                r.update({"checkpoint": ckpt.name, "mode": mode})
            detail_rows.extend(rows)
            summary_rows.append({"checkpoint": ckpt.name, "mode": mode, **summarize(rows)})
            write_csv(out / "frame_probe_details.csv", detail_rows)
            write_csv(out / "frame_probe_summary.csv", summary_rows)

    best = sorted(summary_rows, key=lambda r: (r["mrr"], r["auc"], r["hit3"]), reverse=True)[:8]
    l0_rows = [r for r in summary_rows if r["mode"] == "l0"]
    best_l0 = max(l0_rows, key=lambda r: r["mrr"]) if l0_rows else None
    best_internal = max(
        [r for r in summary_rows if r["mode"] in {"cmul", "merge_no_bias", "centered_tree", "tree"}],
        key=lambda r: r["mrr"],
        default=None,
    )
    verdict = "inconclusive"
    if best_l0 and best_internal:
        if (
            best_internal["mrr"] > best_l0["mrr"]
            and best_internal["auc"] > best_l0["auc"]
            and best_internal["hit3"] > best_l0["hit3"]
        ):
            verdict = "pilot_pass"
        elif best_l0["mrr"] >= best_internal["mrr"] and best_l0["auc"] >= best_internal["auc"]:
            verdict = "pilot_fail_l0_not_beaten"

    summary = {
        "started_finished_at": now(),
        "predict": "P-FRAME01",
        "config": vars(args),
        "n_probes": len(PROBES),
        "n_words": len(words),
        "checkpoints": ckpts,
        "modes": modes,
        "verdict": verdict,
        "best_l0": best_l0,
        "best_internal": best_internal,
        "top_rows": best,
        "events": events,
        "files": ["frame_probe_summary.csv", "frame_probe_details.csv", "frame_probe_summary.json", "README.md"],
    }
    (out / "frame_probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# P-FRAME01 Frame Probe",
        "",
        f"Created: {now()}",
        "",
        "## Predict",
        "",
        "`composite - base` should point toward interpretable relation anchors inside a useful world-model frame.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "This is a pilot diagnostic over legacy checkpoints. It is not final positive evidence for TreeHeap.",
        "",
        "## Files",
        "",
        "- `frame_probe_summary.json`",
        "- `frame_probe_summary.csv`",
        "- `frame_probe_details.csv`",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    event(f"finished verdict={verdict}")


if __name__ == "__main__":
    main()
