#!/usr/bin/env python3
"""S1 world-model coordinate probe with frozen external embeddings.

This experiment uses an existing embedding model only as a frozen coordinate
system.  It does not claim TreeHeap learned the external model's world knowledge.

Question:
  Can a probabilistic vector-plus TreeHeap encoder write two word vectors into a
  zero TreeHeap and produce a compound concept vector close to the frozen target
  coordinate?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Compound:
    left: str
    right: str
    target: str
    split: str
    family: str


def compound_data() -> List[Compound]:
    rows = [
        ("foot", "ball", "football", "train", "ball"),
        ("basket", "ball", "basketball", "train", "ball"),
        ("base", "ball", "baseball", "train", "ball"),
        ("snow", "ball", "snowball", "test", "ball"),
        ("hand", "ball", "handball", "ood", "ball"),
        ("rain", "coat", "raincoat", "train", "clothing"),
        ("over", "coat", "overcoat", "train", "clothing"),
        ("trench", "coat", "trenchcoat", "test", "clothing"),
        ("book", "shelf", "bookshelf", "train", "storage"),
        ("book", "case", "bookcase", "train", "storage"),
        ("cup", "board", "cupboard", "test", "storage"),
        ("key", "board", "keyboard", "ood", "device"),
        ("black", "board", "blackboard", "train", "surface"),
        ("white", "board", "whiteboard", "test", "surface"),
        ("note", "book", "notebook", "train", "book"),
        ("guide", "book", "guidebook", "train", "book"),
        ("text", "book", "textbook", "test", "book"),
        ("scrap", "book", "scrapbook", "ood", "book"),
        ("sun", "light", "sunlight", "train", "light"),
        ("moon", "light", "moonlight", "train", "light"),
        ("star", "light", "starlight", "test", "light"),
        ("flash", "light", "flashlight", "ood", "light"),
        ("bed", "room", "bedroom", "train", "room"),
        ("bath", "room", "bathroom", "train", "room"),
        ("class", "room", "classroom", "test", "room"),
        ("work", "shop", "workshop", "train", "place"),
        ("fire", "place", "fireplace", "train", "place"),
        ("air", "port", "airport", "test", "place"),
        ("sea", "side", "seaside", "train", "place"),
        ("river", "bank", "riverbank", "test", "place"),
        ("tooth", "brush", "toothbrush", "train", "tool"),
        ("hair", "brush", "hairbrush", "test", "tool"),
        ("paint", "brush", "paintbrush", "ood", "tool"),
        ("head", "phone", "headphone", "train", "device"),
        ("ear", "phone", "earphone", "test", "device"),
        ("door", "bell", "doorbell", "train", "device"),
        ("door", "way", "doorway", "ood", "place"),
    ]
    return [Compound(*row) for row in rows]


def normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_projection(raw_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 1.0, size=(raw_dim, out_dim))
    q, _ = np.linalg.qr(mat)
    return q[:, :out_dim]


def load_frozen_vectors(rows: Sequence[Compound], dim: int, seed: int, device: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    snapshot = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    model_source = str(snapshot) if snapshot.exists() else model_name
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SentenceTransformer(model_source, device=device)
    words = sorted({w for r in rows for w in (r.left, r.right, r.target)})
    raw = model.encode(words, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    proj = build_projection(raw.shape[1], dim, seed)
    vec = normalize_np(raw @ proj)
    return {w: vec[i].astype(np.float32) for i, w in enumerate(words)}, {
        "model": model_name,
        "model_source": model_source,
        "raw_dim": int(raw.shape[1]),
        "projected_dim": dim,
        "projection": "fixed random orthonormal projection",
        "frozen": True,
        "distillation_guard": "external embeddings are used only as fixed coordinates/targets, not as trainable teacher behavior",
    }


def split_rows(rows: Sequence[Compound], split: str) -> List[Compound]:
    return [r for r in rows if r.split == split]


def make_arrays(rows: Sequence[Compound], vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.stack([vectors[r.left] for r in rows])
    right = np.stack([vectors[r.right] for r in rows])
    target = np.stack([vectors[r.target] for r in rows])
    return left.astype(np.float32), right.astype(np.float32), target.astype(np.float32)


def candidate_matrix(rows: Sequence[Compound], vectors: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    names = [r.target for r in rows]
    mat = np.stack([vectors[n] for n in names]).astype(np.float32)
    return names, normalize_np(mat)


def metrics_from_pred(pred: np.ndarray, target: np.ndarray, candidates: np.ndarray, candidate_names: Sequence[str], gold_names: Sequence[str]) -> Dict:
    pred = normalize_np(pred)
    target = normalize_np(target)
    cos = np.sum(pred * target, axis=1)
    sims = pred @ candidates.T
    top_idx = sims.argmax(axis=1)
    top_names = [candidate_names[i] for i in top_idx]
    top1 = np.mean([p == g for p, g in zip(top_names, gold_names)])
    mrr = 0.0
    ranks = []
    for row, gold in zip(sims, gold_names):
        order = np.argsort(-row)
        gold_idx = candidate_names.index(gold)
        rank = int(np.where(order == gold_idx)[0][0]) + 1
        ranks.append(rank)
        mrr += 1.0 / rank
    return {
        "mean_cosine": float(np.mean(cos)),
        "min_cosine": float(np.min(cos)),
        "top1": float(top1),
        "mrr": float(mrr / len(gold_names)),
        "mean_rank": float(np.mean(ranks)),
        "examples": [
            {"gold": g, "pred_top1": p, "rank": int(rank), "cosine": float(c)}
            for g, p, rank, c in zip(gold_names[:8], top_names[:8], ranks[:8], cos[:8])
        ],
    }


class ConcatMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        x = torch.cat([left, right, left * right, torch.abs(left - right)], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class TreeHeapVectorPlus(nn.Module):
    def __init__(self, dim: int, nodes: int, hidden: int):
        super().__init__()
        self.nodes = nodes
        self.dim = dim
        self.route = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nodes),
        )
        self.update = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.read = nn.Sequential(
            nn.Linear(nodes * dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def write(self, h: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        summary = h.mean(dim=1)
        logits = self.route(torch.cat([x, summary], dim=-1))
        p = F.softmax(logits, dim=-1)
        u = self.update(x)
        h = h + p[:, :, None] * u[:, None, :]
        return h, p

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(left.shape[0], self.nodes, self.dim, device=left.device)
        h, p1 = self.write(h, left)
        h, p2 = self.write(h, right)
        y = self.read(h.reshape(left.shape[0], self.nodes * self.dim))
        return F.normalize(y, dim=-1), torch.stack([p1, p2], dim=1)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.sum(pred * target, dim=-1)).mean()


def train_model(model: nn.Module, arrays: Tuple[np.ndarray, np.ndarray, np.ndarray], args) -> Tuple[nn.Module, List[Dict]]:
    left, right, target = arrays
    device = torch.device(args.train_device)
    model.to(device)
    x1 = torch.tensor(left, dtype=torch.float32, device=device)
    x2 = torch.tensor(right, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    trace = []
    for epoch in range(args.epochs):
        opt.zero_grad()
        if isinstance(model, TreeHeapVectorPlus):
            pred, route = model(x1, x2)
            route_entropy = -(route * torch.log(route.clamp_min(1e-9))).sum(dim=-1).mean()
            loss = cosine_loss(pred, y) + args.route_entropy_weight * route_entropy
        else:
            pred = model(x1, x2)
            loss = cosine_loss(pred, y)
        loss.backward()
        opt.step()
        if epoch in {0, args.epochs // 2, args.epochs - 1}:
            trace.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    return model.cpu(), trace


def predict_model(model: nn.Module, arrays: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, Dict]:
    left, right, _ = arrays
    with torch.no_grad():
        x1 = torch.tensor(left, dtype=torch.float32)
        x2 = torch.tensor(right, dtype=torch.float32)
        if isinstance(model, TreeHeapVectorPlus):
            pred, route = model(x1, x2)
            return pred.numpy(), {
                "avg_route_left": route[:, 0, :].mean(dim=0).numpy().tolist(),
                "avg_route_right": route[:, 1, :].mean(dim=0).numpy().tolist(),
                "route_argmax_left": route[:, 0, :].argmax(dim=-1).numpy().tolist(),
                "route_argmax_right": route[:, 1, :].argmax(dim=-1).numpy().tolist(),
            }
        pred = model(x1, x2)
        return pred.numpy(), {}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--nodes", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--route-entropy-weight", type=float, default=0.001)
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compound_data()
    vectors, embedding_info = load_frozen_vectors(rows, args.dim, args.seed, args.embed_device)
    all_targets, cand = candidate_matrix(rows, vectors)
    train_rows, test_rows, ood_rows = split_rows(rows, "train"), split_rows(rows, "test"), split_rows(rows, "ood")
    train_arr = make_arrays(train_rows, vectors)
    test_arr = make_arrays(test_rows, vectors)
    ood_arr = make_arrays(ood_rows, vectors)

    models: Dict[str, Dict] = {}
    traces: Dict[str, List[Dict]] = {}

    def eval_pred(name: str, pred_by_split: Dict[str, np.ndarray], route_info=None, params=0):
        models[name] = {
            "parameters": int(params),
            "train": metrics_from_pred(pred_by_split["train"], train_arr[2], cand, all_targets, [r.target for r in train_rows]),
            "test": metrics_from_pred(pred_by_split["test"], test_arr[2], cand, all_targets, [r.target for r in test_rows]),
            "ood": metrics_from_pred(pred_by_split["ood"], ood_arr[2], cand, all_targets, [r.target for r in ood_rows]),
        }
        if route_info:
            models[name]["route_info"] = route_info

    # Frozen vector baselines.
    eval_pred(
        "vector_add",
        {
            "train": normalize_np(train_arr[0] + train_arr[1]),
            "test": normalize_np(test_arr[0] + test_arr[1]),
            "ood": normalize_np(ood_arr[0] + ood_arr[1]),
        },
        params=0,
    )

    concat = ConcatMLP(args.dim, args.hidden)
    concat, traces["concat_mlp"] = train_model(concat, train_arr, args)
    eval_pred(
        "concat_mlp",
        {
            "train": predict_model(concat, train_arr)[0],
            "test": predict_model(concat, test_arr)[0],
            "ood": predict_model(concat, ood_arr)[0],
        },
        params=parameter_count(concat),
    )

    tree = TreeHeapVectorPlus(args.dim, args.nodes, args.hidden)
    tree, traces["treeheap_prob_vector_plus"] = train_model(tree, train_arr, args)
    p_train, route_train = predict_model(tree, train_arr)
    p_test, route_test = predict_model(tree, test_arr)
    p_ood, route_ood = predict_model(tree, ood_arr)
    eval_pred(
        "treeheap_prob_vector_plus",
        {"train": p_train, "test": p_test, "ood": p_ood},
        route_info={"train": route_train, "test": route_test, "ood": route_ood},
        params=parameter_count(tree),
    )

    # Conservative decision: TreeHeap must beat vector_add and be comparable to
    # concat MLP on OOD cosine to be considered a supported coordinate pilot.
    tree_ood = models["treeheap_prob_vector_plus"]["ood"]["mean_cosine"]
    add_ood = models["vector_add"]["ood"]["mean_cosine"]
    mlp_ood = models["concat_mlp"]["ood"]["mean_cosine"]
    supported = bool(tree_ood > add_ood and tree_ood >= mlp_ood - 0.03)

    summary = {
        "claim": "S1-WM-C01",
        "predict": "P-S1-WM01",
        "seed": args.seed,
        "host": "io.grepcode.cn",
        "embedding_coordinate": embedding_info,
        "dataset": {
            "kind": "compound word coordinate probe",
            "train": len(train_rows),
            "test": len(test_rows),
            "ood": len(ood_rows),
            "targets": len(all_targets),
            "rows": [asdict(r) for r in rows],
        },
        "models": models,
        "traces": traces,
        "pilot_pass": supported,
        "interpretation": {
            "supported": "TreeHeap prob vector plus can map compound inputs toward frozen world-coordinate targets at least comparably to concat MLP."
            if supported
            else "TreeHeap prob vector plus did not meet the conservative world-coordinate pilot gate.",
            "not_proved": [
                "not TreeHeap's own learned world model",
                "not proof of superiority over all MLP/Transformer models",
                "not WMT translation",
                "frozen external embedding is a coordinate ruler, not a deployable teacher",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dataset.json").write_text(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for model_name, rows_trace in traces.items():
            for row in rows_trace:
                f.write(json.dumps({"model": model_name, **row}, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# S1 world-model compound coordinate probe",
                "",
                "Frozen sentence-transformer embeddings are used only as an external coordinate system.",
                "The experiment tests whether TreeHeap prob vector plus can write/compose into that coordinate space.",
                "",
                "Artifacts:",
                "",
                "- `summary.json`: metrics and decision",
                "- `dataset.json`: compound split",
                "- `trace.jsonl`: training traces",
                "",
                "This is not a claim that TreeHeap learned its own full world model.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
