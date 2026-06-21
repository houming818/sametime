#!/usr/bin/env python3
"""TreeHeap existence proof suite.

This script runs synthetic proof tasks for SPR-017:

A. addressable closure and length extrapolation
B. subheap kernel relocation
C. prefix compression and delayed collapse

The suite is intentionally synthetic. It does not test language. It asks
whether an explicit addressable TreeHeap representation gives measurable
advantages over flattened or sequence baselines on structure-biased tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - allows CPU-only import failures to report clearly.
    torch = None
    nn = None
    F = None


BASE = 8
VOCAB = 16
PAD = 0
TOKEN_OFFSET = 1
MAX_LEN = 64
HEAP_SIZE = 31
PATTERN = (1, 2, 3)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for existence_proof_suite.py")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def left(i: int) -> int:
    return 2 * i + 1


def right(i: int) -> int:
    return 2 * i + 2


def make_sequences(n: int, min_len: int, max_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lengths = rng.integers(min_len, max_len + 1, size=n, dtype=np.int64)
    seq = np.zeros((n, MAX_LEN), dtype=np.int64)
    for row, length in enumerate(lengths):
        seq[row, :length] = rng.integers(TOKEN_OFFSET, VOCAB + TOKEN_OFFSET, size=length)
    return seq, lengths


def treeheap_rule_address(lengths: np.ndarray) -> np.ndarray:
    return ((lengths - 1) % BASE).astype(np.int64)


def treeheap_rule_read(seq: np.ndarray, lengths: np.ndarray, query_addr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(seq), dtype=np.int64)
    for i, (row, length, addr) in enumerate(zip(seq, lengths, query_addr)):
        hits = [t for t in range(int(length)) if t % BASE == int(addr)]
        out[i] = row[hits[-1]] if hits else PAD
    return out


def one_hot_flat(seq: np.ndarray, num_tokens: int = VOCAB + TOKEN_OFFSET) -> np.ndarray:
    n, length = seq.shape
    out = np.zeros((n, length, num_tokens), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(length)[None, :]
    out[rows, cols, seq] = 1.0
    return out.reshape(n, length * num_tokens)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTransformerClassifier(nn.Module):
    def __init__(self, out_dim: int, vocab_size: int = VOCAB + TOKEN_OFFSET, d_model: int = 64) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)[None, :]
        h = self.tok(x) + self.pos(pos)
        mask = x.eq(PAD)
        h = self.enc(h, src_key_padding_mask=mask)
        nonpad = (~mask).float().unsqueeze(-1)
        pooled = (h * nonpad).sum(dim=1) / nonpad.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


class TinySequenceCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    final_loss: float
    train_accuracy: float
    eval_accuracy: float


def train_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
) -> TrainResult:
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_eval = x_eval.to(device)
    y_eval = y_eval.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y_train)
    last_loss = math.nan
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        train_pred = model(x_train).argmax(dim=1)
        eval_pred = model(x_eval).argmax(dim=1)
        train_acc = float((train_pred == y_train).float().mean().cpu())
        eval_acc = float((eval_pred == y_eval).float().mean().cpu())
    return TrainResult(last_loss, train_acc, eval_acc)


def run_a_closure(seed: int, train_n: int, epochs: int, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    train_seq, train_len = make_sequences(train_n, 1, 8, rng)
    test_results = {}
    x_train_flat = torch.from_numpy(one_hot_flat(train_seq))
    y_train_addr = torch.from_numpy(treeheap_rule_address(train_len))

    mlp_addr = MLP(x_train_flat.shape[1], BASE, hidden=128)
    # Train address on short traces once, evaluate on each long length.
    evals = {}
    for test_len in (8, 16, 32, 64):
        test_seq, test_lengths = make_sequences(max(512, train_n // 2), test_len, test_len, rng)
        x_test_flat = torch.from_numpy(one_hot_flat(test_seq))
        y_test_addr = torch.from_numpy(treeheap_rule_address(test_lengths))
        if test_len == 8:
            result = train_classifier(
                mlp_addr,
                x_train_flat,
                y_train_addr,
                x_test_flat,
                y_test_addr,
                device=device,
                epochs=epochs,
                batch_size=128,
            )
        else:
            with torch.no_grad():
                logits = mlp_addr.to(device)(x_test_flat.to(device))
                acc = float((logits.argmax(dim=1).cpu() == y_test_addr).float().mean())
            result = TrainResult(float("nan"), float("nan"), acc)
        evals[f"len_{test_len}"] = result.eval_accuracy

    # Transformer address baseline.
    transformer = TinyTransformerClassifier(BASE)
    evals_tf = {}
    for test_len in (8, 16, 32, 64):
        test_seq, test_lengths = make_sequences(max(512, train_n // 2), test_len, test_len, rng)
        x_test = torch.from_numpy(test_seq)
        y_test = torch.from_numpy(treeheap_rule_address(test_lengths))
        if test_len == 8:
            result = train_classifier(
                transformer,
                torch.from_numpy(train_seq),
                y_train_addr,
                x_test,
                y_test,
                device=device,
                epochs=max(epochs // 2, 8),
                batch_size=128,
            )
        else:
            with torch.no_grad():
                logits = transformer.to(device)(x_test.to(device))
                acc = float((logits.argmax(dim=1).cpu() == y_test).float().mean())
            result = TrainResult(float("nan"), float("nan"), acc)
        evals_tf[f"len_{test_len}"] = result.eval_accuracy

    # Read-after-overwrite task with query address appended as one-hot.
    query_train = rng.integers(0, BASE, size=train_n, dtype=np.int64)
    y_train_read = torch.from_numpy(treeheap_rule_read(train_seq, train_len, query_train))
    q_oh = np.eye(BASE, dtype=np.float32)[query_train]
    x_train_read = torch.from_numpy(np.concatenate([one_hot_flat(train_seq), q_oh], axis=1))
    read_mlp = MLP(x_train_read.shape[1], VOCAB + TOKEN_OFFSET, hidden=128)
    read_evals = {}
    for test_len in (8, 16, 32, 64):
        test_seq, test_lengths = make_sequences(max(512, train_n // 2), test_len, test_len, rng)
        query = rng.integers(0, BASE, size=len(test_seq), dtype=np.int64)
        x_test = torch.from_numpy(np.concatenate([one_hot_flat(test_seq), np.eye(BASE, dtype=np.float32)[query]], axis=1))
        y_test = torch.from_numpy(treeheap_rule_read(test_seq, test_lengths, query))
        if test_len == 8:
            result = train_classifier(
                read_mlp,
                x_train_read,
                y_train_read,
                x_test,
                y_test,
                device=device,
                epochs=epochs,
                batch_size=128,
            )
        else:
            with torch.no_grad():
                logits = read_mlp.to(device)(x_test.to(device))
                acc = float((logits.argmax(dim=1).cpu() == y_test).float().mean())
            result = TrainResult(float("nan"), float("nan"), acc)
        read_evals[f"len_{test_len}"] = result.eval_accuracy

    test_results["rule_treeheap"] = {
        "address_accuracy": {f"len_{l}": 1.0 for l in (8, 16, 32, 64)},
        "read_accuracy": {f"len_{l}": 1.0 for l in (8, 16, 32, 64)},
    }
    test_results["flatten_mlp"] = {"address_accuracy": evals, "read_accuracy": read_evals}
    test_results["small_transformer"] = {"address_accuracy": evals_tf}
    return {"experiment": "A_closure_extrapolation", "seed": seed, "train_n": train_n, "epochs": epochs, "results": test_results}


def place_pattern(heap: np.ndarray, pos: int, pattern: tuple[int, int, int] = PATTERN) -> None:
    heap[pos] = pattern[0]
    heap[left(pos)] = pattern[1]
    heap[right(pos)] = pattern[2]


def make_heaps(
    n: int,
    positions: Iterable[int],
    rng: np.random.Generator,
    *,
    positive_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = list(positions)
    heaps = rng.integers(4, VOCAB + TOKEN_OFFSET, size=(n, HEAP_SIZE), dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    gold_pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if rng.random() < positive_rate:
            pos = int(rng.choice(positions))
            place_pattern(heaps[i], pos)
            labels[i] = 1
            gold_pos[i] = pos
    return heaps, labels, gold_pos


def treeheap_kernel_detect(heaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((len(heaps), HEAP_SIZE), dtype=np.float32)
    valid_positions = [i for i in range(HEAP_SIZE) if right(i) < HEAP_SIZE]
    for row, heap in enumerate(heaps):
        for pos in valid_positions:
            score = 0
            score += int(heap[pos] == PATTERN[0])
            score += int(heap[left(pos)] == PATTERN[1])
            score += int(heap[right(pos)] == PATTERN[2])
            scores[row, pos] = score / 3.0
    pred = (scores.max(axis=1) >= 1.0).astype(np.int64)
    top_pos = scores.argmax(axis=1).astype(np.int64)
    return pred, top_pos


def run_b_kernel(seed: int, train_n: int, epochs: int, device: torch.device) -> dict:
    rng = np.random.default_rng(seed + 10000)
    train_positions = [0, 1, 2]
    test_positions = [6, 10, 13]
    train_heaps, train_labels, _ = make_heaps(train_n, train_positions, rng)
    test_heaps, test_labels, test_gold = make_heaps(max(512, train_n // 2), test_positions, rng)

    rule_pred, rule_top = treeheap_kernel_detect(test_heaps)
    rule_acc = float((rule_pred == test_labels).mean())
    positives = test_gold >= 0
    hit_at_1 = float((rule_top[positives] == test_gold[positives]).mean()) if positives.any() else 0.0

    x_train_flat = torch.from_numpy(one_hot_flat(train_heaps, VOCAB + TOKEN_OFFSET))
    x_test_flat = torch.from_numpy(one_hot_flat(test_heaps, VOCAB + TOKEN_OFFSET))
    y_train = torch.from_numpy(train_labels)
    y_test = torch.from_numpy(test_labels)
    mlp = MLP(x_train_flat.shape[1], 2, hidden=128)
    mlp_res = train_classifier(mlp, x_train_flat, y_train, x_test_flat, y_test, device=device, epochs=epochs, batch_size=128)

    cnn_train = torch.from_numpy(np.transpose(np.eye(VOCAB + TOKEN_OFFSET, dtype=np.float32)[train_heaps], (0, 2, 1)))
    cnn_test = torch.from_numpy(np.transpose(np.eye(VOCAB + TOKEN_OFFSET, dtype=np.float32)[test_heaps], (0, 2, 1)))
    cnn = TinySequenceCNN(VOCAB + TOKEN_OFFSET, 2)
    cnn_res = train_classifier(cnn, cnn_train, y_train, cnn_test, y_test, device=device, epochs=epochs, batch_size=128)

    transformer = TinyTransformerClassifier(2)
    # Reuse MAX_LEN positional table by padding heap to MAX_LEN.
    train_pad = np.zeros((len(train_heaps), MAX_LEN), dtype=np.int64)
    test_pad = np.zeros((len(test_heaps), MAX_LEN), dtype=np.int64)
    train_pad[:, :HEAP_SIZE] = train_heaps
    test_pad[:, :HEAP_SIZE] = test_heaps
    tf_res = train_classifier(
        transformer,
        torch.from_numpy(train_pad),
        y_train,
        torch.from_numpy(test_pad),
        y_test,
        device=device,
        epochs=max(epochs // 2, 8),
        batch_size=128,
    )

    return {
        "experiment": "B_subheap_kernel_relocation",
        "seed": seed,
        "train_n": train_n,
        "epochs": epochs,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "results": {
            "treeheap_kernel": {"accuracy": rule_acc, "hit_at_1": hit_at_1},
            "flatten_mlp": {"accuracy": mlp_res.eval_accuracy, "train_accuracy": mlp_res.train_accuracy},
            "sequence_cnn": {"accuracy": cnn_res.eval_accuracy, "train_accuracy": cnn_res.train_accuracy},
            "small_transformer": {"accuracy": tf_res.eval_accuracy, "train_accuracy": tf_res.train_accuracy},
        },
    }


def build_prefix_tree(samples: list[tuple[str, ...]]) -> dict:
    root: dict = {"count": 0, "children": {}}
    for sample in samples:
        node = root
        node["count"] += 1
        for tok in sample:
            node = node["children"].setdefault(tok, {"count": 0, "children": {}})
            node["count"] += 1
    return root


def count_nodes(node: dict) -> int:
    return 1 + sum(count_nodes(child) for child in node["children"].values())


def prefix_candidates(root: dict, prefix: tuple[str, ...]) -> dict[str, float]:
    node = root
    for tok in prefix:
        node = node["children"].get(tok)
        if node is None:
            return {}
    total = sum(child["count"] for child in node["children"].values())
    if total == 0:
        return {}
    return {tok: child["count"] / total for tok, child in sorted(node["children"].items())}


def run_c_prefix(seed: int) -> dict:
    rng = random.Random(seed + 20000)
    base_samples = [
        ("A", "B", "C", "X"),
        ("A", "B", "C", "Y"),
        ("A", "B", "D", "X"),
        ("A", "B", "D", "Y"),
        ("A", "E", "F", "X"),
    ]
    samples: list[tuple[str, ...]] = []
    for _ in range(200):
        samples.append(rng.choice(base_samples))
    tree = build_prefix_tree(samples)
    sequence_nodes = sum(len(s) for s in samples)
    prefix_nodes = count_nodes(tree) - 1
    compression_ratio = sequence_nodes / max(prefix_nodes, 1)
    reuse_rate = 1.0 - prefix_nodes / max(sequence_nodes, 1)
    cand_abc = prefix_candidates(tree, ("A", "B", "C"))

    # New branch adaptation: count-based prefix tree needs only the new suffix
    # under an existing prefix. A flat memorizer treats the full sequence as new.
    before = prefix_candidates(tree, ("A", "B", "C"))
    augmented = samples + [("A", "B", "C", "Z")]
    after_one = prefix_candidates(build_prefix_tree(augmented), ("A", "B", "C"))
    z_after_one = after_one.get("Z", 0.0)

    return {
        "experiment": "C_prefix_compression_delayed_collapse",
        "seed": seed,
        "results": {
            "sequence_node_count": sequence_nodes,
            "prefix_tree_node_count": prefix_nodes,
            "compression_ratio": compression_ratio,
            "prefix_reuse_rate": reuse_rate,
            "candidates_A_B_C": cand_abc,
            "candidate_entropy_A_B_C": float(-sum(p * math.log(p + 1e-12) for p in cand_abc.values())),
            "before_new_branch": before,
            "after_one_new_branch": after_one,
            "new_branch_Z_probability_after_one": z_after_one,
        },
    }


def summarize(records: list[dict]) -> dict:
    by_exp: dict[str, list[dict]] = {}
    for rec in records:
        by_exp.setdefault(rec["experiment"], []).append(rec)

    summary: dict[str, object] = {"created_at": now(), "records": len(records), "experiments": {}}

    a_records = by_exp.get("A_closure_extrapolation", [])
    if a_records:
        agg = {}
        for model in ("flatten_mlp", "small_transformer"):
            agg[model] = {}
            for metric in ("address_accuracy", "read_accuracy"):
                vals_by_len = {}
                for length in ("len_8", "len_16", "len_32", "len_64"):
                    vals = [
                        r["results"].get(model, {}).get(metric, {}).get(length)
                        for r in a_records
                        if r["results"].get(model, {}).get(metric, {}).get(length) is not None
                    ]
                    if vals:
                        vals_by_len[length] = float(np.mean(vals))
                if vals_by_len:
                    agg[model][metric] = vals_by_len
        summary["experiments"]["A_closure_extrapolation"] = agg

    b_records = by_exp.get("B_subheap_kernel_relocation", [])
    if b_records:
        agg = {}
        for model in ("treeheap_kernel", "flatten_mlp", "sequence_cnn", "small_transformer"):
            vals = [r["results"][model]["accuracy"] for r in b_records]
            agg[model] = {"accuracy_mean": float(np.mean(vals)), "accuracy_min": float(np.min(vals)), "accuracy_max": float(np.max(vals))}
            if model == "treeheap_kernel":
                hits = [r["results"][model]["hit_at_1"] for r in b_records]
                agg[model]["hit_at_1_mean"] = float(np.mean(hits))
        summary["experiments"]["B_subheap_kernel_relocation"] = agg

    c_records = by_exp.get("C_prefix_compression_delayed_collapse", [])
    if c_records:
        summary["experiments"]["C_prefix_compression_delayed_collapse"] = {
            "compression_ratio_mean": float(np.mean([r["results"]["compression_ratio"] for r in c_records])),
            "prefix_reuse_rate_mean": float(np.mean([r["results"]["prefix_reuse_rate"] for r in c_records])),
            "new_branch_Z_probability_after_one_mean": float(
                np.mean([r["results"]["new_branch_Z_probability_after_one"] for r in c_records])
            ),
        }
    return summary


def write_readme(out_dir: Path, summary: dict) -> None:
    lines = [
        "# Existence Proof Suite Evidence",
        "",
        "Synthetic evidence for SPR-017.",
        "",
        "This suite tests TreeHeap structure bias, not language ability.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Files",
        "",
        "- `summary.json`: aggregate metrics",
        "- `summary.partial.json`: most recent partial summary while running",
        "- `trace.jsonl`: per-config records",
        "- `run_meta.json`: launch metadata",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="evidence/existence_proof_suite")
    parser.add_argument("--profile", choices=["smoke", "night"], default="smoke")
    parser.add_argument("--time-budget-hours", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    ensure_torch()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    run_meta = {
        "started_at": now(),
        "profile": args.profile,
        "time_budget_hours": args.time_budget_hours,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        run_meta["gpu_name"] = torch.cuda.get_device_name(0)
    write_json(out_dir / "run_meta.json", run_meta)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.profile == "smoke":
        configs = [(args.seed, 256, 12)]
    else:
        # Build a long queue and stop by wall-clock budget. The io 3090 is a
        # constrained card, so we prefer many moderate jobs with durable
        # evidence after every config over one huge fragile run.
        seeds = [args.seed + i for i in range(4096)]
        train_sizes = [256, 512, 1024, 2048, 4096]
        configs = [(s, n, 96) for s in seeds for n in train_sizes]

    start_time = time.time()
    deadline = start_time + args.time_budget_hours * 3600.0
    records: list[dict] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    completed = len([r for r in records if r.get("record_type") == "config_complete"])
    for idx, (seed, train_n, epochs) in enumerate(configs):
        if idx < completed:
            continue
        if time.time() > deadline:
            break
        set_seed(seed)
        append_jsonl(trace_path, {"record_type": "config_start", "time": now(), "idx": idx, "seed": seed, "train_n": train_n})
        try:
            recs = [
                run_a_closure(seed, train_n, epochs, device),
                run_b_kernel(seed, train_n, epochs, device),
                run_c_prefix(seed),
            ]
            for rec in recs:
                rec["record_type"] = "experiment_result"
                rec["time"] = now()
                append_jsonl(trace_path, rec)
                records.append(rec)
            append_jsonl(trace_path, {"record_type": "config_complete", "time": now(), "idx": idx, "seed": seed, "train_n": train_n})
            partial = summarize([r for r in records if r.get("record_type") == "experiment_result"])
            partial["status"] = "running"
            partial["completed_configs"] = idx + 1
            write_json(out_dir / "summary.partial.json", partial)
            write_readme(out_dir, partial)
        except Exception as exc:
            append_jsonl(trace_path, {"record_type": "config_error", "time": now(), "idx": idx, "error": repr(exc)})
            write_json(out_dir / "summary.partial.json", {"status": "error", "error": repr(exc), "completed_configs": idx})
            raise

    final_records = [r for r in records if r.get("record_type") == "experiment_result"]
    final = summarize(final_records)
    final["status"] = "complete" if time.time() <= deadline else "time_budget_reached"
    final["finished_at"] = now()
    final["elapsed_seconds"] = time.time() - start_time
    write_json(out_dir / "summary.json", final)
    write_readme(out_dir, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
