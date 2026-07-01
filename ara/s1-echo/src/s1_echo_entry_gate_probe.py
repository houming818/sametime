#!/usr/bin/env python3
"""SPR-041 controlled S1 echo entry-gate proof.

This is a deliberately small bridge from M0 operator proofs to S1 echo.

Question:
  Can gradient training learn token information and a structural read/collapse
  rule in one controlled TreeHeap-shaped echo task?

Task:
  Input:  four token ids and a task flag.
  Output: either the same sequence (identity echo) or the mirrored sequence.

Model:
  - learned token embeddings write token information into four TreeHeap leaves;
  - a small operation gate chooses between two read kernels;
  - each read kernel is a soft route over leaf addresses;
  - a shared decoder collapses read states back to token ids.

This proof does not claim language semantics, translation, or learned natural
mirror triggers. The trigger is the provided task flag. The claim is only that
the minimum S1 echo loop can be made differentiable and trainable:

  token -> TreeHeap leaves -> path-conditioned read kernel -> token collapse.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEQ_LEN = 4
PAD = 0
IDENTITY = torch.tensor([0, 1, 2, 3], dtype=torch.long)
MIRROR = torch.tensor([3, 2, 1, 0], dtype=torch.long)


def make_dataset(samples: int, vocab: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, vocab, size=(samples, SEQ_LEN), dtype=np.int64)
    tasks = rng.integers(0, 2, size=(samples,), dtype=np.int64)
    target = tokens.copy()
    mirror_idx = tasks == 1
    target[mirror_idx] = target[mirror_idx, ::-1]
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(tasks, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )


def split_dataset(tokens: torch.Tensor, tasks: torch.Tensor, targets: torch.Tensor, seed: int):
    rng = np.random.default_rng(seed)
    idx = torch.tensor(rng.permutation(tokens.shape[0]), dtype=torch.long)
    n_train = int(tokens.shape[0] * 0.75)
    n_test = int(tokens.shape[0] * 0.125)
    splits = {}
    for name, sel in {
        "train": idx[:n_train],
        "test": idx[n_train:n_train + n_test],
        "ood": idx[n_train + n_test:],
    }.items():
        splits[name] = (tokens[sel], tasks[sel], targets[sel])
    return splits


class TreeHeapEchoGate(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.decoder = nn.Linear(dim, vocab)
        self.task_gate = nn.Embedding(2, 2)
        self.route_logits = nn.Parameter(torch.zeros(2, SEQ_LEN, SEQ_LEN))
        with torch.no_grad():
            # Break symmetry but do not hand-code the final routes.
            self.route_logits.normal_(mean=0.0, std=0.03)
            self.task_gate.weight.normal_(mean=0.0, std=0.03)

    def forward(self, tokens: torch.Tensor, tasks: torch.Tensor):
        leaves = self.emb(tokens)
        op_prob = F.softmax(self.task_gate(tasks), dim=-1)
        routes = F.softmax(self.route_logits, dim=-1)
        mixed_routes = torch.einsum("bo,opq->bpq", op_prob, routes)
        read_states = torch.einsum("bpq,bqd->bpd", mixed_routes, leaves)
        return self.decoder(read_states), op_prob, routes


class NoTaskSingleKernel(nn.Module):
    """A control baseline with one route kernel and no task-conditioned gate."""

    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.decoder = nn.Linear(dim, vocab)
        self.route_logits = nn.Parameter(torch.zeros(SEQ_LEN, SEQ_LEN))
        with torch.no_grad():
            self.route_logits.normal_(mean=0.0, std=0.03)

    def forward(self, tokens: torch.Tensor, tasks: torch.Tensor):
        leaves = self.emb(tokens)
        routes = F.softmax(self.route_logits, dim=-1)
        read_states = torch.einsum("pq,bqd->bpd", routes, leaves)
        op_prob = torch.full((tokens.shape[0], 1), 1.0, device=tokens.device)
        return self.decoder(read_states), op_prob, routes.unsqueeze(0)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def train_model(model: nn.Module, train, epochs: int, batch: int, lr: float, seed: int, device: str):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tokens, tasks, targets = [x.to(device) for x in train]
    rng = np.random.default_rng(seed)
    trace = []
    n = tokens.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch):
            sel = torch.tensor(order[start:start + batch], dtype=torch.long, device=device)
            opt.zero_grad()
            logits, _, _ = model(tokens[sel], tasks[sel])
            loss = loss_fn(logits, targets[sel])
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * sel.numel()
            seen += sel.numel()
        if epoch in {0, 1, 2, 5, 10, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / seen})
    return model.cpu(), trace


def evaluate(model: nn.Module, split):
    model.eval()
    tokens, tasks, targets = split
    with torch.no_grad():
        logits, op_prob, routes = model(tokens, tasks)
        pred = logits.argmax(dim=-1)
    token_acc = float((pred == targets).float().mean())
    exact = float((pred == targets).all(dim=1).float().mean())
    route = routes.detach().cpu()
    expected = torch.stack(
        [
            F.one_hot(IDENTITY, SEQ_LEN).float(),
            F.one_hot(MIRROR, SEQ_LEN).float(),
        ],
        dim=0,
    )
    if route.shape[0] == 2:
        route_alignment = float((route * expected).sum(dim=-1).mean())
        route_argmax_ok = float((route.argmax(dim=-1) == torch.stack([IDENTITY, MIRROR])).float().mean())
        gate_echo = float(op_prob[tasks == 0, 0].mean()) if (tasks == 0).any() else math.nan
        gate_mirror = float(op_prob[tasks == 1, 1].mean()) if (tasks == 1).any() else math.nan
    else:
        route_alignment = float((route[0] * expected[0]).sum(dim=-1).mean())
        route_argmax_ok = float((route[0].argmax(dim=-1) == IDENTITY).float().mean())
        gate_echo = math.nan
        gate_mirror = math.nan
    return {
        "token_acc": token_acc,
        "exact": exact,
        "route_alignment": route_alignment,
        "route_argmax_ok": route_argmax_ok,
        "gate_echo_identity_prob": gate_echo,
        "gate_mirror_mirror_prob": gate_mirror,
    }


def run(args: argparse.Namespace):
    started = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokens, tasks, targets = make_dataset(args.samples, args.vocab, args.seed)
    splits = split_dataset(tokens, tasks, targets, args.seed + 1)

    model = TreeHeapEchoGate(args.vocab, args.dim)
    model, trace = train_model(model, splits["train"], args.epochs, args.batch, args.lr, args.seed + 2, args.device)
    baseline = NoTaskSingleKernel(args.vocab, args.dim)
    baseline, baseline_trace = train_model(
        baseline, splits["train"], args.epochs, args.batch, args.lr, args.seed + 3, args.device
    )

    metrics = {name: evaluate(model, split) for name, split in splits.items()}
    baseline_metrics = {name: evaluate(baseline, split) for name, split in splits.items()}

    pass_checks = {
        "treeheap_ood_token_acc_high": metrics["ood"]["token_acc"] >= args.min_token_acc,
        "treeheap_ood_exact_high": metrics["ood"]["exact"] >= args.min_exact,
        "routes_are_structural": metrics["ood"]["route_argmax_ok"] >= args.min_route_argmax,
        "gate_selects_identity_for_echo": metrics["ood"]["gate_echo_identity_prob"] >= args.min_gate_prob,
        "gate_selects_mirror_for_mirror": metrics["ood"]["gate_mirror_mirror_prob"] >= args.min_gate_prob,
        "no_task_baseline_worse": baseline_metrics["ood"]["exact"] <= args.max_baseline_exact,
    }

    summary = {
        "claim": "S1-ECHO-GATE-C01",
        "predict": "P-S1-ECHO041",
        "host": args.host_label,
        "device": args.device,
        "config": vars(args),
        "task": {
            "input": "four token ids plus task flag",
            "echo_target": "[t0,t1,t2,t3]",
            "mirror_target": "[t3,t2,t1,t0]",
            "treeheap_leaves": "four ordered leaf addresses",
        },
        "models": {
            "treeheap_echo_gate": {
                "parameters": sum(p.numel() for p in model.parameters()),
                "metrics": metrics,
                "trace": trace,
                "route_probabilities": F.softmax(model.route_logits, dim=-1).detach().tolist(),
                "task_gate_probabilities": F.softmax(model.task_gate.weight, dim=-1).detach().tolist(),
            },
            "no_task_single_kernel": {
                "parameters": sum(p.numel() for p in baseline.parameters()),
                "metrics": baseline_metrics,
                "trace": baseline_trace,
            },
        },
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": (
                "Controlled S1 echo can learn token write/read, structural route selection, "
                "and token collapse from scalar cross-entropy loss."
            ),
            "not_proved": [
                "not WMT translation",
                "not language semantics",
                "not unsupervised natural mirror trigger",
                "not recursive-depth mirror selection",
                "not superiority over all larger sequence models",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    return summary


def write_outputs(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["models"]["treeheap_echo_gate"]["trace"]:
            f.write(json.dumps({"model": "treeheap_echo_gate", **row}, ensure_ascii=False) + "\n")
        for row in summary["models"]["no_task_single_kernel"]["trace"]:
            f.write(json.dumps({"model": "no_task_single_kernel", **row}, ensure_ascii=False) + "\n")
    m = summary["models"]["treeheap_echo_gate"]["metrics"]["ood"]
    b = summary["models"]["no_task_single_kernel"]["metrics"]["ood"]
    readme = f"""# S1 Echo Entry Gate Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
treeheap_ood_token_acc = {m['token_acc']:.6f}
treeheap_ood_exact = {m['exact']:.6f}
treeheap_ood_route_argmax_ok = {m['route_argmax_ok']:.6f}
treeheap_ood_gate_echo_identity_prob = {m['gate_echo_identity_prob']:.6f}
treeheap_ood_gate_mirror_mirror_prob = {m['gate_mirror_mirror_prob']:.6f}
no_task_baseline_ood_exact = {b['exact']:.6f}
```

## Boundary

This is a controlled S1 entry proof. The task flag is given, so it does not
prove natural-language mirror trigger discovery or translation.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_echo_entry_gate_probe")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host-label", default="local")
    parser.add_argument("--min-token-acc", type=float, default=0.995)
    parser.add_argument("--min-exact", type=float, default=0.98)
    parser.add_argument("--min-route-argmax", type=float, default=1.0)
    parser.add_argument("--min-gate-prob", type=float, default=0.75)
    parser.add_argument("--max-baseline-exact", type=float, default=0.65)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
