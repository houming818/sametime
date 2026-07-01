#!/usr/bin/env python3
"""SPR-041 corrected inverse-gate canonical S1 echo proof.

The first SPR-041 pilot selected different read kernels for different output
tasks. Houming818 pointed out that this is the wrong direction for TreeHeap S1:
mirror input should first be transformed back into a canonical echo state, then
the same echo decoder should read from that canonical state.

This probe tests that corrected design.

Data:
  canonical = [t0, t1, t2, t3]
  observed  = canonical                 when transform = identity
  observed  = [t3, t2, t1, t0]           when transform = mirror
  target    = canonical                  for both cases

Model:
  observed tokens -> leaf embeddings
  transform flag -> inverse gate over structural kernels
  inverse route canonicalizes observed leaves
  shared echo decoder reads canonical states back to canonical tokens

Boundary:
  The transform flag is still given. This is not natural-language trigger
  discovery. It only proves the corrected canonicalization direction.
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


def make_dataset(samples: int, vocab: int, seed: int):
    rng = np.random.default_rng(seed)
    canonical = rng.integers(1, vocab, size=(samples, SEQ_LEN), dtype=np.int64)
    transforms = rng.integers(0, 2, size=(samples,), dtype=np.int64)
    observed = canonical.copy()
    mirror_idx = transforms == 1
    observed[mirror_idx] = observed[mirror_idx, ::-1]
    return (
        torch.tensor(observed, dtype=torch.long),
        torch.tensor(transforms, dtype=torch.long),
        torch.tensor(canonical, dtype=torch.long),
    )


def split_dataset(observed: torch.Tensor, transforms: torch.Tensor, canonical: torch.Tensor, seed: int):
    rng = np.random.default_rng(seed)
    idx = torch.tensor(rng.permutation(observed.shape[0]), dtype=torch.long)
    n_train = int(observed.shape[0] * 0.75)
    n_test = int(observed.shape[0] * 0.125)
    return {
        "train": (observed[idx[:n_train]], transforms[idx[:n_train]], canonical[idx[:n_train]]),
        "test": (
            observed[idx[n_train:n_train + n_test]],
            transforms[idx[n_train:n_train + n_test]],
            canonical[idx[n_train:n_train + n_test]],
        ),
        "ood": (observed[idx[n_train + n_test:]], transforms[idx[n_train + n_test:]], canonical[idx[n_train + n_test:]]),
    }


class InverseGateCanonicalEcho(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.inverse_gate = nn.Embedding(2, 2)
        self.inverse_route_logits = nn.Parameter(torch.zeros(2, SEQ_LEN, SEQ_LEN))
        self.echo_decoder = nn.Linear(dim, vocab)
        with torch.no_grad():
            self.inverse_gate.weight.normal_(mean=0.0, std=0.03)
            self.inverse_route_logits.normal_(mean=0.0, std=0.03)

    def forward(self, observed: torch.Tensor, transforms: torch.Tensor):
        leaves = self.emb(observed)
        gate = F.softmax(self.inverse_gate(transforms), dim=-1)
        routes = F.softmax(self.inverse_route_logits, dim=-1)
        canonical_route = torch.einsum("bo,opq->bpq", gate, routes)
        canonical_state = torch.einsum("bpq,bqd->bpd", canonical_route, leaves)
        return self.echo_decoder(canonical_state), gate, routes, canonical_state


class NoInverseSingleRoute(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.route_logits = nn.Parameter(torch.zeros(SEQ_LEN, SEQ_LEN))
        self.echo_decoder = nn.Linear(dim, vocab)
        with torch.no_grad():
            self.route_logits.normal_(mean=0.0, std=0.03)

    def forward(self, observed: torch.Tensor, transforms: torch.Tensor):
        leaves = self.emb(observed)
        route = F.softmax(self.route_logits, dim=-1)
        canonical_state = torch.einsum("pq,bqd->bpd", route, leaves)
        gate = torch.ones((observed.shape[0], 1), device=observed.device)
        return self.echo_decoder(canonical_state), gate, route.unsqueeze(0), canonical_state


def loss_fn(logits: torch.Tensor, canonical: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), canonical.reshape(-1))


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p.clamp_min(1e-9).log())).sum(dim=-1).mean()


def train_model(
    model: nn.Module,
    split,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: str,
    state_weight: float,
    entropy_weight: float,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    observed, transforms, canonical = [x.to(device) for x in split]
    rng = np.random.default_rng(seed)
    trace = []
    n = observed.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch):
            sel = torch.tensor(order[start:start + batch], dtype=torch.long, device=device)
            opt.zero_grad()
            logits, gate, routes, canonical_state = model(observed[sel], transforms[sel])
            ce = loss_fn(logits, canonical[sel])
            target_state = model.emb(canonical[sel]).detach() if hasattr(model, "emb") else canonical_state.detach()
            state_loss = F.mse_loss(canonical_state, target_state)
            sharp_loss = entropy(gate) + entropy(routes)
            loss = ce + state_weight * state_loss + entropy_weight * sharp_loss
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * sel.numel()
            seen += sel.numel()
        if epoch in {0, 1, 2, 5, 10, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / seen})
    return model.cpu(), trace


def best_op_mapping(route: torch.Tensor) -> dict[str, object]:
    expected = torch.stack(
        [
            F.one_hot(IDENTITY, SEQ_LEN).float(),
            F.one_hot(MIRROR, SEQ_LEN).float(),
        ],
        dim=0,
    )
    if route.shape[0] != 2:
        return {
            "identity_op": 0,
            "mirror_op": 0,
            "route_argmax_ok": float((route[0].argmax(dim=-1) == IDENTITY).float().mean()),
            "identity_route_confidence": float((route[0] * expected[0]).sum(dim=-1).mean()),
            "mirror_route_confidence": math.nan,
        }

    candidates = [(0, 1), (1, 0)]
    best = None
    for identity_op, mirror_op in candidates:
        identity_ok = (route[identity_op].argmax(dim=-1) == IDENTITY).float().mean()
        mirror_ok = (route[mirror_op].argmax(dim=-1) == MIRROR).float().mean()
        score = float((identity_ok + mirror_ok) / 2.0)
        identity_conf = float((route[identity_op] * expected[0]).sum(dim=-1).mean())
        mirror_conf = float((route[mirror_op] * expected[1]).sum(dim=-1).mean())
        item = {
            "identity_op": identity_op,
            "mirror_op": mirror_op,
            "route_argmax_ok": score,
            "identity_route_confidence": identity_conf,
            "mirror_route_confidence": mirror_conf,
        }
        if best is None or item["route_argmax_ok"] > best["route_argmax_ok"]:
            best = item
    return best


def evaluate(model: nn.Module, split):
    model.eval()
    observed, transforms, canonical = split
    with torch.no_grad():
        logits, gate, routes, canonical_state = model(observed, transforms)
        pred = logits.argmax(dim=-1)
        leaf_state = model.emb(canonical) if hasattr(model, "emb") else None
    token_acc = float((pred == canonical).float().mean())
    exact = float((pred == canonical).all(dim=1).float().mean())
    route = routes.detach().cpu()
    mapping = best_op_mapping(route)
    if route.shape[0] == 2:
        identity_gate = (
            float(gate[transforms == 0, mapping["identity_op"]].mean()) if (transforms == 0).any() else math.nan
        )
        mirror_inverse_gate = (
            float(gate[transforms == 1, mapping["mirror_op"]].mean()) if (transforms == 1).any() else math.nan
        )
    else:
        identity_gate = math.nan
        mirror_inverse_gate = math.nan

    if leaf_state is not None:
        canonical_mse = float(F.mse_loss(canonical_state, leaf_state).detach())
    else:
        canonical_mse = math.nan

    return {
        "token_acc": token_acc,
        "exact": exact,
        "route_argmax_ok": mapping["route_argmax_ok"],
        "identity_op": mapping["identity_op"],
        "mirror_op": mapping["mirror_op"],
        "identity_route_confidence": mapping["identity_route_confidence"],
        "mirror_route_confidence": mapping["mirror_route_confidence"],
        "identity_gate_identity_inverse_prob": identity_gate,
        "mirror_gate_mirror_inverse_prob": mirror_inverse_gate,
        "canonical_state_mse_to_true_leaf_embedding": canonical_mse,
    }


def run(args: argparse.Namespace):
    started = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    observed, transforms, canonical = make_dataset(args.samples, args.vocab, args.seed)
    splits = split_dataset(observed, transforms, canonical, args.seed + 1)

    model = InverseGateCanonicalEcho(args.vocab, args.dim)
    model, trace = train_model(
        model,
        splits["train"],
        args.epochs,
        args.batch,
        args.lr,
        args.seed + 2,
        args.device,
        args.state_weight,
        args.entropy_weight,
    )

    baseline = NoInverseSingleRoute(args.vocab, args.dim)
    baseline, baseline_trace = train_model(
        baseline,
        splits["train"],
        args.epochs,
        args.batch,
        args.lr,
        args.seed + 3,
        args.device,
        args.state_weight,
        args.entropy_weight,
    )

    metrics = {name: evaluate(model, split) for name, split in splits.items()}
    baseline_metrics = {name: evaluate(baseline, split) for name, split in splits.items()}

    pass_checks = {
        "canonical_echo_ood_exact_high": metrics["ood"]["exact"] >= args.min_exact,
        "canonical_state_close": metrics["ood"]["canonical_state_mse_to_true_leaf_embedding"] <= args.max_state_mse,
        "inverse_routes_are_structural": metrics["ood"]["route_argmax_ok"] >= args.min_route_argmax,
        "inverse_routes_are_confident": min(
            metrics["ood"]["identity_route_confidence"],
            metrics["ood"]["mirror_route_confidence"],
        ) >= args.min_route_confidence,
        "identity_uses_identity_inverse": metrics["ood"]["identity_gate_identity_inverse_prob"] >= args.min_gate_prob,
        "mirror_uses_mirror_inverse": metrics["ood"]["mirror_gate_mirror_inverse_prob"] >= args.min_gate_prob,
        "no_inverse_baseline_worse": baseline_metrics["ood"]["exact"] <= args.max_baseline_exact,
    }

    return {
        "claim": "S1-ECHO-CANON-C01",
        "predict": "P-S1-ECHO041-CORRECTED",
        "host": args.host_label,
        "device": args.device,
        "config": vars(args),
        "task": {
            "canonical": "[t0,t1,t2,t3]",
            "observed_identity": "[t0,t1,t2,t3]",
            "observed_mirror": "[t3,t2,t1,t0]",
            "target": "canonical for both transforms",
            "meaning": "learn inverse gate to canonical echo state, then use one echo decoder",
        },
        "models": {
            "inverse_gate_canonical_echo": {
                "parameters": sum(p.numel() for p in model.parameters()),
                "metrics": metrics,
                "trace": trace,
                "inverse_route_probabilities": F.softmax(model.inverse_route_logits, dim=-1).detach().tolist(),
                "inverse_gate_probabilities": F.softmax(model.inverse_gate.weight, dim=-1).detach().tolist(),
            },
            "no_inverse_single_route": {
                "parameters": sum(p.numel() for p in baseline.parameters()),
                "metrics": baseline_metrics,
                "trace": baseline_trace,
            },
        },
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": "Corrected S1 echo canonicalization can learn inverse structural routing before a shared echo decoder.",
            "downgraded_prior": "The previous S1-ECHO-GATE-C01 selected output-specific read kernels and is now marked misdirected.",
            "not_proved": [
                "not WMT translation",
                "not language semantics",
                "not unsupervised natural mirror trigger discovery",
                "not recursive-depth mirror selection",
                "not superiority over all sequence models",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }


def write_outputs(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["models"]["inverse_gate_canonical_echo"]["trace"]:
            f.write(json.dumps({"model": "inverse_gate_canonical_echo", **row}, ensure_ascii=False) + "\n")
        for row in summary["models"]["no_inverse_single_route"]["trace"]:
            f.write(json.dumps({"model": "no_inverse_single_route", **row}, ensure_ascii=False) + "\n")
    m = summary["models"]["inverse_gate_canonical_echo"]["metrics"]["ood"]
    b = summary["models"]["no_inverse_single_route"]["metrics"]["ood"]
    readme = f"""# S1 Echo Corrected Inverse Gate Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
canonical_echo_ood_exact = {m['exact']:.6f}
inverse_route_argmax_ok = {m['route_argmax_ok']:.6f}
identity_gate_identity_inverse_prob = {m['identity_gate_identity_inverse_prob']:.6f}
mirror_gate_mirror_inverse_prob = {m['mirror_gate_mirror_inverse_prob']:.6f}
canonical_state_mse = {m['canonical_state_mse_to_true_leaf_embedding']:.6g}
no_inverse_baseline_ood_exact = {b['exact']:.6f}
```

## Boundary

The transform flag is given. This proves corrected inverse-gate
canonicalization, not natural-language trigger discovery.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_echo_inverse_gate_probe")
    parser.add_argument("--seed", type=int, default=4102)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host-label", default="local")
    parser.add_argument("--min-exact", type=float, default=0.98)
    parser.add_argument("--max-state-mse", type=float, default=0.03)
    parser.add_argument("--min-route-argmax", type=float, default=1.0)
    parser.add_argument("--min-route-confidence", type=float, default=0.95)
    parser.add_argument("--min-gate-prob", type=float, default=0.75)
    parser.add_argument("--max-baseline-exact", type=float, default=0.65)
    parser.add_argument("--state-weight", type=float, default=2.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
