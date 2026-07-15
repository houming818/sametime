#!/usr/bin/env python3
"""Held-out causal audit of functional equivalence in WMT TreeHeap frontiers."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
from s3_wmt_frontier_bottleneck import FixedFrontier, LearnedFrontier, RandomFrontier


MODELS = {
    "learned_frontier": LearnedFrontier,
    "fixed_frontier": FixedFrontier,
    "random_frontier": RandomFrontier,
}


def fit_kmeans(x: torch.Tensor, groups: int, seed: int, steps: int = 40) -> torch.Tensor:
    """Small deterministic cosine k-means used only on the nomination split."""
    x = F.normalize(x.float(), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=generator)[:groups]].clone()
    for _ in range(steps):
        labels = (x @ centers.T).argmax(-1)
        updated = []
        for group in range(groups):
            members = x[labels == group]
            if members.numel() == 0:
                updated.append(x[torch.randint(x.shape[0], (1,), generator=generator)[0]])
            else:
                updated.append(F.normalize(members.mean(0), dim=0))
        next_centers = torch.stack(updated)
        if torch.allclose(next_centers, centers, atol=1e-5):
            centers = next_centers
            break
        centers = next_centers
    return centers


def collect_memory(model, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    memories: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for src, length, _, _ in loader:
            memory, mask = model.encode(src.to(device), length.to(device))
            memories.append(memory.cpu())
            masks.append(mask.cpu())
    return torch.cat(memories), torch.cat(masks)


def assign_groups(memory: torch.Tensor, centers: List[torch.Tensor]) -> torch.Tensor:
    labels = []
    for address, address_centers in enumerate(centers):
        x = F.normalize(memory[:, address].float(), dim=-1)
        labels.append((x @ address_centers.T).argmax(-1))
    return torch.stack(labels, dim=1)


def build_donors(
    test_memory: torch.Tensor,
    test_labels: torch.Tensor,
    bank_memory: torch.Tensor,
    bank_labels: torch.Tensor,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Choose same-group and distance-matched different-group donor states."""
    rng = random.Random(seed)
    count, addresses, _ = test_memory.shape
    chosen = {name: torch.empty_like(test_memory) for name in ("same", "different", "random")}
    distances = {name: torch.zeros((count, addresses)) for name in chosen}
    matched = torch.zeros((count, addresses), dtype=torch.bool)
    bank_norm = F.normalize(bank_memory.float(), dim=-1)
    test_norm = F.normalize(test_memory.float(), dim=-1)

    for row in range(count):
        for address in range(addresses):
            group = int(test_labels[row, address])
            same_ids = torch.where(bank_labels[:, address] == group)[0]
            different_ids = torch.where(bank_labels[:, address] != group)[0]
            if same_ids.numel() == 0 or different_ids.numel() == 0:
                raise RuntimeError("empty donor group; lower --groups")

            same_distance_all = 1.0 - bank_norm[same_ids, address] @ test_norm[row, address]
            different_distance_all = 1.0 - bank_norm[different_ids, address] @ test_norm[row, address]
            pair_error = (same_distance_all[:, None] - different_distance_all[None, :]).abs()
            nearest_error, nearest_different = pair_error.min(dim=1)
            eligible = torch.where(nearest_error <= 0.01)[0]
            if eligible.numel() > 0:
                same_offset = int(eligible[rng.randrange(eligible.numel())])
                matched[row, address] = True
            else:
                same_offset = int(nearest_error.argmin())
            different_offset = int(nearest_different[same_offset])
            same_id = int(same_ids[same_offset])
            different_id = int(different_ids[different_offset])
            same_distance = float(same_distance_all[same_offset])
            random_id = rng.randrange(bank_memory.shape[0])

            for name, donor_id in (("same", same_id), ("different", different_id), ("random", random_id)):
                chosen[name][row, address] = bank_memory[donor_id, address]
                distances[name][row, address] = 1.0 - (
                    test_norm[row, address] * bank_norm[donor_id, address]
                ).sum()
    chosen.update({f"{name}_distance": value for name, value in distances.items()})
    chosen["matched"] = matched
    return chosen


def token_losses(logits: torch.Tensor, target: torch.Tensor, pad: int) -> torch.Tensor:
    losses = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=pad, reduction="none")
    valid = target.ne(pad)
    return (losses * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def bootstrap_interval(values: List[float], seed: int, rounds: int = 2000) -> Tuple[float, float]:
    x = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(rounds):
        index = torch.randint(x.numel(), (x.numel(),), generator=generator)
        means.append(x[index].mean())
    q = torch.quantile(torch.stack(means), torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(q[0]), float(q[1])


def audit_model(name: str, checkpoint_path: Path, cfg, rows, pieces: int, sp, args) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = MODELS[name](pieces + 1, cfg.dim, cfg.hidden).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    nominate = rows[cfg.train_samples:cfg.train_samples + cfg.valid_samples]
    test = rows[cfg.train_samples + cfg.valid_samples:]
    collate = base.collate(pieces)
    nomination_loader = DataLoader(base.ParallelDataset(nominate), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(base.ParallelDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    bank_memory, bank_mask = collect_memory(model, nomination_loader, args.device)
    test_memory, test_mask = collect_memory(model, test_loader, args.device)
    if not bool(bank_mask.all() and test_mask.all()):
        raise RuntimeError("frontier audit requires four live states for every sentence")

    centers = [fit_kmeans(bank_memory[:, address], args.groups, args.seed + address) for address in range(4)]
    bank_labels = assign_groups(bank_memory, centers)
    test_labels = assign_groups(test_memory, centers)
    donors = build_donors(test_memory, test_labels, bank_memory, bank_labels, args.seed + 101)

    per_example = {mode: [] for mode in ("baseline", "same", "different", "random")}
    kl_values = {mode: [] for mode in ("same", "different", "random")}
    flip_values = {mode: [] for mode in ("same", "different", "random")}
    cursor = 0
    with torch.no_grad():
        for src, length, target, _ in test_loader:
            size = src.shape[0]
            target = target.to(args.device)
            memory = test_memory[cursor:cursor + size].to(args.device)
            mask = torch.ones((size, 4), dtype=torch.bool, device=args.device)
            baseline_logits = model.decoder.teacher(memory, mask, target, sp.bos_id())
            baseline_loss = token_losses(baseline_logits, target, pieces)
            baseline_logp = F.log_softmax(baseline_logits, dim=-1)
            baseline_p = baseline_logp.exp()
            valid = target.ne(pieces)

            address = torch.arange(cursor, cursor + size) % 4
            local_index = torch.arange(size)
            matched = donors["matched"][cursor:cursor + size][local_index, address]
            per_example["baseline"].extend(baseline_loss.cpu()[matched].tolist())
            for mode in ("same", "different", "random"):
                changed = memory.clone()
                donor = donors[mode][cursor:cursor + size].to(args.device)
                batch_index = torch.arange(size, device=args.device)
                changed[batch_index, address.to(args.device)] = donor[batch_index, address.to(args.device)]
                logits = model.decoder.teacher(changed, mask, target, sp.bos_id())
                loss = token_losses(logits, target, pieces)
                per_example[mode].extend(loss.cpu()[matched].tolist())
                logp = F.log_softmax(logits, dim=-1)
                kl = (baseline_p * (baseline_logp - logp)).sum(-1)
                kl = (kl * valid).sum(-1) / valid.sum(-1).clamp_min(1)
                kl_values[mode].extend(kl.cpu()[matched].tolist())
                flips = ((baseline_logits.argmax(-1) != logits.argmax(-1)) & valid).sum(-1) / valid.sum(-1).clamp_min(1)
                flip_values[mode].extend(flips.cpu()[matched].tolist())
            cursor += size

    baseline = torch.tensor(per_example["baseline"])
    deltas = {mode: torch.tensor(per_example[mode]) - baseline for mode in ("same", "different", "random")}
    gap = (deltas["different"] - deltas["same"]).tolist()
    ci = bootstrap_interval(gap, args.seed + 202)
    used_addresses = torch.arange(test_memory.shape[0]) % 4
    matched_examples = donors["matched"][torch.arange(test_memory.shape[0]), used_addresses]
    distance = {}
    for mode in ("same", "different", "random"):
        selected_distance = donors[f"{mode}_distance"][torch.arange(test_memory.shape[0]), used_addresses]
        distance[mode] = float(selected_distance[matched_examples].mean())

    return {
        "checkpoint": str(checkpoint_path),
        "test_examples": len(test),
        "distance_matched_examples": int(matched_examples.sum()),
        "distance_match_coverage": float(matched_examples.float().mean()),
        "groups_per_address": args.groups,
        "baseline_nll": float(baseline.mean()),
        "interventions": {
            mode: {
                "nll": float(torch.tensor(per_example[mode]).mean()),
                "nll_delta": float(deltas[mode].mean()),
                "mean_kl": float(torch.tensor(kl_values[mode]).mean()),
                "argmax_flip_rate": float(torch.tensor(flip_values[mode]).mean()),
                "cosine_distance": distance[mode],
            }
            for mode in ("same", "different", "random")
        },
        "different_minus_same_nll_delta": float(torch.tensor(gap).mean()),
        "different_minus_same_bootstrap_95": list(ci),
        "distance_match_error": abs(distance["different"] - distance["same"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=71501)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    started = time.time()
    checkpoint_dir = Path(args.checkpoint_dir)
    first = torch.load(checkpoint_dir / "checkpoint_best_learned_frontier.pt", map_location="cpu", weights_only=False)
    cfg = base.Config(**first["config"])
    cfg.device = args.device
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    rows, pieces = base.load_rows(cfg, sp)

    results = {}
    for name in MODELS:
        print(f"[{name}] auditing held-out TreeHeap exchanges", flush=True)
        results[name] = audit_model(
            name,
            checkpoint_dir / f"checkpoint_best_{name}.pt",
            cfg,
            rows,
            pieces,
            sp,
            args,
        )
        print(json.dumps(results[name], indent=2), flush=True)

    learned_gap = results["learned_frontier"]["different_minus_same_nll_delta"]
    control_gap = max(
        results["fixed_frontier"]["different_minus_same_nll_delta"],
        results["random_frontier"]["different_minus_same_nll_delta"],
    )
    learned_ci = results["learned_frontier"]["different_minus_same_bootstrap_95"]
    gates = {
        "P1_distance_matched": (
            results["learned_frontier"]["distance_match_error"] <= 0.02
            and results["learned_frontier"]["distance_match_coverage"] >= 0.5
        ),
        "P2_learned_functional_gap": learned_gap > 0.02 and learned_ci[0] > 0.0,
        "P3_stronger_than_tree_controls": learned_gap - control_gap > 0.01,
    }
    summary = {
        "claim": "S2-TREEHEAP-FUNCTIONAL-EQUIV-C01",
        "terminology": "TreeHeap/subheap only; functional equivalence is a relation, not a new object type",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "source_checkpoint_config": first["config"],
        "models": results,
        "gates": gates,
        "decision": "supported" if all(gates.values()) else "partial" if gates["P2_learned_functional_gap"] else "not_supported",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("README.md").write_text(
        "# WMT TreeHeap Functional Equivalence\n\n```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
