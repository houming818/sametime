#!/usr/bin/env python3
"""A15 frozen TreeHeap embedding handoff gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def iter_english(path: Path) -> Iterable[Tuple[int, bytes, str]]:
    with path.open("rb") as handle:
        for raw_index, raw in enumerate(handle):
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError:
                continue
            if len(fields) >= 2 and fields[1].strip():
                yield raw_index, raw, fields[1]


def load_checkpoint(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "token_path_bits",
        "token_route_margin",
        "token_leaf",
        "token_context_sqrt_probability",
        "target_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"checkpoint missing fields: {missing}")
    return payload


def make_treeheap_coordinates(payload: Dict) -> torch.Tensor:
    bits = payload["token_path_bits"].to(torch.float32) * 2.0 - 1.0
    margins = payload["token_route_margin"].to(torch.float32)
    scale = margins.std(dim=0, unbiased=False).clamp_min(1e-6)
    margins = torch.tanh((margins - margins.mean(dim=0)) / scale)
    leaves = payload["token_leaf"].to(torch.long)
    leaf_count = int(leaves.max().item()) + 1
    one_hot = F.one_hot(leaves, num_classes=leaf_count).to(torch.float32)
    return F.normalize(torch.cat([bits, margins, one_hot], dim=1), dim=1)


def make_frequency_shuffle(coords: torch.Tensor, frequency: torch.Tensor, seed: int) -> torch.Tensor:
    order = torch.argsort(frequency, descending=True)
    permutation = torch.arange(coords.shape[0])
    generator = torch.Generator().manual_seed(seed)
    for chunk in order.split(max(8, coords.shape[0] // 8)):
        if chunk.numel() > 1:
            shuffled = chunk[torch.randperm(chunk.numel(), generator=generator)]
            permutation[chunk] = shuffled
    return coords[permutation]


def make_random(coords: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(coords.shape, generator=generator)
    return F.normalize(table, dim=1)


def make_context_projection(payload: Dict, width: int, seed: int) -> torch.Tensor:
    source = payload["token_context_sqrt_probability"].to(torch.float32)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(source.shape[1], width, generator=generator) / math.sqrt(source.shape[1])
    return F.normalize(source @ projection, dim=1)


def extract_sequences(
    data_path: Path,
    spm_path: Path,
    target_ids: Sequence[int],
    start_valid_line: int,
    scan_valid_lines: int,
    train_sequences: int,
    test_sequences: int,
    min_len: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    lookup = {int(token_id): index for index, token_id in enumerate(target_ids)}
    sequences: List[List[int]] = []
    region_digest = hashlib.sha256()
    valid_index = 0
    first_raw = None
    last_raw = None
    total_needed = train_sequences + test_sequences
    for raw_index, raw, text in iter_english(data_path):
        if valid_index < start_valid_line:
            valid_index += 1
            continue
        if valid_index >= start_valid_line + scan_valid_lines or len(sequences) >= total_needed:
            break
        region_digest.update(raw)
        if first_raw is None:
            first_raw = raw_index
        last_raw = raw_index
        ids = [lookup[token] for token in sp.encode(text, out_type=int) if token in lookup]
        if len(ids) >= min_len:
            sequences.append(ids[:max_len])
        valid_index += 1
    if len(sequences) < total_needed:
        raise RuntimeError(f"collected {len(sequences)} sequences, need {total_needed}")

    def pad(rows: Sequence[Sequence[int]]) -> torch.Tensor:
        out = torch.full((len(rows), max_len), -1, dtype=torch.long)
        for row_index, row in enumerate(rows):
            out[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        return out

    train = pad(sequences[:train_sequences])
    test = pad(sequences[train_sequences:total_needed])
    meta = {
        "start_valid_line": start_valid_line,
        "scan_valid_lines": scan_valid_lines,
        "first_raw_line": first_raw,
        "last_raw_line": last_raw,
        "region_sha256": region_digest.hexdigest(),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "min_len": min_len,
        "max_len": max_len,
        "train_mean_length": float((train >= 0).sum(dim=1).float().mean()),
        "test_mean_length": float((test >= 0).sum(dim=1).float().mean()),
    }
    return train, test, meta


class SequenceProbe(nn.Module):
    def __init__(self, table: torch.Tensor, hidden: int, trainable: bool):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(table.clone(), freeze=not trainable)
        self.rnn = nn.GRU(table.shape[1], hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, table.shape[0])

    def forward(self, tokens: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
        valid = tokens >= 0
        safe = tokens.clamp_min(0)
        states = self.embedding(safe)
        states = states * valid.unsqueeze(-1)
        if mask_center:
            lengths = valid.sum(dim=1)
            center = torch.div(lengths - 1, 2, rounding_mode="floor")
            states[torch.arange(states.shape[0], device=states.device), center] = 0.0
        encoded, _ = self.rnn(states)
        return self.head(encoded)


def batch_indices(size: int, batch: int, steps: int, seed: int) -> Iterable[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield torch.randint(size, (batch,), generator=generator)


def train_sequence_probe(
    table: torch.Tensor,
    trainable: bool,
    train_data: torch.Tensor,
    task: str,
    seed: int,
    steps: int,
    batch: int,
    hidden: int,
    lr: float,
    device: torch.device,
) -> Tuple[SequenceProbe, List[Dict], str, str]:
    set_seed(seed)
    model = SequenceProbe(table, hidden, trainable).to(device)
    before_hash = tensor_sha256(model.embedding.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trace: List[Dict] = []
    model.train()
    for step, indices in enumerate(batch_indices(len(train_data), batch, steps, seed + 17), start=1):
        tokens = train_data[indices].to(device)
        logits = model(tokens, mask_center=task == "masked")
        if task == "echo":
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1), ignore_index=-1)
        else:
            lengths = (tokens >= 0).sum(dim=1)
            center = torch.div(lengths - 1, 2, rounding_mode="floor")
            target = tokens[torch.arange(tokens.shape[0], device=device), center]
            center_logits = logits[torch.arange(tokens.shape[0], device=device), center]
            loss = F.cross_entropy(center_logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {task} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, max(1, steps // 4), max(1, steps // 2), max(1, 3 * steps // 4), steps}:
            trace.append({"step": step, "loss": float(loss.detach().cpu())})
    after_hash = tensor_sha256(model.embedding.weight)
    return model, trace, before_hash, after_hash


@torch.no_grad()
def evaluate_sequence(
    model: SequenceProbe,
    data: torch.Tensor,
    task: str,
    batch: int,
    device: torch.device,
    replacement: torch.Tensor | None = None,
) -> Dict[str, float]:
    original = None
    if replacement is not None:
        original = model.embedding.weight.detach().clone()
        model.embedding.weight.copy_(replacement.to(device))
    model.eval()
    loss_sum = 0.0
    loss_count = 0
    correct = 0
    total = 0
    exact = 0
    top5 = 0
    for start in range(0, len(data), batch):
        tokens = data[start : start + batch].to(device)
        logits = model(tokens, mask_center=task == "masked")
        if task == "echo":
            valid = tokens >= 0
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1), ignore_index=-1)
            prediction = logits.argmax(dim=-1)
            correct += int(((prediction == tokens) & valid).sum())
            total += int(valid.sum())
            exact += int((((prediction == tokens) | ~valid).all(dim=1)).sum())
            batch_loss_count = int(valid.sum())
        else:
            lengths = (tokens >= 0).sum(dim=1)
            center = torch.div(lengths - 1, 2, rounding_mode="floor")
            target = tokens[torch.arange(tokens.shape[0], device=device), center]
            center_logits = logits[torch.arange(tokens.shape[0], device=device), center]
            loss = F.cross_entropy(center_logits, target)
            prediction = center_logits.argmax(dim=-1)
            correct += int((prediction == target).sum())
            top5 += int((center_logits.topk(5, dim=-1).indices == target.unsqueeze(1)).any(dim=1).sum())
            total += len(tokens)
            batch_loss_count = len(tokens)
        loss_sum += float(loss.cpu()) * batch_loss_count
        loss_count += batch_loss_count
    if original is not None:
        model.embedding.weight.copy_(original)
    result = {"nll": loss_sum / loss_count, "top1": correct / total}
    if task == "echo":
        result["sequence_exact"] = exact / len(data)
    else:
        result["top5"] = top5 / total
    return result


def train_token_read(
    table: torch.Tensor,
    seed: int,
    steps: int,
    lr: float,
    noise_std: float,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    features = table.to(device)
    labels = torch.arange(table.shape[0], device=device)
    decoder = nn.Linear(table.shape[1], table.shape[0]).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    trace = []
    for step in range(1, steps + 1):
        logits = decoder(features)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite token-read loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, max(1, steps // 2), steps}:
            trace.append({"step": step, "loss": float(loss.detach().cpu())})
    with torch.no_grad():
        clean = float((decoder(features).argmax(dim=1) == labels).float().mean())
        generator = torch.Generator(device=device).manual_seed(seed + 901)
        noisy = F.normalize(features + noise_std * torch.randn(features.shape, generator=generator, device=device), dim=1)
        robust = float((decoder(noisy).argmax(dim=1) == labels).float().mean())
    return {"clean_top1": clean, "noise_top1": robust, "noise_std": noise_std, "trace": trace}


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--data", required=True)
    parser.add_argument("--spm-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="20260930,20260931,20260932")
    parser.add_argument("--start-valid-line", type=int, default=1100000)
    parser.add_argument("--scan-valid-lines", type=int, default=250000)
    parser.add_argument("--train-sequences", type=int, default=20000)
    parser.add_argument("--test-sequences", type=int, default=5000)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--read-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--read-lr", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for item in args.checkpoint:
        name, raw_path = item.split("=", 1)
        checkpoints[name] = Path(raw_path)
    payloads = {name: load_checkpoint(path) for name, path in checkpoints.items()}
    first = next(iter(payloads.values()))
    target_ids = [int(value) for value in first["target_ids"]]
    for name, payload in payloads.items():
        if [int(value) for value in payload["target_ids"]] != target_ids:
            raise RuntimeError(f"target vocabulary mismatch: {name}")

    train_data, test_data, data_meta = extract_sequences(
        Path(args.data), Path(args.spm_model), target_ids,
        args.start_valid_line, args.scan_valid_lines,
        args.train_sequences, args.test_sequences, args.min_len, args.max_len,
    )
    frequency = torch.bincount(train_data[train_data >= 0], minlength=len(target_ids)).to(torch.float32)
    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    summary: Dict = {
        "claim": "S1-F-HANDOFF-A15-C01",
        "format": "frozen_treeheap_embedding_handoff_gate_a15_v1",
        "contract": vars(args),
        "data": data_meta,
        "data_size_bytes": Path(args.data).stat().st_size,
        "spm_sha256": sha256(Path(args.spm_model)),
        "train_tensor_sha256": tensor_sha256(train_data),
        "test_tensor_sha256": tensor_sha256(test_data),
        "candidates": {},
    }

    for candidate_name, payload in payloads.items():
        native = make_treeheap_coordinates(payload)
        tables = {
            "treeheap_native": native,
            "frequency_shuffle": make_frequency_shuffle(native, frequency, 20260930),
            "random_fixed": make_random(native, 20260930),
            "context_projection": make_context_projection(payload, native.shape[1], 20260930),
        }
        candidate = {
            "checkpoint": str(checkpoints[candidate_name]),
            "checkpoint_sha256": sha256(checkpoints[candidate_name]),
            "coordinate_width": native.shape[1],
            "coordinate_sha256": {name: tensor_sha256(table) for name, table in tables.items()},
            "seeds": {},
        }
        for seed in seeds:
            seed_result = {"arms": {}}
            for arm_name in ["treeheap_native", "frequency_shuffle", "random_fixed", "context_projection", "learned_embedding"]:
                table = tables.get(arm_name, make_random(native, seed + 313))
                trainable = arm_name == "learned_embedding"
                arm = {"token_read": train_token_read(table, seed, args.read_steps, args.read_lr, args.noise_std, device)}
                for task in ["echo", "masked"]:
                    model, trace, before_hash, after_hash = train_sequence_probe(
                        table, trainable, train_data, task, seed, args.steps, args.batch,
                        args.hidden, args.lr, device,
                    )
                    result = evaluate_sequence(model, test_data, task, args.batch, device)
                    result["trace"] = trace
                    result["embedding_sha256_before"] = before_hash
                    result["embedding_sha256_after"] = after_hash
                    result["embedding_frozen_unchanged"] = trainable or before_hash == after_hash
                    if arm_name == "treeheap_native":
                        result["interventions"] = {
                            "frequency_shuffle": evaluate_sequence(
                                model, test_data, task, args.batch, device, tables["frequency_shuffle"]
                            ),
                            "zero": evaluate_sequence(
                                model, test_data, task, args.batch, device, torch.zeros_like(native)
                            ),
                        }
                    arm[task] = result
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                seed_result["arms"][arm_name] = arm
                print(json.dumps({
                    "candidate": candidate_name,
                    "seed": seed,
                    "arm": arm_name,
                    "echo_top1": arm["echo"]["top1"],
                    "masked_top1": arm["masked"]["top1"],
                }), flush=True)
            native_arm = seed_result["arms"]["treeheap_native"]
            echo_native = native_arm["echo"]["top1"]
            echo_shuffle = native_arm["echo"]["interventions"]["frequency_shuffle"]["top1"]
            echo_zero = native_arm["echo"]["interventions"]["zero"]["top1"]
            seed_result["mechanical_gate"] = {
                "finite": all(math.isfinite(seed_result["arms"][arm][task]["nll"])
                              for arm in seed_result["arms"] for task in ["echo", "masked"]),
                "frozen_unchanged": all(
                    seed_result["arms"][arm][task]["embedding_frozen_unchanged"]
                    for arm in ["treeheap_native", "frequency_shuffle", "random_fixed", "context_projection"]
                    for task in ["echo", "masked"]
                ),
                "token_read": native_arm["token_read"]["clean_top1"] >= 0.95,
                "echo": echo_native >= 0.80,
                "zero_drop": echo_native - echo_zero >= 0.50,
                "shuffle_drop": echo_native - echo_shuffle >= 0.25,
            }
            seed_result["mechanical_gate"]["pass"] = all(seed_result["mechanical_gate"].values())
            candidate["seeds"][str(seed)] = seed_result

        native_masked = [candidate["seeds"][str(seed)]["arms"]["treeheap_native"]["masked"]["top1"] for seed in seeds]
        random_masked = [candidate["seeds"][str(seed)]["arms"]["random_fixed"]["masked"]["top1"] for seed in seeds]
        shuffle_masked = [candidate["seeds"][str(seed)]["arms"]["frequency_shuffle"]["masked"]["top1"] for seed in seeds]
        candidate["aggregate"] = {
            "native_masked_top1_median": median(native_masked),
            "random_masked_top1_median": median(random_masked),
            "frequency_shuffle_masked_top1_median": median(shuffle_masked),
            "native_minus_random": median(native_masked) - median(random_masked),
            "native_minus_frequency_shuffle": median(native_masked) - median(shuffle_masked),
            "mechanical_gate_all_seeds": all(candidate["seeds"][str(seed)]["mechanical_gate"]["pass"] for seed in seeds),
        }
        candidate["aggregate"]["semantic_gate"] = (
            candidate["aggregate"]["native_minus_random"] >= 0.01
            and candidate["aggregate"]["native_minus_frequency_shuffle"] >= 0.01
        )
        summary["candidates"][candidate_name] = candidate

    summary["claim_result"] = {
        "mechanical_handoff_supported": any(
            candidate["aggregate"]["mechanical_gate_all_seeds"] for candidate in summary["candidates"].values()
        ),
        "semantic_handoff_supported": any(
            candidate["aggregate"]["semantic_gate"] for candidate in summary["candidates"].values()
        ),
    }
    output = out / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["claim_result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
