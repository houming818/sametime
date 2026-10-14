#!/usr/bin/env python3
"""Matched full-only versus depth-aligned multiresolution TreeHeap training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import socket
import sys
import time
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_treeheap_butterfly_wmt import ButterflyRecursive  # noqa: E402


CLAIM = "S3-MULTIRES-CURRICULUM-A01-R2-C01"
SCALES = ("core", "short", "reduced", "full")
NEGATIONS = ("不", "未", "无", "没有", "不能", "并非", "否", "拒绝", "禁止")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_rows(path: Path, seed: int, max_items: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: stable_int(f"select:{seed}:{row['source_line']}"))
    if max_items:
        rows = rows[:max_items]
    for row in rows:
        slot = stable_int(f"split:{seed}:{row['source_line']}") % 10
        row["split"] = "test" if slot == 0 else "valid" if slot == 1 else "train"
    if any(not any(row["split"] == split for row in rows) for split in ("train", "valid", "test")):
        raise RuntimeError("deterministic split produced an empty partition")
    return rows


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?%?", text))


def has_negation(text: str) -> bool:
    return any(marker in text for marker in NEGATIONS)


@torch.no_grad()
def reranker_audit(rows: list[dict], model_path: Path, batch_size: int, device: str) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    ordered = sorted(rows, key=lambda row: row["source_line"])
    shifted = ordered[1:] + ordered[:1]
    pairs: list[list[str]] = []
    labels: list[tuple[str, str]] = []
    for scale in SCALES[:-1]:
        key = f"target_zh_{scale}"
        for row, other in zip(ordered, shifted):
            pairs.append([row["target_zh_full"], row[key]])
            labels.append((scale, "aligned"))
            pairs.append([row["target_zh_full"], other[key]])
            labels.append((scale, "shuffled"))
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = tokenizer(
            pairs[start : start + batch_size], padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits.float().reshape(-1)
        scores.extend(logits.cpu().tolist())
    grouped = {scale: {kind: [] for kind in ("aligned", "shuffled")} for scale in SCALES[:-1]}
    for (scale, kind), score in zip(labels, scores):
        grouped[scale][kind].append(score)
    result = {}
    for scale, values in grouped.items():
        aligned, shuffled = values["aligned"], values["shuffled"]
        deltas = [a - b for a, b in zip(aligned, shuffled)]
        result[scale] = {
            "aligned_mean_logit": sum(aligned) / len(aligned),
            "shuffled_mean_logit": sum(shuffled) / len(shuffled),
            "paired_delta_mean": sum(deltas) / len(deltas),
            "aligned_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def invariant_audit(rows: list[dict]) -> dict:
    result = {}
    for scale in SCALES[:-1]:
        key = f"target_zh_{scale}"
        number_total = number_kept = negation_total = negation_kept = 0
        for row in rows:
            numbers = extract_numbers(row["target_zh_full"])
            if numbers:
                number_total += len(numbers)
                number_kept += sum(number in row[key] for number in numbers)
            if has_negation(row["target_zh_full"]):
                negation_total += 1
                negation_kept += int(has_negation(row[key]))
        result[scale] = {
            "number_items": number_total,
            "number_recall": number_kept / max(1, number_total),
            "negated_rows": negation_total,
            "negation_recall": negation_kept / max(1, negation_total),
        }
    return result


def encode_rows(rows: list[dict], processor, direction_id: int, eos: int, max_content: int) -> list[dict]:
    encoded = []
    for row in rows:
        source = [direction_id, *processor.encode(row["source_en"], out_type=int), eos]
        targets = {
            scale: [*processor.encode(row[f"target_zh_{scale}"], out_type=int), eos]
            for scale in SCALES
        }
        if len(source) > max_content + 2 or any(len(target) > max_content + 1 for target in targets.values()):
            continue
        encoded.append({**row, "source_ids": source, "target_ids": targets})
    return encoded


def source_width(row: dict) -> int:
    needed = max(2, len(row["source_ids"]))
    return max(32, min(256, 1 << (needed - 1).bit_length()))


def batches(rows: list[dict], batch_size: int, seed: int, shuffle: bool):
    groups = {32: [], 64: [], 128: [], 256: []}
    local = rows[:]
    if shuffle:
        random.Random(seed).shuffle(local)
    for row in local:
        group = groups[source_width(row)]
        group.append(row)
        if len(group) == batch_size:
            yield group[:]
            group.clear()
    for width in groups:
        if groups[width]:
            yield groups[width]


def collate(batch: list[dict], pad: int, device: str):
    source_max = max(len(row["source_ids"]) for row in batch)
    source = torch.full((len(batch), source_max), pad, dtype=torch.long, device=device)
    length = torch.tensor([len(row["source_ids"]) for row in batch], dtype=torch.long, device=device)
    targets = {}
    for index, row in enumerate(batch):
        source[index, : len(row["source_ids"])] = torch.tensor(row["source_ids"], device=device)
    for scale in SCALES:
        target_max = max(len(row["target_ids"][scale]) for row in batch)
        target = torch.full((len(batch), target_max), pad, dtype=torch.long, device=device)
        for index, row in enumerate(batch):
            ids = row["target_ids"][scale]
            target[index, : len(ids)] = torch.tensor(ids, device=device)
        targets[scale] = target
    return source, length, targets


def scale_depths(level_count: int) -> dict[str, int]:
    end = level_count - 1
    values = [0, round(end / 3), round(2 * end / 3), end]
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + 1)
    values[-1] = end
    return dict(zip(SCALES, values))


def read_exact_depth(decoder, hidden: torch.Tensor, levels, masks, depth: int) -> torch.Tensor:
    depth = min(depth, len(levels) - 1)
    active = masks[0].to(hidden.dtype)
    for current in range(depth):
        nodes = levels[current]
        children = levels[current + 1].reshape(nodes.shape[0], nodes.shape[1], 2, nodes.shape[2])
        child_valid = masks[current + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        scores = (decoder.branch(hidden)[:, None, None] * children).sum(-1) / math.sqrt(nodes.shape[-1])
        probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
        active = (active[:, :, None] * probability).reshape(nodes.shape[0], -1)
    return (active[:, :, None] * levels[depth]).sum(1)


def teacher_exact_depth(decoder, levels, masks, target: torch.Tensor, bos: int, depth: int):
    hidden = levels[0].new_zeros((levels[0].shape[0], decoder.hidden))
    previous = torch.full((levels[0].shape[0],), bos, device=levels[0].device, dtype=torch.long)
    logits = []
    for step in range(target.shape[1]):
        context = read_exact_depth(decoder, hidden, levels, masks, depth)
        hidden = decoder.cell(torch.cat((decoder.embedding(previous), context), dim=-1), hidden)
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        previous = target[:, step]
    return torch.stack(logits, dim=1)


@torch.no_grad()
def greedy_exact_depth(decoder, levels, masks, bos: int, eos: int, max_len: int, depth: int):
    hidden = levels[0].new_zeros((levels[0].shape[0], decoder.hidden))
    previous = torch.full((levels[0].shape[0],), bos, device=levels[0].device, dtype=torch.long)
    done = torch.zeros(levels[0].shape[0], device=levels[0].device, dtype=torch.bool)
    output = []
    for _ in range(max_len):
        context = read_exact_depth(decoder, hidden, levels, masks, depth)
        hidden = decoder.cell(torch.cat((decoder.embedding(previous), context), dim=-1), hidden)
        previous = decoder.output(torch.cat((hidden, context), dim=-1)).argmax(-1)
        output.append(previous)
        done |= previous.eq(eos)
        if bool(done.all()):
            break
    return torch.stack(output, dim=1)


def clean(ids: list[int], eos: int, pieces: int) -> list[int]:
    result = []
    for token in ids:
        if token == eos:
            break
        if 0 <= token < pieces:
            result.append(token)
    return result


def finite_token_ce(logits: torch.Tensor, target: torch.Tensor, pad: int) -> torch.Tensor:
    """Return finite non-pad token losses without an FP32 reduction overflow."""
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        ignore_index=pad, reduction="none",
    )
    selected = per_token[target.reshape(-1).ne(pad)]
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise RuntimeError("non-finite or empty token cross entropy")
    return selected


def prefix_control_target(targets: dict[str, torch.Tensor], scale: str, pad: int, eos: int) -> torch.Tensor:
    """Length-match a compressed target with a raw full-sentence prefix plus EOS."""
    reference_lengths = targets[scale].ne(pad).sum(1)
    width = int(reference_lengths.max().item())
    output = torch.full(
        (targets["full"].shape[0], width), pad,
        dtype=targets["full"].dtype, device=targets["full"].device,
    )
    for index, length_tensor in enumerate(reference_lengths):
        length = int(length_tensor.item())
        if length > 1:
            output[index, : length - 1] = targets["full"][index, : length - 1]
        output[index, length - 1] = eos
    return output


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_parameter_audit(payload: dict, processor, rows: list[dict], pieces: int) -> dict:
    state = payload["state_dict"]
    tensor_maxima = []
    for name, value in state.items():
        if torch.is_floating_point(value):
            tensor_maxima.append({"name": name, "max_abs": float(value.detach().abs().max())})
    tensor_maxima.sort(key=lambda row: row["max_abs"], reverse=True)

    embedding = state["encoder.embedding.weight"].detach()
    row_max = embedding.abs().amax(dim=1)
    top_values, top_ids = torch.topk(row_max, min(32, row_max.numel()))
    occurrences: dict[int, list[int]] = {}
    for row in rows:
        for token_id in set(row["source_ids"]):
            occurrences.setdefault(token_id, []).append(row["source_line"])

    top_embedding_rows = []
    for token_id, maximum in zip(top_ids.tolist(), top_values.tolist()):
        if token_id < pieces:
            piece = processor.id_to_piece(token_id)
        elif token_id == pieces:
            piece = "<pad>"
        elif token_id == pieces + 1:
            piece = "<direction>"
        else:
            piece = f"<extra:{token_id}>"
        top_embedding_rows.append({
            "token_id": token_id,
            "piece": piece,
            "max_abs": maximum,
            "source_line_count": len(occurrences.get(token_id, [])),
            "source_lines_first_16": occurrences.get(token_id, [])[:16],
        })
    return {
        "top_state_tensors": tensor_maxima[:32],
        "top_encoder_embedding_rows": top_embedding_rows,
        "nonfinite_state_tensors": [
            name for name, value in state.items()
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all())
        ],
    }


@torch.no_grad()
def evaluate(model, rows, processor, args, pad, bos, eos, pieces) -> dict:
    model.eval()
    totals = {scale: [0.0, 0] for scale in SCALES}
    examples = []
    generation = {scale: {"lengths": [], "nonempty": 0, "repeat": [], "texts": []} for scale in SCALES}
    for batch_index, batch in enumerate(batches(rows, args.batch_size, args.seed + 9000, False)):
        source, length, targets = collate(batch, pad, args.device)
        _, _, _, levels, masks = model.states(source, length)
        depth_map = scale_depths(len(levels))
        for scale in SCALES:
            target = targets[scale]
            logits = teacher_exact_depth(model.decoder, levels, masks, target, bos, depth_map[scale])
            token_loss = finite_token_ce(logits, target, pad)
            totals[scale][0] += float(token_loss.double().sum())
            totals[scale][1] += token_loss.numel()
            if batch_index * args.batch_size < args.generate_items:
                generated = greedy_exact_depth(
                    model.decoder, levels, masks, bos, eos, args.max_generate, depth_map[scale],
                )
                for local, ids in enumerate(generated.tolist()):
                    token_ids = clean(ids, eos, pieces)
                    text = processor.decode(token_ids)
                    stats = generation[scale]
                    stats["lengths"].append(len(token_ids))
                    stats["nonempty"] += int(bool(token_ids))
                    stats["repeat"].append(
                        sum(a == b for a, b in zip(token_ids, token_ids[1:])) / max(1, len(token_ids) - 1)
                    )
                    stats["texts"].append(text)
                    per_scale_examples = max(1, args.example_items // len(SCALES))
                    if batch_index == 0 and local < min(len(batch), per_scale_examples):
                        examples.append({
                            "source_line": batch[local]["source_line"], "scale": scale,
                            "depth": depth_map[scale], "source": batch[local]["source_en"],
                            "target": batch[local][f"target_zh_{scale}"], "generated": text,
                        })
    metrics = {}
    for scale in SCALES:
        stats = generation[scale]
        n = len(stats["lengths"])
        metrics[scale] = {
            "nll": totals[scale][0] / max(1, totals[scale][1]),
            "tokens": totals[scale][1],
            "generation_mean_tokens": sum(stats["lengths"]) / max(1, n),
            "generation_nonempty": stats["nonempty"] / max(1, n),
            "generation_repeat_adjacent": sum(stats["repeat"]) / max(1, n),
            "generation_unique_ratio": len(set(stats["texts"])) / max(1, n),
        }
    mean_lengths = [metrics[scale]["generation_mean_tokens"] for scale in SCALES]
    return {
        "scales": metrics,
        "mean_length_strictly_increasing": all(a < b for a, b in zip(mean_lengths, mean_lengths[1:])),
        "examples": examples,
    }


def make_model(payload: dict, args, vocab: int, pad: int):
    model = ButterflyRecursive(
        vocab, args.dim, args.hidden, args.heap_width, pad,
        "butterfly", args.coupling_scale, dynamic_width=True,
    ).to(args.device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def train_arm(name, payload, train_rows, valid_rows, test_rows, processor, args, pad, bos, eos, pieces, output):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = make_model(payload, args, pieces + 3, pad)
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoints = set(args.eval_epochs)
    trace = []
    finite_gradients = True
    target_tokens = 0
    updates = 0
    started = time.time()
    evaluations = {"0": evaluate(model, valid_rows, processor, args, pad, bos, eos, pieces)}
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        scale_sum = {scale: 0.0 for scale in SCALES}
        local_updates = 0
        for batch in batches(train_rows, args.batch_size, args.seed + epoch * 1009, True):
            source, length, targets = collate(batch, pad, args.device)
            optimizer.zero_grad(set_to_none=True)
            local_losses = {}
            for scale in SCALES:
                selected = (
                    prefix_control_target(targets, scale, pad, eos)
                    if name == "prefix_control" else targets[scale]
                )
                _, _, _, levels, masks = model.states(source, length)
                depth = scale_depths(len(levels))[scale]
                logits = teacher_exact_depth(model.decoder, levels, masks, selected, bos, depth)
                loss = finite_token_ce(logits, selected, pad).double().mean()
                (loss / len(SCALES)).backward()
                nonfinite_parameters = [
                    parameter_name
                    for parameter_name, parameter in model.named_parameters()
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                ]
                if nonfinite_parameters:
                    level_energy = []
                    for level_index, (level, mask) in enumerate(zip(levels, masks)):
                        valid_values = level.detach()[mask]
                        level_energy.append({
                            "depth": level_index,
                            "nodes": int(mask.sum()),
                            "max_abs": float(valid_values.abs().max()),
                            "rms_fp64": float(valid_values.double().square().mean().sqrt()),
                        })
                    diagnostic = {
                        "arm": name,
                        "epoch": epoch,
                        "update_in_epoch": local_updates + 1,
                        "scale": scale,
                        "depth": depth,
                        "source_lines": [row["source_line"] for row in batch],
                        "source_token_lengths": length.detach().cpu().tolist(),
                        "target_token_lengths": selected.ne(pad).sum(dim=1).detach().cpu().tolist(),
                        "loss": float(loss.detach()),
                        "max_abs_logit": float(logits.detach().abs().max()),
                        "level_energy": level_energy,
                        "branch_weight_max_abs": float(model.decoder.branch.weight.detach().abs().max()),
                        "nonfinite_parameters": nonfinite_parameters,
                    }
                    (output / "nonfinite_gradient.json").write_text(
                        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    raise RuntimeError(
                        f"non-finite gradient in arm={name} epoch={epoch} "
                        f"update={local_updates + 1} scale={scale} depth={depth}"
                    )
                local_losses[scale] = float(loss.detach())
                target_tokens += int(selected.ne(pad).sum())
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            finite_gradients = finite_gradients and gradients_finite
            if not gradients_finite:
                raise RuntimeError(f"non-finite accumulated gradient in arm={name} epoch={epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            value = sum(local_losses.values()) / len(SCALES)
            loss_sum += value
            for scale in SCALES:
                scale_sum[scale] += local_losses[scale]
            updates += 1
            local_updates += 1
        row = {
            "arm": name, "epoch": epoch, "updates": updates,
            "train_nll_equal_scale_mean": loss_sum / max(1, local_updates),
            "train_nll_by_depth": {scale: scale_sum[scale] / max(1, local_updates) for scale in SCALES},
            "target_tokens_seen": target_tokens, "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch in checkpoints:
            evaluations[str(epoch)] = evaluate(model, valid_rows, processor, args, pad, bos, eos, pieces)
    test = evaluate(model, test_rows, processor, args, pad, bos, eos, pieces)
    result = {
        "arm": name, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_parameter_hash": initial_hash, "final_parameter_hash": parameter_hash(model),
        "finite_gradients": finite_gradients, "updates": updates,
        "target_tokens_seen": target_tokens, "trace": trace,
        "valid_by_epoch": evaluations, "test": test,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.0,
    }
    (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spm-model", type=Path, default=Path("/home/nio/datasets/wmt_massive/sp_bpe_massive.model"))
    parser.add_argument("--reranker", type=Path, default=Path("/home/nio/models/bge-reranker-v2-m3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-epochs", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch", type=int, default=32)
    parser.add_argument("--generate-items", type=int, default=16)
    parser.add_argument("--example-items", type=int, default=16)
    parser.add_argument("--max-generate", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--max-content", type=int, default=253)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--coupling-scale", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    defaults = {
        "smoke": {"max_items": 96, "epochs": 2, "eval_epochs": [1, 2]},
        "formal": {"max_items": 0, "epochs": 32, "eval_epochs": [1, 2, 4, 8, 16, 32]},
    }[args.mode]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    raw_rows = load_rows(args.data, args.seed, args.max_items)
    audit = {
        "reranker": reranker_audit(raw_rows, args.reranker, args.reranker_batch, args.device),
        "invariants": invariant_audit(raw_rows),
    }
    audit["aligned_better_scales"] = sum(
        row["paired_delta_mean"] > 0 for row in audit["reranker"].values()
    )
    (args.output / "semantic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"event": "semantic_audit", **audit}, ensure_ascii=False), flush=True)

    processor = spm.SentencePieceProcessor(model_file=str(args.spm_model))
    pieces, eos, bos = processor.get_piece_size(), processor.eos_id(), processor.bos_id()
    pad, direction_id = pieces, pieces + 1
    encoded = encode_rows(raw_rows, processor, direction_id, eos, args.max_content)
    split_rows = {split: [row for row in encoded if row["split"] == split] for split in ("train", "valid", "test")}
    if any(not rows for rows in split_rows.values()):
        raise RuntimeError("an encoded split is empty")
    split_ids = {split: sorted(row["source_line"] for row in rows) for split, rows in split_rows.items()}
    if set(split_ids["train"]) & set(split_ids["valid"]) or set(split_ids["train"]) & set(split_ids["test"]) or set(split_ids["valid"]) & set(split_ids["test"]):
        raise RuntimeError("split leakage")
    append_jsonl(args.output / "split_manifest.jsonl", [
        {"source_line": row["source_line"], "split": row["split"]} for row in encoded
    ])

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parameter_audit = checkpoint_parameter_audit(payload, processor, encoded, pieces)
    (args.output / "checkpoint_parameter_audit.json").write_text(
        json.dumps(parameter_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arms = {}
    for arm in ("prefix_control", "depth_curriculum"):
        arms[arm] = train_arm(
            arm, payload, split_rows["train"], split_rows["valid"], split_rows["test"],
            processor, args, pad, bos, eos, pieces, args.output,
        )
    same_initialization = arms["prefix_control"]["initial_parameter_hash"] == arms["depth_curriculum"]["initial_parameter_hash"]
    comparison = {}
    for scale in SCALES:
        control = arms["prefix_control"]["test"]["scales"][scale]
        curriculum = arms["depth_curriculum"]["test"]["scales"][scale]
        comparison[scale] = {
            "curriculum_minus_control_nll": curriculum["nll"] - control["nll"],
            "control_generation_mean_tokens": control["generation_mean_tokens"],
            "curriculum_generation_mean_tokens": curriculum["generation_mean_tokens"],
        }
    summary = {
        "claim": CLAIM, "mode": args.mode, "host": socket.gethostname(),
        "elapsed_sec": time.time() - started, "config": vars(args),
        "hashes": {
            "data": sha256(args.data), "checkpoint": sha256(args.checkpoint),
            "spm_model": sha256(args.spm_model),
        },
        "rows": {split: len(rows) for split, rows in split_rows.items()},
        "semantic_audit": audit, "same_initialization": same_initialization,
        "parameter_matched": arms["prefix_control"]["parameters"] == arms["depth_curriculum"]["parameters"],
        "finite_gradients": all(row["finite_gradients"] for row in arms.values()),
        "comparison": comparison,
        "control_length_order": arms["prefix_control"]["test"]["mean_length_strictly_increasing"],
        "curriculum_length_order": arms["depth_curriculum"]["test"]["mean_length_strictly_increasing"],
        "boundary": "Teacher-synthetic small-data depth-conditioning experiment; not human semantic truth, general summarization, energy efficiency, or Transformer superiority.",
    }
    serializable = json.loads(json.dumps(summary, default=str))
    (args.output / "summary.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", "summary": serializable}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
