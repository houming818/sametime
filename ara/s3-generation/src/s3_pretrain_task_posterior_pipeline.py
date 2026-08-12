#!/usr/bin/env python3
"""C10 pretrain -> matched task -> posterior-collapse TreeHeap pipeline.

Smoke mode validates the pipeline and evidence contract only. It cannot support
the registered claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_treeheap_butterfly_wmt import ButterflyRecursive  # noqa: E402
import s3_conditional_denoising_seq2seq as raw  # noqa: E402
import s3_treeheap_butterfly_bilingual_full as wmt  # noqa: E402
import s3_wmt_treeheap_seq2seq as wmt_metrics  # noqa: E402


CLAIM = "S3-TREEHEAP-PRETRAIN-POSTERIOR-C10"
CONTEXT_WIDTHS = (4, 8, 16, 32, 64, 128)


def json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_file_fingerprint(path: Path, chunk: int = 1024 * 1024) -> Dict[str, object]:
    """Fingerprint large corpora without rereading every byte."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if stat.st_size > chunk:
            handle.seek(max(0, stat.st_size - chunk))
            digest.update(handle.read(chunk))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampled_sha256": digest.hexdigest(),
    }


def state_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_checkpoint(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, vocab: int, pad: int) -> ButterflyRecursive:
    return ButterflyRecursive(
        vocab,
        args.dim,
        args.hidden,
        args.heap_width,
        pad,
        "butterfly",
        args.coupling_scale,
        dynamic_width=True,
    ).to(args.device)


def model_contract(model: torch.nn.Module, args, vocab: int, pad: int) -> Dict[str, object]:
    return {
        "class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "vocab": vocab,
        "pad": pad,
        "dim": args.dim,
        "hidden": args.hidden,
        "heap_width": args.heap_width,
        "coupling_scale": args.coupling_scale,
        "dynamic_width": True,
        "communication": "xor_butterfly",
        "fold": "learned_lifting",
        "read": "recursive_probability_container",
    }


def wmt_monolingual_documents(path: Path, split: str) -> Iterator[str]:
    while True:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                pair = wmt.parse_pair(line)
                if pair is None or wmt.partition(line) != "train":
                    continue
                zh, en = pair
                # No pair label or direction token enters Stage A.
                for language, text in (("zh", zh), ("en", en)):
                    bucket = raw.document_bucket(f"wmt-{language}", text)
                    if len(text) >= 32 and raw.in_split(bucket, split):
                        yield text


def mixed_pretrain_documents(args, split: str, seed: int) -> Iterator[str]:
    rng = random.Random(seed)
    natural = iter(raw.MixedDocuments(Path(args.raw_root), split, seed))
    monolingual = wmt_monolingual_documents(Path(args.wmt_data), split)
    while True:
        if rng.random() < args.pretrain_wmt_ratio:
            yield next(monolingual)
        else:
            yield next(natural)


def collect_pretrain_rows(
    args,
    sp: spm.SentencePieceProcessor,
    split: str,
    wanted: int,
    seed: int,
) -> Tuple[List[Tuple[List[int], List[int], int]], Dict[str, int]]:
    rows: List[Tuple[List[int], List[int], int]] = []
    counts = Counter()
    rng = random.Random(seed)
    for text in mixed_pretrain_documents(args, split, seed):
        ids = sp.encode(text, out_type=int)
        counts["documents"] += 1
        counts["raw_pieces"] += len(ids)
        if len(ids) < min(CONTEXT_WIDTHS) + args.pretrain_target:
            continue
        widths = list(CONTEXT_WIDTHS)
        rng.shuffle(widths)
        for width in widths:
            total = width + args.pretrain_target
            if len(ids) < total:
                continue
            max_start = len(ids) - total
            start = rng.randrange(max_start + 1) if max_start else 0
            source = ids[start : start + width]
            target = ids[start + width : start + total]
            rows.append((source, target, width))
            counts[f"width_{width}"] += 1
            counts["target_pieces"] += len(target)
            if len(rows) >= wanted:
                return rows, dict(counts)
        if counts["documents"] >= args.max_pretrain_scan_documents:
            break
    if len(rows) < wanted:
        raise RuntimeError(f"pretrain row collection stopped at {len(rows)}/{wanted}: {dict(counts)}")
    return rows, dict(counts)


class FreshPretrainSchedule:
    """Finite schedule that reads fresh corpus windows instead of cycling rows."""

    def __init__(self, args, sp, steps: int, batch_size: int, seed: int):
        self.args = args
        self.sp = sp
        self.steps = steps
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed)
        buffers = {width: [] for width in CONTEXT_WIDTHS}
        emitted = 0
        for text in mixed_pretrain_documents(self.args, "train", self.seed):
            ids = self.sp.encode(text, out_type=int)
            widths = list(CONTEXT_WIDTHS)
            rng.shuffle(widths)
            for width in widths:
                total = width + self.args.pretrain_target
                if len(ids) < total:
                    continue
                max_start = len(ids) - total
                start = rng.randrange(max_start + 1) if max_start else 0
                buffers[width].append((
                    ids[start : start + width],
                    ids[start + width : start + total],
                    width,
                ))
                if len(buffers[width]) >= self.batch_size:
                    yield buffers[width][: self.batch_size]
                    buffers[width] = buffers[width][self.batch_size :]
                    emitted += 1
                    if emitted >= self.steps:
                        return
        raise RuntimeError(f"fresh pretrain stream ended at {emitted}/{self.steps} batches")


def collate_rows(rows: Sequence[Tuple[List[int], List[int], object]], pad: int, device: str):
    source_width = max(len(row[0]) for row in rows)
    target_width = max(len(row[1]) for row in rows)
    source = torch.full((len(rows), source_width), pad, dtype=torch.long)
    target = torch.full((len(rows), target_width), pad, dtype=torch.long)
    length = torch.tensor([len(row[0]) for row in rows], dtype=torch.long)
    for index, row in enumerate(rows):
        source[index, : len(row[0])] = torch.tensor(row[0])
        target[index, : len(row[1])] = torch.tensor(row[1])
    return source.to(device), length.to(device), target.to(device)


def rows_schedule(rows, steps: int, batch_size: int, seed: int):
    by_width = defaultdict(list)
    for row in rows:
        by_width[len(row[0])].append(row)
    rng = random.Random(seed)
    order = sorted(by_width)
    positions = {width: 0 for width in order}
    for width in order:
        rng.shuffle(by_width[width])
    schedule = []
    for step in range(steps):
        width = order[step % len(order)]
        bucket = by_width[width]
        position = positions[width]
        if position + batch_size > len(bucket):
            rng.shuffle(bucket)
            position = 0
        batch = bucket[position : position + batch_size]
        if not batch:
            raise RuntimeError(f"empty batch for width {width}")
        positions[width] = position + len(batch)
        schedule.append(batch)
    return schedule


def stream_sha256(schedule) -> str:
    digest = hashlib.sha256()
    for batch in schedule:
        update_stream_digest(digest, batch)
    return digest.hexdigest()


def update_stream_digest(digest, batch) -> None:
    for row in batch:
        for values in row[:2]:
            digest.update(len(values).to_bytes(4, "big"))
            for value in values:
                digest.update(int(value).to_bytes(4, "big", signed=True))


@torch.no_grad()
def evaluate_nll(model, rows, pad: int, bos: int, device: str, batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for start in range(0, len(rows), batch_size):
        source, length, target = collate_rows(rows[start : start + batch_size], pad, device)
        logits, _ = model.teacher(source, length, target, bos)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="sum",
        ))
        token_count += int(target.ne(pad).sum())
    nll = loss_sum / max(1, token_count)
    return {"nll": nll, "ppl": math.exp(min(20.0, nll)), "tokens": token_count}


def train_schedule(
    model,
    schedule,
    valid_rows,
    args,
    pad: int,
    bos: int,
    output: Path,
    stage: str,
    parent_sha: str,
) -> Dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    initial = evaluate_nll(model, valid_rows, pad, bos, args.device, args.eval_batch)
    best = initial["nll"]
    best_state = cpu_state(model)
    started = time.time()
    train_loss_sum = 0.0
    train_tokens = 0
    finite_gradients = True
    schedule_digest = hashlib.sha256()
    for step, batch in enumerate(schedule, 1):
        update_stream_digest(schedule_digest, batch)
        model.train()
        source, length, target = collate_rows(batch, pad, args.device)
        logits, route = model.teacher(source, length, target, bos)
        tokens = int(target.ne(pad).sum())
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="sum",
        )
        loss = loss_sum / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        finite_gradients &= finite
        if not finite:
            raise RuntimeError(f"non-finite gradient in {stage} step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        train_loss_sum += float(loss_sum.detach())
        train_tokens += tokens
        if step == 1 or step == len(schedule) or step % args.log_every == 0:
            valid = evaluate_nll(model, valid_rows, pad, bos, args.device, args.eval_batch)
            row = {
                "stage": stage,
                "step": step,
                "train_nll": train_loss_sum / max(1, train_tokens),
                "valid_nll": valid["nll"],
                "valid_ppl": valid["ppl"],
                "grad_norm": grad_norm,
                "route": [float(value) for value in route.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if valid["nll"] < best:
                best = valid["nll"]
                best_state = cpu_state(model)
    model.load_state_dict(best_state, strict=True)
    final = evaluate_nll(model, valid_rows, pad, bos, args.device, args.eval_batch)
    checkpoint = output / "checkpoint_best.pt"
    payload = {
        "claim": CLAIM,
        "stage": stage,
        "state_dict": best_state,
        "state_sha256": state_sha256(best_state),
        "parent_state_sha256": parent_sha,
        "config": vars(args),
        "steps": len(schedule),
        "train_tokens": train_tokens,
        "stream_sha256": schedule_digest.hexdigest(),
        "initial_valid": initial,
        "best_valid": final,
    }
    atomic_checkpoint(checkpoint, payload)
    summary = {
        "stage": stage,
        "parent_state_sha256": parent_sha,
        "state_sha256": payload["state_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "steps": len(schedule),
        "train_tokens": train_tokens,
        "stream_sha256": payload["stream_sha256"],
        "train_nll": train_loss_sum / max(1, train_tokens),
        "initial_valid": initial,
        "best_valid": final,
        "finite_gradients": finite_gradients,
        "seconds": time.time() - started,
    }
    json_write(output / "summary.json", summary)
    return summary


def collect_wmt_rows(args, sp, direction_ids, eos: int):
    train_rows = []
    valid_rows = []
    test_rows = []
    with Path(args.wmt_data).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if (
                len(train_rows) >= args.task_train_rows
                and len(valid_rows) >= args.task_eval_rows
                and len(test_rows) >= args.task_eval_rows
            ):
                break
            pair = wmt.parse_pair(line)
            if pair is None:
                continue
            split = wmt.partition(line)
            target = train_rows if split == "train" else valid_rows if split == "valid" else test_rows
            limit = args.task_train_rows if split == "train" else args.task_eval_rows
            if len(target) >= limit:
                continue
            direction = wmt.DIRECTIONS[wmt.stable_int(f"c10:{line_no}:{line}") & 1]
            encoded = wmt.encode_pair(pair, sp, direction, direction_ids, eos)
            if wmt.eligible(encoded, args.max_content):
                target.append((*encoded, direction, pair, line_no))
            if line_no + 1 >= args.max_wmt_scan_lines:
                break
    if min(len(train_rows), len(valid_rows), len(test_rows)) == 0:
        raise RuntimeError(
            f"failed WMT collection train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)}"
        )
    return train_rows, valid_rows, test_rows


@torch.no_grad()
def task_generation_metrics(model, rows, args, sp, pad, bos, eos, pieces, limit=32):
    model.eval()
    hypotheses: List[List[int]] = []
    references: List[List[int]] = []
    examples = []
    for row in rows[:limit]:
        source, length, target = collate_rows([row], pad, args.device)
        generated, route = model.greedy(source, length, bos, eos, min(args.max_generation, target.shape[1] + 16))
        hypothesis = wmt.clean(generated[0].tolist(), eos, pieces)
        reference = wmt.clean(target[0].tolist(), eos, pieces)
        hypotheses.append(hypothesis)
        references.append(reference)
        if len(examples) < 8:
            examples.append({
                "direction": row[2],
                "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                "reference": sp.decode(reference),
                "generation": sp.decode(hypothesis),
                "route": [float(value) for value in route.detach().cpu()],
            })
    return {
        "token_bleu4": wmt_metrics.bleu4(hypotheses, references),
        "nonempty": sum(bool(row) for row in hypotheses) / max(1, len(hypotheses)),
        "examples": examples,
    }


def train_task_arm(
    name: str,
    initial_state,
    initial_sha: str,
    schedule,
    valid_rows,
    test_rows,
    args,
    sp,
    vocab,
    pad,
    bos,
    eos,
    pieces,
    output: Path,
):
    model = build_model(args, vocab, pad)
    model.load_state_dict(initial_state, strict=True)
    loaded_sha = state_sha256(cpu_state(model))
    if loaded_sha != initial_sha:
        raise RuntimeError(f"{name} parent mismatch: {loaded_sha} != {initial_sha}")
    summary = train_schedule(
        model, schedule, valid_rows, args, pad, bos, output / name, f"task-{name}", initial_sha,
    )
    summary["test"] = evaluate_nll(model, test_rows, pad, bos, args.device, args.eval_batch)
    summary["generation"] = task_generation_metrics(
        model, test_rows, args, sp, pad, bos, eos, pieces, args.generation_examples,
    )
    json_write(output / name / "summary.json", summary)
    return summary


def build_candidate_bank(args, sp, seed: int):
    context_lengths = tuple(int(value) for value in args.posterior_context_lengths.split(","))
    counters = {length: defaultdict(Counter) for length in context_lengths}
    document_ids = {length: {} for length in context_lengths}
    unigram = Counter()
    seen_tokens = 0
    documents = 0
    for text in mixed_pretrain_documents(args, "test", seed):
        ids = sp.encode(text, out_type=int)
        document_id = hashlib.blake2s(
            text[:2048].encode("utf-8", errors="replace"), digest_size=8,
        ).hexdigest()
        documents += 1
        unigram.update(ids)
        seen_tokens += len(ids)
        for position in range(max(max(context_lengths), args.posterior_source_width), len(ids)):
            for length in context_lengths:
                key = tuple(ids[position - length : position])
                table = counters[length]
                if key in table or len(table) < args.posterior_max_keys:
                    table[key][ids[position]] += 1
                    document_ids[length].setdefault(key, document_id)
        if seen_tokens >= args.posterior_scan_tokens or documents >= args.max_pretrain_scan_documents:
            break
    ranked = []
    for length in context_lengths:
        for key, counts in counters[length].items():
            count = sum(counts.values())
            if count >= args.posterior_min_count:
                ranked.append((count, len(counts), length, key, counts))
    ranked.sort(key=lambda row: (row[1] > 1, row[0], row[1]), reverse=True)
    selected = []
    per_length = Counter()
    per_document = Counter()
    per_length_limit = max(1, math.ceil(args.posterior_contexts / len(context_lengths)))
    for count, diversity, length, key, counts in ranked:
        if per_length[length] >= per_length_limit:
            continue
        document_id = document_ids[length][key]
        if per_document[document_id] >= 2:
            continue
        selected.append({
            "context_length": length,
            "suffix_ids": list(key),
            "source_ids": list(key),
            "document_id": document_id,
            "occurrences": count,
            "candidate_counts": {str(token): value for token, value in counts.items()},
            "candidate_diversity": diversity,
        })
        per_length[length] += 1
        per_document[document_id] += 1
        if len(selected) >= args.posterior_contexts:
            break
    if len(selected) < args.posterior_contexts:
        used = {(row["context_length"], tuple(row["suffix_ids"])) for row in selected}
        for count, diversity, length, key, counts in ranked:
            if len(selected) >= args.posterior_contexts:
                break
            if (length, key) in used or per_length[length] >= per_length_limit:
                continue
            document_id = document_ids[length][key]
            selected.append({
                "context_length": length,
                "suffix_ids": list(key),
                "source_ids": list(key),
                "document_id": document_id,
                "occurrences": count,
                "candidate_counts": {str(token): value for token, value in counts.items()},
                "candidate_diversity": diversity,
            })
            per_length[length] += 1
            per_document[document_id] += 1
    if not selected:
        raise RuntimeError("no repeated held-out posterior contexts were found")
    return selected, unigram, {
        "documents": documents,
        "tokens": seen_tokens,
        "selected": len(selected),
        "per_length": dict(per_length),
        "unique_documents": len(per_document),
    }


def js_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    midpoint = [(left + right) * 0.5 for left, right in zip(first, second)]

    def kl(left, right):
        return sum(a * math.log(a / b) for a, b in zip(left, right) if a > 0.0 and b > 0.0)

    return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)


def bucket_metrics(probabilities: torch.Tensor, candidate_counts: Dict[str, int], unigram: Counter):
    tokens = sorted(int(token) for token in candidate_counts)
    count_total = sum(candidate_counts.values())
    empirical = [candidate_counts[str(token)] / count_total for token in tokens]
    model_values = [float(probabilities[token]) for token in tokens]
    unigram_total = sum(unigram.values())
    unigram_values = [unigram[token] / max(1, unigram_total) for token in tokens]
    empirical_bucket = [*empirical, 0.0]
    model_bucket = [*model_values, max(0.0, 1.0 - sum(model_values))]
    unigram_bucket = [*unigram_values, max(0.0, 1.0 - sum(unigram_values))]
    return {
        "model_js": js_divergence(model_bucket, empirical_bucket),
        "unigram_js": js_divergence(unigram_bucket, empirical_bucket),
        "model_candidate_mass": sum(model_values),
        "unigram_candidate_mass": sum(unigram_values),
    }


def first_token_probabilities(model, candidates, args, pad, bos, pieces, mode, pair_break_depth=-1):
    rows = [
        (row["source_ids"], [int(next(iter(row["candidate_counts"])))], None)
        for row in candidates
    ]
    source, length, target = collate_rows(rows, pad, args.device)
    if mode == "wrong_source":
        source = source.roll(1, dims=0)
        length = length.roll(1, dims=0)
    previous = model.encoder.runtime_mode
    model.encoder.runtime_mode = "identity" if mode == "identity" else "butterfly"
    try:
        logits, route = model.teacher(
            source,
            length,
            target,
            bos,
            pair_break_depth=pair_break_depth,
        )
    finally:
        model.encoder.runtime_mode = previous
    probabilities = F.softmax(logits[:, 0, :pieces], dim=-1).detach().cpu()
    return probabilities, [float(value) for value in route.detach().cpu()]


@torch.no_grad()
def posterior_proof(model, candidates, unigram, args, sp, pad, bos, eos, pieces, output: Path):
    model.eval()
    modes = {
        "native": -1,
        "wrong_source": -1,
        "identity": -1,
        "pair_break_depth_0": 0,
    }
    all_probabilities = {}
    routes = {}
    for mode, pair_break_depth in modes.items():
        probabilities, route = first_token_probabilities(
            model, candidates, args, pad, bos, pieces,
            "native" if mode == "pair_break_depth_0" else mode,
            pair_break_depth,
        )
        all_probabilities[mode] = probabilities
        routes[mode] = route
    native_greedy = all_probabilities["native"].argmax(-1)
    unigram_top1 = unigram.most_common(1)[0][0]
    rng = torch.Generator().manual_seed(args.seed + 7007)
    rows = []
    generations = []
    for index, candidate in enumerate(candidates):
        row = {
            "index": index,
            "context_length": candidate["context_length"],
            "suffix": sp.decode(candidate["suffix_ids"]),
            "source": sp.decode(candidate["source_ids"]),
            "occurrences": candidate["occurrences"],
            "empirical_candidates": [
                {
                    "token": sp.id_to_piece(int(token)),
                    "id": int(token),
                    "count": count,
                }
                for token, count in sorted(
                    candidate["candidate_counts"].items(), key=lambda item: item[1], reverse=True,
                )
            ],
            "modes": {},
        }
        for mode, probabilities in all_probabilities.items():
            local = probabilities[index]
            metrics = bucket_metrics(local, candidate["candidate_counts"], unigram)
            top_values, top_ids = torch.topk(local, min(20, local.numel()))
            greedy = int(top_ids[0])
            metrics.update({
                "greedy_id": greedy,
                "greedy": sp.id_to_piece(greedy),
                "greedy_in_empirical": str(greedy) in candidate["candidate_counts"],
                "entropy": float(-(local * local.clamp_min(1e-12).log()).sum()),
                "top20": [
                    {"id": int(token), "token": sp.id_to_piece(int(token)), "probability": float(value)}
                    for value, token in zip(top_values, top_ids)
                ],
            })
            row["modes"][mode] = metrics
        sampled = int(torch.multinomial(all_probabilities["native"][index], 1, generator=rng))
        row["native_top_p_proxy_sample"] = {"id": sampled, "token": sp.id_to_piece(sampled)}
        rows.append(row)

        source, length, _ = collate_rows(
            [(candidate["source_ids"], [0], None)], pad, args.device,
        )
        generated, route = model.greedy(source, length, bos, eos, args.posterior_generation_length)
        ids = wmt.clean(generated[0].tolist(), eos, pieces)
        generations.append({
            "index": index,
            "source": sp.decode(candidate["source_ids"]),
            "generation": sp.decode(ids),
            "ids": ids,
            "route": [float(value) for value in route.detach().cpu()],
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "posterior_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    json_write(output / "generations.json", generations)

    def mode_average(mode, key):
        return sum(row["modes"][mode][key] for row in rows) / max(1, len(rows))

    native_ids = [row["modes"]["native"]["greedy_id"] for row in rows]
    severe_repetition = []
    for generation in generations:
        ids = generation["ids"]
        fraction = sum(left == right for left, right in zip(ids, ids[1:])) / max(1, len(ids) - 1)
        severe_repetition.append(fraction >= 0.5)
    mode_summary = {}
    for mode in modes:
        mode_summary[mode] = {
            "model_js": mode_average(mode, "model_js"),
            "unigram_js": mode_average(mode, "unigram_js"),
            "model_candidate_mass": mode_average(mode, "model_candidate_mass"),
            "unigram_candidate_mass": mode_average(mode, "unigram_candidate_mass"),
            "greedy_in_empirical": mode_average(mode, "greedy_in_empirical"),
            "entropy": mode_average(mode, "entropy"),
            "route": routes[mode],
            "greedy_changed_from_native": sum(
                row["modes"][mode]["greedy_id"] != native_ids[index]
                for index, row in enumerate(rows)
            ) / max(1, len(rows)),
        }
    summary = {
        "contexts": len(rows),
        "unigram_top1_id": unigram_top1,
        "unigram_top1": sp.id_to_piece(unigram_top1),
        "native_unique_greedy_rate": len(set(native_ids)) / max(1, len(native_ids)),
        "native_greedy_differs_from_unigram_rate": sum(value != unigram_top1 for value in native_ids) / max(1, len(native_ids)),
        "nonempty_generation_rate": sum(bool(row["ids"]) for row in generations) / max(1, len(generations)),
        "severe_adjacent_repetition_rate": sum(severe_repetition) / max(1, len(severe_repetition)),
        "maximum_greedy_share": max(Counter(native_ids).values()) / max(1, len(native_ids)),
        "modes": mode_summary,
    }
    json_write(output / "summary.json", summary)
    return summary


def reload_check(args, vocab, pad, checkpoint: Path) -> bool:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    matched = state_sha256(cpu_state(model)) == payload["state_sha256"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matched


def configure(args):
    if args.mode == "smoke":
        args.pretrain_batch = args.pretrain_batch or 16
        args.pretrain_train_rows = args.pretrain_train_rows or 192
        args.pretrain_valid_rows = args.pretrain_valid_rows or 48
        args.pretrain_steps = args.pretrain_steps or 20
        args.task_train_rows = args.task_train_rows or 256
        args.task_eval_rows = args.task_eval_rows or 48
        args.task_steps = args.task_steps or 20
        args.posterior_contexts = args.posterior_contexts or 16
        args.posterior_scan_tokens = args.posterior_scan_tokens or 100_000
        args.posterior_min_count = args.posterior_min_count or 2
        args.posterior_max_keys = args.posterior_max_keys or 20_000
        args.log_every = args.log_every or 10
    else:
        args.pretrain_batch = args.pretrain_batch or 32
        args.pretrain_train_rows = args.pretrain_train_rows or 0
        args.pretrain_valid_rows = args.pretrain_valid_rows or 2_048
        args.pretrain_steps = args.pretrain_steps or math.ceil(
            100_000_000 / (args.pretrain_batch * args.pretrain_target)
        )
        args.task_train_rows = args.task_train_rows or 200_000
        args.task_eval_rows = args.task_eval_rows or 1_000
        args.task_steps = args.task_steps or 25_000
        args.posterior_contexts = args.posterior_contexts or 256
        args.posterior_scan_tokens = args.posterior_scan_tokens or 2_000_000
        args.posterior_min_count = args.posterior_min_count or 4
        args.posterior_max_keys = args.posterior_max_keys or 100_000
        args.log_every = args.log_every or 1_000
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/smoke")
    parser.add_argument("--raw-root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--coupling-scale", type=float, default=0.25)
    parser.add_argument("--max-content", type=int, default=253)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pretrain-wmt-ratio", type=float, default=0.25)
    parser.add_argument("--pretrain-target", type=int, default=32)
    parser.add_argument("--pretrain-batch", type=int)
    parser.add_argument("--task-batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--pretrain-train-rows", type=int)
    parser.add_argument("--pretrain-valid-rows", type=int)
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--task-train-rows", type=int)
    parser.add_argument("--task-eval-rows", type=int)
    parser.add_argument("--task-steps", type=int)
    parser.add_argument("--posterior-contexts", type=int)
    parser.add_argument("--posterior-context-lengths", default="4,8,16")
    parser.add_argument("--posterior-source-width", type=int, default=128)
    parser.add_argument("--posterior-scan-tokens", type=int)
    parser.add_argument("--posterior-min-count", type=int)
    parser.add_argument("--posterior-max-keys", type=int)
    parser.add_argument("--posterior-generation-length", type=int, default=24)
    parser.add_argument("--generation-examples", type=int, default=16)
    parser.add_argument("--max-generation", type=int, default=96)
    parser.add_argument("--max-pretrain-scan-documents", type=int, default=200_000)
    parser.add_argument("--max-wmt-scan-lines", type=int, default=3_000_000)
    parser.add_argument("--log-every", type=int)
    args = configure(parser.parse_args())
    if args.heap_width != 256 or args.heap_width & (args.heap_width - 1):
        raise ValueError("C10 freezes a 256-leaf power-of-two TreeHeap")
    if args.posterior_source_width > args.heap_width:
        raise ValueError("posterior source exceeds TreeHeap width")
    if not 0.0 <= args.pretrain_wmt_ratio <= 1.0:
        raise ValueError("pretrain WMT ratio must be in [0,1]")

    seed_all(args.seed)
    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces = sp.get_piece_size()
    eos, bos = sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3

    model = build_model(args, vocab, pad)
    contract = model_contract(model, args, vocab, pad)
    initial_state = cpu_state(model)
    initial_sha = state_sha256(initial_state)
    initial_checkpoint = output / "initial_state.pt"
    atomic_checkpoint(initial_checkpoint, {
        "claim": CLAIM,
        "state_dict": initial_state,
        "state_sha256": initial_sha,
        "contract": contract,
        "config": vars(args),
    })
    json_write(output / "initial_state.json", {
        "state_sha256": initial_sha,
        "checkpoint_sha256": sha256_file(initial_checkpoint),
        "contract": contract,
    })

    data_manifest = {
        "tokenizer": {"path": args.spm_model, "sha256": sha256_file(Path(args.spm_model))},
        "wmt": sampled_file_fingerprint(Path(args.wmt_data)),
        "raw_root": str(Path(args.raw_root)),
        "raw_sources": [str(path) for name in raw.SOURCES for path in raw.files(Path(args.raw_root), name)],
        "splits": "raw document hash 96/2/2; WMT line hash train/valid/test",
    }
    json_write(output / "data_manifest.json", data_manifest)
    json_write(output / "environment.json", {
        "host": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    json_write(output / "config.json", vars(args))

    print(json.dumps({"event": "collect_pretrain_rows", "mode": args.mode}), flush=True)
    if args.mode == "pilot":
        pretrain_train = []
        pretrain_train_counts = {
            "streaming": True,
            "steps": args.pretrain_steps,
            "target_pieces_per_batch": args.pretrain_batch * args.pretrain_target,
        }
        pretrain_schedule = FreshPretrainSchedule(
            args, sp, args.pretrain_steps, args.pretrain_batch, args.seed + 13,
        )
    else:
        pretrain_train, pretrain_train_counts = collect_pretrain_rows(
            args, sp, "train", args.pretrain_train_rows, args.seed + 11,
        )
        pretrain_schedule = rows_schedule(
            pretrain_train, args.pretrain_steps, args.pretrain_batch, args.seed + 13,
        )
    pretrain_valid, pretrain_valid_counts = collect_pretrain_rows(
        args, sp, "valid", args.pretrain_valid_rows, args.seed + 12,
    )
    pretrain_summary = train_schedule(
        model,
        pretrain_schedule,
        pretrain_valid,
        args,
        pad,
        bos,
        output / "pretrain",
        "pretrain",
        initial_sha,
    )
    pretrain_stream_hash = pretrain_summary["stream_sha256"]
    pretrained_state = cpu_state(model)
    pretrained_sha = state_sha256(pretrained_state)
    if pretrained_sha != pretrain_summary["state_sha256"]:
        raise RuntimeError("in-memory pretrain state does not match checkpoint summary")

    print(json.dumps({"event": "collect_wmt_rows"}), flush=True)
    task_train, task_valid, task_test = collect_wmt_rows(args, sp, direction_ids, eos)
    task_schedule = rows_schedule(task_train, args.task_steps, args.task_batch, args.seed + 21)
    task_stream_hash = stream_sha256(task_schedule)
    task_output = output / "task"
    pt_summary = train_task_arm(
        "PT", pretrained_state, pretrained_sha, task_schedule, task_valid, task_test,
        args, sp, vocab, pad, bos, eos, pieces, task_output,
    )
    sc_summary = train_task_arm(
        "SC", initial_state, initial_sha, task_schedule, task_valid, task_test,
        args, sp, vocab, pad, bos, eos, pieces, task_output,
    )

    print(json.dumps({"event": "build_posterior_bank"}), flush=True)
    candidates, unigram, candidate_counts = build_candidate_bank(args, sp, args.seed + 31)
    json_write(output / "proof" / "candidate_bank.json", {
        "metadata": candidate_counts,
        "candidates": candidates,
        "unigram_top100": unigram.most_common(100),
    })
    model.load_state_dict(pretrained_state, strict=True)
    proof_summary = posterior_proof(
        model, candidates, unigram, args, sp, pad, bos, eos, pieces, output / "proof",
    )

    integrity = {
        "architecture_parameter_count_frozen": contract["parameters"]
        == sum(parameter.numel() for parameter in model.parameters()),
        "pretrain_parent_is_initial": pretrain_summary["parent_state_sha256"] == initial_sha,
        "pt_parent_is_pretrained": pt_summary["parent_state_sha256"] == pretrained_sha,
        "sc_parent_is_initial": sc_summary["parent_state_sha256"] == initial_sha,
        "matched_task_stream_hash": pt_summary["stream_sha256"]
        == sc_summary["stream_sha256"]
        == task_stream_hash,
        "finite_gradients": all(row["finite_gradients"] for row in (pretrain_summary, pt_summary, sc_summary)),
        "pretrain_checkpoint_reload": reload_check(
            args, vocab, pad, output / "pretrain" / "checkpoint_best.pt",
        ),
        "pt_checkpoint_reload": reload_check(
            args, vocab, pad, output / "task" / "PT" / "checkpoint_best.pt",
        ),
        "sc_checkpoint_reload": reload_check(
            args, vocab, pad, output / "task" / "SC" / "checkpoint_best.pt",
        ),
    }
    report = {
        "claim": CLAIM,
        "mode": args.mode,
        "claim_status": "open / smoke validates code only" if args.mode == "smoke" else "open / pilot evidence",
        "contract": contract,
        "hashes": {
            "initial_state": initial_sha,
            "pretrained_state": pretrained_sha,
            "pretrain_stream": pretrain_stream_hash,
            "task_stream_shared_by_PT_and_SC": task_stream_hash,
        },
        "row_counts": {
            "pretrain_train": len(pretrain_train) if pretrain_train else "fresh streaming windows",
            "pretrain_valid": len(pretrain_valid),
            "pretrain_train_scan": pretrain_train_counts,
            "pretrain_valid_scan": pretrain_valid_counts,
            "task_train": len(task_train),
            "task_valid": len(task_valid),
            "task_test": len(task_test),
            "posterior": candidate_counts,
        },
        "pretrain": pretrain_summary,
        "task": {"PT": pt_summary, "SC": sc_summary},
        "proof": proof_summary,
        "integrity": integrity,
        "p0_smoke_pass": all(integrity.values()),
        "seconds": time.time() - started,
        "not_proved": [
            "Bayesian or semantic reasoning",
            "product-quality generation",
            "architecture superiority",
            "any C10 prediction when mode=smoke",
        ],
    }
    json_write(output / "report.json", report)
    (output / "README.md").write_text(
        "# C10 Pretrain-to-Task Posterior Pipeline\n\n"
        f"Mode: `{args.mode}`  \nClaim: `{CLAIM}`  \n"
        f"P0 smoke pass: `{report['p0_smoke_pass']}`\n\n"
        "Smoke validates executable data flow and artifacts only. It is not claim evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete",
        "p0_smoke_pass": report["p0_smoke_pass"],
        "pretrain_nll": pretrain_summary["best_valid"]["nll"],
        "task_PT_nll": pt_summary["best_valid"]["nll"],
        "task_SC_nll": sc_summary["best_valid"]["nll"],
        "posterior_native": proof_summary["modes"]["native"],
        "seconds": report["seconds"],
        "evidence": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
