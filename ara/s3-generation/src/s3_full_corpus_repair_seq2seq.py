#!/usr/bin/env python3
"""Long-running repair-aware TreeHeap seq2seq pretraining on io-local corpora."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_annealed_frontier_pretrain as anneal
import s3_multiresolution_damage_repair as repair_probe
import s3_wmt_treeheap_seq2seq as base


SOURCE_NAMES = (
    "news_cont", "wiki_cont", "web_qa", "belle_2m", "belle_1m",
    "baike_qa", "translation", "wmt",
)
SOURCE_WEIGHTS = (0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05, 0.10)
RAW_CHUNK_CHARS = 4096
ENCODE_CHARS_PER_TOKEN = 12


def json_rows(paths: Sequence[Path], partition: str | None = None) -> Iterator[dict]:
    while True:
        for path in paths:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for row_number, line in enumerate(handle):
                    is_holdout = row_number % 1000 == 0
                    if partition == "train" and is_holdout:
                        continue
                    if partition == "valid" and not is_holdout:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row


def raw_records(paths: Sequence[Path], kind: str) -> Iterator[Tuple[str, str]]:
    for row in json_rows(paths):
        if kind == "wiki":
            text = (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
        else:
            text = (str(row.get("title", "")) + "\n" + str(row.get("content", ""))).strip()
        if len(text) >= 64 and text.count("�") <= 2:
            for start in range(0, len(text), RAW_CHUNK_CHARS):
                chunk = text[start : start + RAW_CHUNK_CHARS]
                if len(chunk) >= 64:
                    yield chunk, ""


def pair_records(paths: Sequence[Path], kind: str, partition: str | None = None) -> Iterator[Tuple[str, str]]:
    for row in json_rows(paths, partition):
        if kind == "web":
            source = (str(row.get("title", "")) + "\n" + str(row.get("desc", ""))).strip()
            target = str(row.get("content", "")).strip()
        elif kind == "belle":
            source = (str(row.get("instruction", "")) + "\n" + str(row.get("input", ""))).strip()
            target = str(row.get("output", "")).strip()
        elif kind == "baike":
            source = (str(row.get("title", "")) + "\n" + str(row.get("desc", ""))).strip()
            target = str(row.get("answer", "")).strip()
        elif kind == "translation":
            source = str(row.get("english", "")).strip()
            target = str(row.get("chinese", "")).strip()
        else:
            raise ValueError(kind)
        if len(source) >= 2 and len(target) >= 2 and source.count("�") <= 2 and target.count("�") <= 2:
            yield source, target


def tsv_records(path: Path, source_first: bool) -> Iterator[Tuple[str, str]]:
    while True:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                if source_first:
                    english, chinese = fields[0].strip(), fields[1].strip()
                else:
                    chinese, english = fields[0].strip(), fields[1].strip()
                if len(chinese) >= 2 and len(english) >= 2 and chinese.count("�") <= 2 and english.count("�") <= 2:
                    yield english, chinese


def source_streams(root: Path, split: str) -> Dict[str, Iterator[Tuple[str, str]]]:
    base_dir = root / "pretrain" / "Chinese-Train-Datasets"
    suffix = "train" if split == "train" else "valid"
    news = base_dir / "new2016zh" / f"news2016zh_{suffix}.json"
    web = base_dir / "webtext2019zh" / f"webtext_zh_{suffix}.json"
    baike = base_dir / "baike2018qa" / f"baike_qa_{suffix}.json"
    translation = base_dir / "translation2019zh" / f"translation2019zh_{suffix}.json"
    if split == "train":
        belle2 = base_dir / "belle_zh" / "train_2M_CN.json"
        belle1 = base_dir / "belle_zh" / "Belle_open_source_1M.json"
        wiki_paths = sorted((base_dir / "wiki_zh").rglob("wiki_*"))[:-16]
        wmt = root / "wmt_massive" / "train.massive.zh-en.tsv"
    else:
        # Fixed held-out files are preferred; BELLE/WMT use a separate stream
        # position and seed but remain marked as train-origin in evidence.
        belle2 = base_dir / "belle_zh" / "train_2M_CN.json"
        belle1 = base_dir / "belle_zh" / "Belle_open_source_1M.json"
        wiki_paths = sorted((base_dir / "wiki_zh").rglob("wiki_*"))[-16:]
        wmt = root / "wmt17" / "train.zh-en"
    streams = {
        "news_cont": raw_records([news], "news"),
        "wiki_cont": raw_records(wiki_paths, "wiki"),
        "web_qa": pair_records([web], "web"),
        "belle_2m": pair_records([belle2], "belle", split),
        "belle_1m": pair_records([belle1], "belle", split),
        "baike_qa": pair_records([baike], "baike"),
        "translation": pair_records([translation], "translation"),
        "wmt": tsv_records(wmt, source_first=split != "train"),
    }
    return streams


def fixed(ids: Sequence[int], width: int, eos: int, pad: int) -> Tuple[torch.Tensor, int]:
    values = list(ids[: max(0, width - 1)]) + [eos]
    length = len(values)
    values.extend([pad] * (width - len(values)))
    return torch.tensor(values, dtype=torch.long), length


class FullCorpusPairs(IterableDataset):
    def __init__(self, root: Path, spm_model: str, split: str, seed: int, source_width: int, target_width: int):
        self.root, self.spm_model, self.split, self.seed = root, spm_model, split, seed
        self.source_width, self.target_width = source_width, target_width

    def __iter__(self):
        sp = spm.SentencePieceProcessor(model_file=self.spm_model)
        eos, pad = sp.eos_id(), sp.get_piece_size()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + worker_id * 100_003)
        streams = source_streams(self.root, self.split)
        raw_buffers = {"news_cont": deque(), "wiki_cont": deque()}
        while True:
            name = rng.choices(SOURCE_NAMES, weights=SOURCE_WEIGHTS)[0]
            if name.endswith("_cont"):
                buffer = raw_buffers[name]
                total = self.source_width + self.target_width
                while len(buffer) < total:
                    source_text, _ = next(streams[name])
                    buffer.extend(sp.encode(source_text, out_type=int))
                    buffer.append(eos)
                block = [buffer.popleft() for _ in range(total)]
                source_ids, target_ids = block[: self.source_width], block[self.source_width :]
                source = torch.tensor(source_ids, dtype=torch.long)
                target = torch.tensor(target_ids, dtype=torch.long)
                length = self.source_width
            else:
                source_text, target_text = next(streams[name])
                source_chars = max(256, self.source_width * ENCODE_CHARS_PER_TOKEN)
                target_chars = max(256, self.target_width * ENCODE_CHARS_PER_TOKEN)
                source_ids = sp.encode(source_text[:source_chars], out_type=int)
                target_ids = sp.encode(target_text[:target_chars], out_type=int)
                source, length = fixed(source_ids, self.source_width, eos, pad)
                target, _ = fixed(target_ids, self.target_width, eos, pad)
            yield source, torch.tensor(length), target, torch.tensor(SOURCE_NAMES.index(name))


def collate(batch):
    return tuple(torch.stack(items) for items in zip(*batch))


def make_loader(args, split: str, seed: int, batch: int | None = None) -> DataLoader:
    workers = args.num_workers if split == "train" else 0
    return DataLoader(
        FullCorpusPairs(Path(args.data_root), args.spm_model, split, seed, args.source_width, args.target_width),
        batch_size=batch or args.batch, collate_fn=collate, num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=args.device.startswith("cuda"),
    )


def load_warm(args):
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = repair_probe.make_model(base_checkpoint, args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    repair_checkpoint = torch.load(args.repair_checkpoint, map_location="cpu", weights_only=False)
    kernel = repair_probe.RepairKernel(args.dim, model.depths, args.repair_hidden, False).to(args.device)
    kernel.load_state_dict(repair_checkpoint["parent_only"])
    return model, kernel


def make_damage_mask(batch: int, width: int, severity: str, rng: random.Random, device):
    count = 1 if severity == "single" else max(1, math.ceil(width * 0.25))
    mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
    for row in range(batch):
        start = rng.randrange(width - count + 1)
        mask[row, start : start + count] = True
    return mask


def encode_task(model, repair, source, length, rng: random.Random, task: str):
    leaf, root, details, fold_masks = model.encoder.fold(source, length)
    levels, level_masks = model.encoder.unfold(root, details, fold_masks)
    latent_loss = leaf.new_zeros(())
    metadata = {"depth": None, "severity": None}
    if task == "clean_leaf":
        memory, valid = leaf, fold_masks[0]
    elif task == "clean_multires":
        depth = rng.randrange(model.depths + 1)
        memory, valid = levels[depth], level_masks[depth]
        memory = memory + model.resolution.weight[depth]
        metadata["depth"] = depth
    elif task == "repair_leaf":
        depth = rng.randrange(model.depths)
        severity = rng.choice(("single", "quarter"))
        target_detail = details[depth]
        damage = make_damage_mask(source.shape[0], target_detail.shape[1], severity, rng, source.device)
        damaged = target_detail.masked_fill(damage[:, :, None], 0.0)
        parent = levels[model.depths - depth - 1]
        prediction = repair(parent, root, damaged, depth)
        local_details = list(details)
        local_details[depth] = torch.where(damage[:, :, None], prediction, target_detail)
        repaired_levels, repaired_masks = model.encoder.unfold(root, local_details, fold_masks)
        memory, valid = repaired_levels[-1], repaired_masks[-1]
        scale = target_detail[damage].square().mean().detach().clamp_min(1e-6)
        latent_loss = (prediction[damage] - target_detail[damage]).square().mean() / scale
        metadata.update(depth=depth, severity=severity)
    else:
        raise ValueError(task)
    if task != "clean_multires":
        memory = memory + model.resolution.weight[model.depths]
    return memory, valid, latent_loss, metadata


def task_choice(rng: random.Random) -> str:
    value = rng.random()
    if value < 0.50: return "repair_leaf"
    if value < 0.75: return "clean_multires"
    return "clean_leaf"


@torch.no_grad()
def evaluate(model, repair, args, sp, batches: int, generate: bool) -> dict:
    model.eval(); repair.eval()
    rng = random.Random(args.seed + 8801)
    stream = iter(make_loader(args, "valid", args.seed + 9000, args.eval_batch))
    totals = Counter(); examples = []; outputs = []
    for _ in range(batches):
        source, length, target, source_id = next(stream)
        source, length, target = source.to(args.device), length.to(args.device), target.to(args.device)
        leaf, root, details, fold_masks = model.encoder.fold(source, length)
        clean_memory = leaf + model.resolution.weight[model.depths]
        clean_logits = model.decoder.teacher(clean_memory, fold_masks[0], target, args.bos)
        clean_loss = F.cross_entropy(clean_logits.flatten(0, 1), target.flatten(), ignore_index=args.pad, reduction="sum")
        valid_tokens = int(target.ne(args.pad).sum())
        totals["tokens"] += valid_tokens; totals["clean_loss"] += float(clean_loss)
        shuffled = source.roll(1, dims=0); shuffled_length = length.roll(1, dims=0)
        shuffled_leaf, _, _, shuffled_masks = model.encoder.fold(shuffled, shuffled_length)
        shuffled_logits = model.decoder.teacher(shuffled_leaf + model.resolution.weight[model.depths], shuffled_masks[0], target, args.bos)
        totals["shuffle_loss"] += float(F.cross_entropy(shuffled_logits.flatten(0, 1), target.flatten(), ignore_index=args.pad, reduction="sum"))
        levels, _ = model.encoder.unfold(root, details, fold_masks)
        depth = rng.randrange(model.depths); detail = details[depth]
        damage_mask = make_damage_mask(source.shape[0], detail.shape[1], "quarter", rng, source.device)
        zero = torch.zeros_like(detail); damaged = detail.masked_fill(damage_mask[:, :, None], 0.0)
        damaged_details = list(details); damaged_details[depth] = damaged
        damaged_leaf, damaged_valid = model.encoder.unfold(root, damaged_details, fold_masks)
        parent = levels[model.depths - depth - 1]
        prediction = repair(parent, root, damaged, depth)
        repaired_details = list(details); repaired_details[depth] = torch.where(damage_mask[:, :, None], prediction, detail)
        repaired_leaf, repaired_valid = model.encoder.unfold(root, repaired_details, fold_masks)
        damaged_logits = model.decoder.teacher(damaged_leaf[-1] + model.resolution.weight[model.depths], damaged_valid[-1], target, args.bos)
        repaired_logits = model.decoder.teacher(repaired_leaf[-1] + model.resolution.weight[model.depths], repaired_valid[-1], target, args.bos)
        totals["damaged_loss"] += float(F.cross_entropy(damaged_logits.flatten(0, 1), target.flatten(), ignore_index=args.pad, reduction="sum"))
        totals["repaired_loss"] += float(F.cross_entropy(repaired_logits.flatten(0, 1), target.flatten(), ignore_index=args.pad, reduction="sum"))
        if generate and len(examples) < args.example_count:
            predicted = model.decoder.greedy(clean_memory, fold_masks[0], args.bos, args.eos, args.target_width)
            for row in range(min(source.shape[0], args.example_count - len(examples))):
                src_ids = base.clean(source[row].cpu().tolist(), args.eos, args.pad)
                ref_ids = base.clean(target[row].cpu().tolist(), args.eos, args.pad)
                out_ids = base.clean(predicted[row].cpu().tolist(), args.eos, args.pad)
                outputs.append(tuple(out_ids))
                examples.append({
                    "source_type": SOURCE_NAMES[int(source_id[row])],
                    "source": sp.decode(src_ids), "reference": sp.decode(ref_ids), "generated": sp.decode(out_ids),
                })
    tokens = max(1, totals["tokens"])
    clean = totals["clean_loss"] / tokens; damaged = totals["damaged_loss"] / tokens; repaired = totals["repaired_loss"] / tokens
    repair_fraction = (damaged - repaired) / max(1e-12, damaged - clean) if damaged > clean else 0.0
    nonempty = sum(bool(row) for row in outputs) / max(1, len(outputs)) if generate else None
    repeats = []
    for row in outputs:
        repeats.append(sum(row[i] == row[i-1] for i in range(1, len(row))) / max(1, len(row)-1))
    return {
        "clean_nll": clean,
        "shuffle_nll": totals["shuffle_loss"] / tokens,
        "source_shuffle_damage": totals["shuffle_loss"] / tokens - clean,
        "damaged_nll": damaged,
        "repaired_nll": repaired,
        "repair_fraction": repair_fraction,
        "nonempty_fraction": nonempty,
        "adjacent_repeat_fraction": sum(repeats) / max(1, len(repeats)) if generate else None,
        "unique_output_fraction": len(set(outputs)) / max(1, len(outputs)) if generate else None,
        "examples": examples,
    }


def atomic_torch_save(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, model, repair, optimizer, scaler, step, trace, counters, args, rng):
    atomic_torch_save({
        "step": step, "model": model.state_dict(), "repair": repair.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "trace": trace, "source_counts": dict(counters), "config": vars(args),
        "python_rng": random.getstate(), "task_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def write_status(output: Path, payload: dict):
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output / "status.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/nio/datasets")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--base-checkpoint", default="/home/nio/datasets/treeheap_checkpoints/checkpoint_annealed.pt")
    parser.add_argument("--repair-checkpoint", default="/home/nio/datasets/treeheap_checkpoints/repair_kernels.pt")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_full_repair_seq2seq")
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--eval-batch", type=int, default=24)
    parser.add_argument("--source-width", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--repair-hidden", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--repair-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=74101)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=10000)
    parser.add_argument("--valid-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--example-count", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.data_root.startswith("/mnt/nas"):
        raise ValueError("training hot path must use /home/nio/datasets, not NAS")
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    args.pad = sp.get_piece_size(); args.vocab = args.pad + 1
    args.bos = sp.bos_id(); args.eos = sp.eos_id(); args.dim = 256; args.hidden = 256
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    model, repair = load_warm(args)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": repair.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    rng = random.Random(args.seed + 400)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    latest = output / "checkpoint_latest.pt"; start_step = 0; trace = []; counters = Counter()
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); repair.load_state_dict(saved["repair"])
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"]); trace = list(saved["trace"]); counters.update(saved["source_counts"])
        random.setstate(saved["python_rng"]); rng.setstate(saved["task_rng"]); torch.set_rng_state(saved["torch_rng"])
        if saved.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(saved["cuda_rng"])
    train_stream = iter(make_loader(args, "train", args.seed))
    initial = evaluate(model, repair, args, sp, args.valid_batches, generate=True)
    started = time.time(); interval_started = started; interval_tokens = 0; finite = True
    for step in range(start_step + 1, args.steps + 1):
        model.train(); repair.train()
        source, length, target, source_ids = next(train_stream)
        source, length, target = source.to(args.device), length.to(args.device), target.to(args.device)
        for source_id in source_ids.tolist(): counters[SOURCE_NAMES[source_id]] += 1
        task = task_choice(rng); counters[f"task:{task}"] += source.shape[0]
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and args.device.startswith("cuda"))
        with amp_context:
            memory, valid, latent_loss, meta = encode_task(model, repair, source, length, rng, task)
            logits = model.decoder.teacher(memory, valid, target, args.bos)
            language_loss = base.ce(logits, target, args.pad)
            loss = language_loss + args.repair_loss_weight * latent_loss
        optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(repair.parameters()), 1.0)
        step_finite = bool(torch.isfinite(loss))
        if step == 1 or step % args.log_every == 0:
            step_finite = step_finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in list(model.parameters()) + list(repair.parameters())
            )
        finite = finite and step_finite
        scaler.step(optimizer); scaler.update()
        valid_target_tokens = int(target.ne(args.pad).sum())
        interval_tokens += valid_target_tokens
        counters["valid_target_tokens"] += valid_target_tokens
        if step == 1 or step % args.log_every == 0:
            now = time.time(); elapsed = now - interval_started
            row = {"step": step, "task": task, "train_nll": float(language_loss.detach()), "repair_nmse": float(latent_loss.detach()), "tokens_per_sec": interval_tokens / max(1e-9, elapsed), "elapsed_sec": now - started, **meta}
            print(json.dumps(row), flush=True); interval_started = now; interval_tokens = 0
        if step % args.valid_every == 0 or step == args.steps:
            metrics = evaluate(model, repair, args, sp, args.valid_batches, generate=True)
            row = {"step": step, "elapsed_sec": time.time() - started, **metrics}
            trace.append(row); print(json.dumps({"validation": row}, ensure_ascii=False), flush=True)
            (output / "trace.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in trace) + "\n", encoding="utf-8")
            (output / "examples_latest.json").write_text(json.dumps(metrics["examples"], indent=2, ensure_ascii=False), encoding="utf-8")
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(latest, model, repair, optimizer, scaler, step, trace, counters, args, rng)
            write_status(output, {"step": step, "steps": args.steps, "finite": finite, "elapsed_sec": time.time() - started, "source_counts": dict(counters), "latest_validation": trace[-1] if trace else None})
    final = evaluate(model, repair, args, sp, args.valid_batches * 2, generate=True)
    # Exact reload audit uses the state-only path to avoid perturbing the live model.
    audit_model, audit_repair = load_warm(args)
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    audit_model.load_state_dict(saved["model"]); audit_repair.load_state_dict(saved["repair"])
    reloaded = evaluate(audit_model, audit_repair, args, sp, args.valid_batches * 2, generate=False)
    reload_delta = abs(final["clean_nll"] - reloaded["clean_nll"])
    processed_target_tokens = counters["valid_target_tokens"]
    gates = {
        "P1_finite": finite,
        "P2_data_pressure": all(counters[name] >= 1000 for name in SOURCE_NAMES) and processed_target_tokens >= 500_000_000,
        "P3_learning": initial["clean_nll"] - final["clean_nll"] >= 0.20,
        "P4_repair": final["repaired_nll"] - final["clean_nll"] <= 0.50 and final["repair_fraction"] >= 0.50,
        "P5_generation": final["nonempty_fraction"] >= 0.90 and final["adjacent_repeat_fraction"] <= 0.35 and final["unique_output_fraction"] >= 0.50,
        "P6_conditioning": final["source_shuffle_damage"] >= 0.20,
        "P7_resume": reload_delta <= 1e-5,
    }
    summary = {"claim": "S3-FULL-REPAIR-SEQ2SEQ-C01", "host": socket.gethostname(), "config": vars(args), "initial": initial, "final": final, "source_counts": dict(counters), "processed_valid_target_tokens": processed_target_tokens, "trace": trace, "reload_nll_delta": reload_delta, "gates": gates, "decision": "supported" if all(gates.values()) else "partial or rejected; inspect gates"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text("# Full-Corpus Repair-Aware Seq2Seq\n\nAll training data is read from `/home/nio/datasets`. See `status.json`, `trace.jsonl`, and `summary.json`.\n", encoding="utf-8")
    print(json.dumps({"final": final, "gates": gates, "decision": summary["decision"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
