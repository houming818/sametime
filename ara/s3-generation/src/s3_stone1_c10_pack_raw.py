#!/usr/bin/env python3
"""Deterministically pack all C10 raw documents with the frozen 32K tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from collections import Counter, deque
from pathlib import Path
from typing import Iterator

import numpy as np
import sentencepiece as spm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path, split: str) -> list[tuple[str, Path]]:
    base = root / "pretrain" / "Chinese-Train-Datasets"
    if split == "train":
        rows = [
            ("news", base / "new2016zh" / "news2016zh_train.json"),
            *(('wiki', path) for path in sorted((base / "wiki_zh").rglob("wiki_*"))),
            ("web", base / "webtext2019zh" / "webtext_zh_train.json"),
        ]
    else:
        rows = [
            ("news", base / "new2016zh" / "news2016zh_valid.json"),
            ("web", base / "webtext2019zh" / "webtext_zh_valid.json"),
        ]
    return [(source, path) for source, path in rows if path.is_file()]


def row_text(row: dict, source: str) -> str:
    if source == "wiki":
        return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
    return (str(row.get("title", "")) + "\n" + str(row.get("content", ""))).strip()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/nio/datasets")
    parser.add_argument(
        "--spm-model",
        default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/nio/datasets/derived/stone1_c10/raw_32k_256",
    )
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_stone1_c10_raw_pack",
    )
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-docs", type=int, default=256)
    parser.add_argument("--tokenize-threads", type=int, default=8)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--min-chars", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-docs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    evidence = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    state_path = output / f"state-{args.split}.json"
    manifest_path = output / f"manifest-{args.split}.json"
    spm_path = Path(args.spm_model).resolve()
    tokenizer_hash = sha256(spm_path)
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    pieces = sp.get_piece_size()
    if pieces >= 65535 or args.width < 2:
        raise ValueError("uint16 packing requires vocab < 65535 and width >= 2")
    pad, eos = pieces, sp.eos_id()
    files = source_files(root, args.split)
    file_contract = [
        {
            "index": index,
            "source": source,
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for index, (source, path) in enumerate(files)
    ]
    contract = {
        "split": args.split,
        "width": args.width,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_vocab": pieces,
        "pad": pad,
        "eos": eos,
        "files": file_contract,
    }

    cursor_file = 0
    cursor_line = 0
    token_buffer: deque[int] = deque()
    shard_rows: list[np.ndarray] = []
    shards: list[dict] = []
    counters: Counter = Counter()
    started = time.time()
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("resume contract differs from current corpus/tokenizer")
        cursor_file = state["cursor_file"]
        cursor_line = state["cursor_line"]
        token_buffer.extend(state["token_buffer"])
        shards = state["shards"]
        counters.update(state["counters"])
        for orphan in output.glob(f"{args.split}-*.npy"):
            index = int(orphan.stem.rsplit("-", 1)[1])
            if index >= len(shards):
                orphan.unlink()
        print(json.dumps({"resume": True, "cursor_file": cursor_file,
                          "cursor_line": cursor_line, "shards": len(shards)}), flush=True)
    elif state_path.exists() or manifest_path.exists():
        raise FileExistsError("output exists; use --resume or choose a clean output directory")

    def save_state(next_file: int, next_line: int) -> None:
        atomic_json(state_path, {
            "contract": contract,
            "cursor_file": next_file,
            "cursor_line": next_line,
            "token_buffer": list(token_buffer),
            "shards": shards,
            "counters": dict(counters),
            "updated_at": time.time(),
        })

    def flush_shard() -> None:
        if not shard_rows:
            return
        index = len(shards)
        final_path = output / f"{args.split}-{index:05d}.npy"
        temporary = output / f".{final_path.name}.tmp.npy"
        array = np.stack(shard_rows).astype(np.uint16, copy=False)
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, final_path)
        row = {
            "path": final_path.name,
            "rows": int(array.shape[0]),
            "bytes": final_path.stat().st_size,
            "sha256": sha256(final_path),
        }
        shards.append(row)
        shard_rows.clear()
        print(json.dumps({"shard": index, "rows": row["rows"],
                          "total_blocks": counters["blocks"]}), flush=True)

    def accept_encoded(encoded_rows: list[list[int]], sources: list[str]) -> None:
        for ids, source in zip(encoded_rows, sources):
            ids.append(eos)
            counters[f"{source}_accepted_docs"] += 1
            counters[f"{source}_tokens"] += len(ids)
            token_buffer.extend(ids)
            while len(token_buffer) >= args.width:
                shard_rows.append(np.fromiter(
                    (token_buffer.popleft() for _ in range(args.width)),
                    dtype=np.uint16, count=args.width,
                ))
                counters["blocks"] += 1
                if len(shard_rows) >= args.shard_rows:
                    flush_shard()

    docs: list[str] = []
    sources: list[str] = []
    processed_this_run = 0
    stop_requested = False
    resume_file = cursor_file
    resume_line = cursor_line
    for file_index in range(cursor_file, len(files)):
        source, path = files[file_index]
        start_line = cursor_line if file_index == cursor_file else 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                if line_index < start_line:
                    continue
                counters[f"{source}_physical_rows"] += 1
                try:
                    value = json.loads(line)
                    text = row_text(value, source) if isinstance(value, dict) else ""
                except (json.JSONDecodeError, TypeError, ValueError):
                    counters[f"{source}_invalid_rows"] += 1
                    text = ""
                if len(text) >= args.min_chars and text.count("\ufffd") <= 2:
                    docs.append(text)
                    sources.append(source)
                else:
                    counters[f"{source}_filtered_rows"] += 1
                processed_this_run += 1
                next_file, next_line = file_index, line_index + 1
                resume_file, resume_line = next_file, next_line
                if len(docs) >= args.batch_docs:
                    encoded = sp.encode(
                        docs, out_type=int, num_threads=args.tokenize_threads,
                    )
                    accept_encoded(encoded, sources)
                    docs.clear()
                    sources.clear()
                    save_state(next_file, next_line)
                if args.max_docs and processed_this_run >= args.max_docs:
                    stop_requested = True
                    break
        if docs:
            encoded = sp.encode(docs, out_type=int, num_threads=args.tokenize_threads)
            accept_encoded(encoded, sources)
            docs.clear()
            sources.clear()
        if stop_requested:
            save_state(resume_file, resume_line)
            break
        resume_file, resume_line = file_index + 1, 0
        save_state(resume_file, resume_line)
        cursor_line = 0

    complete = not stop_requested and (not files or file_index + 1 == len(files))
    if complete and token_buffer:
        valid = len(token_buffer)
        final = list(token_buffer) + [pad] * (args.width - valid)
        shard_rows.append(np.asarray(final, dtype=np.uint16))
        counters["blocks"] += 1
        counters["final_block_valid_tokens"] = valid
        token_buffer.clear()
    flush_shard()
    save_state(len(files) if complete else resume_file, 0 if complete else resume_line)

    manifest = {
        "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
        "host": socket.gethostname(),
        "format": "npy uint16 [rows, 256]; first 128 source, last 128 target",
        "contract": contract,
        "shards": shards,
        "counters": dict(counters),
        "complete_source_pass": complete,
        "elapsed_sec_this_run": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(evidence / f"summary-{args.split}.json", manifest)
    print(json.dumps({
        "complete_source_pass": complete,
        "files": len(files),
        "blocks": counters["blocks"],
        "tokens": sum(value for key, value in counters.items() if key.endswith("_tokens")),
        "shards": len(shards),
        "elapsed_sec": manifest["elapsed_sec_this_run"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
