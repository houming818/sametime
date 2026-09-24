"""Build cumulative corpus-count snapshots with one fixed sealed test field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_english(path: Path) -> Iterable[Tuple[int, bytes, str]]:
    with path.open("rb") as handle:
        for raw_index, raw in enumerate(handle):
            try:
                decoded = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                continue
            fields = decoded.split("\t")
            if len(fields) < 2 or not fields[1].strip():
                continue
            yield raw_index, raw, fields[1]


def add_sentence(
    matrix: np.ndarray,
    ids: Sequence[int],
    target_lookup: np.ndarray,
    context_lookup: np.ndarray,
    window: int,
) -> int:
    token_ids = np.asarray(ids, dtype=np.int64)
    pairs = 0
    for offset in range(1, window + 1):
        if len(token_ids) <= offset:
            break
        left = token_ids[:-offset]
        right = token_ids[offset:]

        left_target = target_lookup[left]
        right_context = context_lookup[right]
        mask = (left_target >= 0) & (right_context >= 0)
        np.add.at(matrix, (left_target[mask], right_context[mask]), 1)
        pairs += int(mask.sum())

        right_target = target_lookup[right]
        left_context = context_lookup[left]
        mask = (right_target >= 0) & (left_context >= 0)
        np.add.at(matrix, (right_target[mask], left_context[mask]), 1)
        pairs += int(mask.sum())
    return pairs


def parse_scales(raw: str) -> List[int]:
    scales = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not scales or scales[0] <= 0:
        raise ValueError("train scales must contain positive integers")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--spm-model", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-scales", default="50000,100000,200000,500000,1000000")
    parser.add_argument("--test-lines", type=int, default=100000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-sentence-tokens", type=int, default=96)
    args = parser.parse_args()

    data_path = Path(args.data)
    spm_path = Path(args.spm_model)
    vocab_path = Path(args.vocab_json)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.train_scales)
    max_train = scales[-1]
    required_valid = max_train + args.test_lines

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    target_ids = [int(value) for value in vocab["target_ids"]]
    context_ids = [int(value) for value in vocab["context_ids"]]
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    lookup_size = max(sp.get_piece_size(), max(target_ids), max(context_ids)) + 1
    target_lookup = np.full(lookup_size, -1, dtype=np.int64)
    context_lookup = np.full(lookup_size, -1, dtype=np.int64)
    target_lookup[np.asarray(target_ids)] = np.arange(len(target_ids))
    context_lookup[np.asarray(context_ids)] = np.arange(len(context_ids))

    train = np.zeros((len(target_ids), len(context_ids)), dtype=np.int64)
    test = np.zeros_like(train)
    train_pairs = 0
    test_pairs = 0
    valid_lines = 0
    last_raw_index = -1
    train_digest = hashlib.sha256()
    test_digest = hashlib.sha256()
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_meta: Dict[int, Dict[str, object]] = {}

    for raw_index, raw, text in iter_english(data_path):
        if valid_lines >= required_valid:
            break
        ids = [token for token in sp.encode(text, out_type=int) if token >= 4]
        ids = ids[: args.max_sentence_tokens]
        if valid_lines < max_train:
            train_digest.update(raw)
            train_pairs += add_sentence(train, ids, target_lookup, context_lookup, args.window)
            valid_lines += 1
            if valid_lines in scales:
                snapshots[valid_lines] = train.copy()
                snapshot_meta[valid_lines] = {
                    "train_valid_lines": valid_lines,
                    "train_pairs": train_pairs,
                    "train_region_sha256": train_digest.copy().hexdigest(),
                    "last_raw_line_index": raw_index,
                }
                print(json.dumps({"snapshot": valid_lines, **snapshot_meta[valid_lines]}), flush=True)
        else:
            test_digest.update(raw)
            test_pairs += add_sentence(test, ids, target_lookup, context_lookup, args.window)
            valid_lines += 1
            if (valid_lines - max_train) % 10000 == 0:
                print(json.dumps({"sealed_test_lines": valid_lines - max_train, "test_pairs": test_pairs}), flush=True)
        last_raw_index = raw_index

    if valid_lines != required_valid:
        raise RuntimeError(f"needed {required_valid} valid lines, found {valid_lines}")
    if set(snapshots) != set(scales):
        raise RuntimeError(f"missing snapshots: {sorted(set(scales) - set(snapshots))}")

    test_tensor = torch.from_numpy(test).to(torch.float64)
    test_nonempty_rows = int((test_tensor.sum(dim=1) > 0).sum().item())
    if test_nonempty_rows != len(target_ids):
        raise RuntimeError(f"sealed test has {test_nonempty_rows}/{len(target_ids)} nonempty target rows")

    arms = []
    for scale in scales:
        scale_dir = out / f"train_{scale:07d}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "fixed_vocab_fixed_sealed_test_context_counts_v1",
            "train": torch.from_numpy(snapshots[scale]).to(torch.float64),
            "test": test_tensor,
            "target_ids": target_ids,
            "context_ids": context_ids,
            "train_valid_lines": scale,
            "sealed_test_start_valid_line": max_train,
            "sealed_test_valid_lines": args.test_lines,
        }
        counts_path = scale_dir / "context_counts.pt"
        torch.save(payload, counts_path)
        arm = {
            **snapshot_meta[scale],
            "sealed_test_start_valid_line": max_train,
            "sealed_test_valid_lines": args.test_lines,
            "sealed_test_pairs": test_pairs,
            "sealed_test_sha256": test_digest.hexdigest(),
            "sealed_test_nonempty_target_rows": test_nonempty_rows,
            "counts_path": str(counts_path),
            "counts_sha256": file_sha256(counts_path),
        }
        (scale_dir / "manifest.json").write_text(json.dumps(arm, indent=2) + "\n", encoding="utf-8")
        arms.append(arm)

    manifest = {
        "claim": "S1-F-SCALE-A14-C01",
        "format": "fixed_vocab_fixed_sealed_test_corpus_ladder_v1",
        "data": str(data_path),
        "data_size_bytes": data_path.stat().st_size,
        "spm_model": str(spm_path),
        "spm_sha256": file_sha256(spm_path),
        "vocab_json": str(vocab_path),
        "vocab_sha256": file_sha256(vocab_path),
        "target_tokens": len(target_ids),
        "context_tokens": len(context_ids),
        "window": args.window,
        "max_sentence_tokens": args.max_sentence_tokens,
        "train_scales": scales,
        "sealed_test_start_valid_line": max_train,
        "sealed_test_valid_lines": args.test_lines,
        "sealed_test_pairs": test_pairs,
        "sealed_test_sha256": test_digest.hexdigest(),
        "last_raw_line_index": last_raw_index,
        "arms": arms,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
