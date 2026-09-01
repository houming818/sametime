#!/usr/bin/env python3
"""Streaming integrity and duplicate-topology audit for Nio corpus releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
LENGTH_BOUNDS = (0, 4, 15, 31, 63, 127, 255, 511, 1023)


class StreamStats:
    def __init__(self, seed: int, reservoir_size: int = 100_000) -> None:
        self.count = 0
        self.blank = 0
        self.chars = 0
        self.han = 0
        self.latin = 0
        self.replacement = 0
        self.histogram: Counter[str] = Counter()
        self.reservoir: list[int] = []
        self.reservoir_size = reservoir_size
        self.random = random.Random(seed)

    def add(self, text: str) -> tuple[float, float]:
        length = len(text)
        self.count += 1
        self.blank += int(not text.strip())
        self.chars += length
        han = len(HAN_RE.findall(text))
        latin = len(LATIN_RE.findall(text))
        self.han += han
        self.latin += latin
        self.replacement += text.count("\ufffd")
        self.histogram[self._bucket(length)] += 1
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(length)
        else:
            index = self.random.randrange(self.count)
            if index < self.reservoir_size:
                self.reservoir[index] = length
        denominator = max(length, 1)
        return han / denominator, latin / denominator

    @staticmethod
    def _bucket(length: int) -> str:
        if length == 0:
            return "0"
        previous = 1
        for bound in LENGTH_BOUNDS[1:]:
            if length <= bound:
                return f"{previous}-{bound}"
            previous = bound + 1
        return "1024+"

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.reservoir)

        def quantile(q: float) -> int | None:
            if not ordered:
                return None
            position = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
            return ordered[position]

        return {
            "count": self.count,
            "blank": self.blank,
            "characters": self.chars,
            "han_characters": self.han,
            "ascii_latin_characters": self.latin,
            "replacement_characters": self.replacement,
            "han_ratio": self.han / max(self.chars, 1),
            "ascii_latin_ratio": self.latin / max(self.chars, 1),
            "length_histogram": dict(sorted(self.histogram.items())),
            "length_quantiles": {
                "p00": quantile(0.0),
                "p25": quantile(0.25),
                "p50": quantile(0.5),
                "p75": quantile(0.75),
                "p90": quantile(0.9),
                "p95": quantile(0.95),
                "p99": quantile(0.99),
                "p100": quantile(1.0),
            },
            "quantile_sample_size": len(ordered),
        }


def digest128(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def emit_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def base_summary(path: Path, started: float, sha256: str, rows: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "claim": "NIO-CORPUS-INTEGRITY-C01",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "input": str(path.resolve()),
        "input_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "sha256": sha256,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def scan_mono(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.jsonl"
    progress.unlink(missing_ok=True)
    started = time.monotonic()
    file_hash = hashlib.sha256()
    stats = StreamStats(args.seed)
    seen_ids: set[str] = set()
    seen_text: set[bytes] = set()
    duplicate_ids = 0
    duplicate_text = 0
    parse_errors = 0
    schema_errors = 0
    sources: Counter[str] = Counter()
    duplicate_samples: list[dict[str, Any]] = []

    with path.open("rb") as handle:
        for row, raw in enumerate(handle, 1):
            file_hash.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parse_errors += 1
                continue
            identifier = record.get("id")
            text = record.get("text")
            if not isinstance(identifier, str) or not isinstance(text, str):
                schema_errors += 1
                continue
            source = identifier.split(":", 1)[0]
            sources[source] += 1
            if identifier in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(identifier)
            token = digest128(text)
            if token in seen_text:
                duplicate_text += 1
                if len(duplicate_samples) < 20:
                    duplicate_samples.append({"row": row, "id": identifier, "text": text[:240]})
            else:
                seen_text.add(token)
            stats.add(text)
            if row % args.progress_every == 0:
                emit_progress(progress, {
                    "kind": "mono",
                    "row": row,
                    "elapsed_seconds": time.monotonic() - started,
                    "duplicate_text": duplicate_text,
                    "parse_errors": parse_errors,
                    "schema_errors": schema_errors,
                })

    rows = row if "row" in locals() else 0
    summary = base_summary(path, started, file_hash.hexdigest(), rows)
    summary.update({
        "kind": "mono",
        "expected_rows": args.expected_rows,
        "expected_bytes": args.expected_bytes,
        "parse_errors": parse_errors,
        "schema_errors": schema_errors,
        "duplicate_ids": duplicate_ids,
        "duplicate_text_digests": duplicate_text,
        "duplicate_text_digest_rate": duplicate_text / max(rows, 1),
        "source_counts": dict(sources.most_common()),
        "text": stats.summary(),
        "duplicate_samples": duplicate_samples,
    })
    summary["gates"] = {
        "row_count": rows == args.expected_rows,
        "byte_count": path.stat().st_size == args.expected_bytes,
        "parse_integrity": parse_errors == 0 and schema_errors == 0,
        "finite": all(math.isfinite(value) for value in (
            summary["duplicate_text_digest_rate"],
            summary["text"]["han_ratio"],
            summary["text"]["ascii_latin_ratio"],
        )),
    }
    summary["passed"] = all(summary["gates"].values())
    write_json(output / "summary.json", summary)
    return summary


def scan_parallel(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.jsonl"
    progress.unlink(missing_ok=True)
    started = time.monotonic()
    file_hash = hashlib.sha256()
    source_stats = StreamStats(args.seed)
    target_stats = StreamStats(args.seed + 1)
    seen_pairs: set[bytes] = set()
    seen_source: set[bytes] = set()
    seen_target: set[bytes] = set()
    duplicate_pairs = 0
    duplicate_source = 0
    duplicate_target = 0
    malformed = 0
    low_source_han = 0
    low_target_latin = 0
    duplicate_samples: list[dict[str, Any]] = []

    with path.open("rb") as handle:
        for row, raw in enumerate(handle, 1):
            file_hash.update(raw)
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                malformed += 1
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                malformed += 1
                continue
            source, target = fields
            source_han, _ = source_stats.add(source)
            _, target_latin = target_stats.add(target)
            low_source_han += int(len(source) >= 10 and source_han < 0.2)
            low_target_latin += int(len(target) >= 10 and target_latin < 0.2)
            source_digest = digest128(source)
            target_digest = digest128(target)
            pair_digest = digest128(source + "\u0000" + target)
            if pair_digest in seen_pairs:
                duplicate_pairs += 1
                if len(duplicate_samples) < 20:
                    duplicate_samples.append({
                        "row": row,
                        "source": source[:200],
                        "target": target[:200],
                    })
            else:
                seen_pairs.add(pair_digest)
            duplicate_source += int(source_digest in seen_source)
            duplicate_target += int(target_digest in seen_target)
            seen_source.add(source_digest)
            seen_target.add(target_digest)
            if row % args.progress_every == 0:
                emit_progress(progress, {
                    "kind": "parallel",
                    "row": row,
                    "elapsed_seconds": time.monotonic() - started,
                    "duplicate_pairs": duplicate_pairs,
                    "malformed": malformed,
                })

    rows = row if "row" in locals() else 0
    summary = base_summary(path, started, file_hash.hexdigest(), rows)
    summary.update({
        "kind": "parallel",
        "expected_rows": args.expected_rows,
        "expected_bytes": args.expected_bytes,
        "malformed": malformed,
        "duplicate_pair_digests": duplicate_pairs,
        "duplicate_source_digests": duplicate_source,
        "duplicate_target_digests": duplicate_target,
        "duplicate_pair_digest_rate": duplicate_pairs / max(rows, 1),
        "duplicate_source_digest_rate": duplicate_source / max(rows, 1),
        "duplicate_target_digest_rate": duplicate_target / max(rows, 1),
        "low_source_han_rows": low_source_han,
        "low_target_ascii_latin_rows": low_target_latin,
        "source": source_stats.summary(),
        "target": target_stats.summary(),
        "duplicate_samples": duplicate_samples,
    })
    summary["gates"] = {
        "row_count": rows == args.expected_rows,
        "byte_count": path.stat().st_size == args.expected_bytes,
        "parse_integrity": malformed == 0,
        "finite": all(math.isfinite(value) for value in (
            summary["duplicate_pair_digest_rate"],
            summary["source"]["han_ratio"],
            summary["target"]["ascii_latin_ratio"],
        )),
    }
    summary["passed"] = all(summary["gates"].values())
    write_json(output / "summary.json", summary)
    return summary


def merge(args: argparse.Namespace) -> dict[str, Any]:
    mono = json.loads(Path(args.mono_summary).read_text(encoding="utf-8"))
    parallel = json.loads(Path(args.parallel_summary).read_text(encoding="utf-8"))
    combined = {
        "claim": "NIO-CORPUS-INTEGRITY-C01",
        "host": socket.gethostname(),
        "generated_at_unix": time.time(),
        "passed": bool(mono.get("passed")) and bool(parallel.get("passed")),
        "registered_total_rows": 10_277_334,
        "observed_total_rows": mono.get("rows", 0) + parallel.get("rows", 0),
        "mono": mono,
        "parallel": parallel,
    }
    write_json(Path(args.output), combined)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("mono", "parallel", "merge"), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-bytes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=16001)
    parser.add_argument("--mono-summary")
    parser.add_argument("--parallel-summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "mono":
        result = scan_mono(args)
    elif args.kind == "parallel":
        result = scan_parallel(args)
    else:
        result = merge(args)
    print(json.dumps({
        "claim": result["claim"],
        "kind": args.kind,
        "passed": result.get("passed"),
        "rows": result.get("rows", result.get("observed_total_rows")),
    }, sort_keys=True))
    if not result.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
