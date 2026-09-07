#!/usr/bin/env python3
"""Summarize the bounded E02 batch and prefetch throughput experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CASES = ("b64-raw", "b64-prefetch2", "b96-prefetch2", "b128-prefetch2")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def steady_rate(trace_path: Path) -> dict:
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(events) < 3:
        raise RuntimeError(f"too few trace events in {trace_path}")
    first = events[1]
    last = events[-1]
    seconds = float(last["elapsed_seconds"]) - float(first["elapsed_seconds"])
    tokens = int(last["processed_tokens"]) - int(first["processed_tokens"])
    examples = int(last["processed_examples"]) - int(first["processed_examples"])
    if seconds <= 0 or tokens <= 0 or examples <= 0:
        raise RuntimeError(f"invalid throughput interval in {trace_path}")
    return {
        "start_step": int(first["step"]),
        "end_step": int(last["step"]),
        "seconds": seconds,
        "tokens": tokens,
        "examples": examples,
        "tokens_per_second": tokens / seconds,
        "examples_per_second": examples / seconds,
    }


def peak_memory(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower().startswith("timestamp"):
                continue
            try:
                values.append(float(row[6].split()[0]))
            except (IndexError, ValueError):
                continue
    if not values:
        raise RuntimeError(f"no GPU memory samples in {path}")
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = {}
    for name in CASES:
        case = args.root / name
        summary = read_json(case / "summary.json")
        if not all(summary["gates"].values()):
            raise RuntimeError(f"failed gates for {name}: {summary['gates']}")
        rate = steady_rate(case / "trace.jsonl")
        rows[name] = {
            "batch_size": int(summary["initial_valid"]["count"] * 0 + summary["segment_examples"] // (summary["step"] - summary["start_step"])),
            "prefetch_batches": 0 if name == "b64-raw" else 2,
            **rate,
            "peak_memory_mib": peak_memory(case / "gpu_samples.csv"),
            "reload_nll_delta": float(summary["reload_nll_delta"]),
        }
        if not all(math.isfinite(value) for key, value in rows[name].items() if isinstance(value, float)):
            raise RuntimeError(f"non-finite result for {name}")

    baseline = rows["b64-raw"]["tokens_per_second"]
    for row in rows.values():
        row["speedup_vs_b64_raw"] = row["tokens_per_second"] / baseline
    supported = [
        name for name, row in rows.items()
        if row["peak_memory_mib"] <= 23_000 and row["reload_nll_delta"] < 1e-9
    ]
    selected = max(supported, key=lambda name: rows[name]["tokens_per_second"])
    result = {
        "claim": "S3-EPOCH-REPEAT-THROUGHPUT-E02",
        "status": "complete",
        "cases": rows,
        "selection": selected,
        "selection_rule": "highest steady token/s among finite reload-safe cases below 23000 MiB",
    }
    (args.root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
