#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


ARMS = ("probe-only", "predictive-gradient")


def numeric(value: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if not match:
        raise ValueError(f"no numeric value in {value!r}")
    return float(match.group(0))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_trace(path: Path):
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "train":
                events.append(row)
    return events


def arm_throughput(root: Path, arm: str, warmup_step: int):
    summary = read_json(root / arm / "summary.json")
    events = read_trace(root / arm / "trace.jsonl")
    start = min((row for row in events if row["step"] >= warmup_step), key=lambda row: row["step"])
    end = max(events, key=lambda row: row["step"])
    seconds = end["elapsed_seconds"] - start["elapsed_seconds"]
    tokens = end["tokens"] - start["tokens"]
    if seconds <= 0 or tokens <= 0:
        raise ValueError(f"invalid throughput interval for {arm}")
    return {
        "step_start": start["step"], "step_end": end["step"],
        "tokens": tokens, "seconds": seconds,
        "target_tokens_per_second": tokens / seconds,
        "steps_per_second": (end["step"] - start["step"]) / seconds,
        "final_nll": summary["final"]["nll"],
        "final_predictive_mse": summary["final"]["predictive_mse_mean"],
        "gates": summary["gates"],
    }


def gpu_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            rows.append({
                "power_limit": numeric(row[2]), "power": numeric(row[3]),
                "temperature": numeric(row[4]), "memory_mib": numeric(row[5]),
                "memory_total_mib": numeric(row[6]), "utilization": numeric(row[7]),
            })
    if not rows:
        raise ValueError(f"no GPU samples in {path}")
    active = [row["utilization"] for row in rows if row["utilization"] >= 5]
    return {
        "samples": len(rows),
        "utilization_mean": sum(row["utilization"] for row in rows) / len(rows),
        "utilization_active_mean": sum(active) / len(active) if active else 0.0,
        "utilization_max": max(row["utilization"] for row in rows),
        "memory_max_gib": max(row["memory_mib"] for row in rows) / 1024.0,
        "power_mean_watts": sum(row["power"] for row in rows) / len(rows),
        "power_max_watts": max(row["power"] for row in rows),
        "power_limit_max_watts": max(row["power_limit"] for row in rows),
        "temperature_max_celsius": max(row["temperature"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=(16, 32, 64))
    parser.add_argument("--warmup-step", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.root)
    results = []
    for batch in args.batches:
        run = root / f"batch_{batch}"
        exit_code_path = run / "exit_code.txt"
        if not exit_code_path.exists():
            results.append({
                "batch_size": batch, "exit_code": None,
                "eligible": False, "status": "not_run_after_lower_batch_failure",
            })
            continue
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
        row = {"batch_size": batch, "exit_code": exit_code, "eligible": False}
        if exit_code == 0:
            arms = {arm: arm_throughput(run, arm, args.warmup_step) for arm in ARMS}
            gpu = gpu_summary(run / "gpu_samples.csv")
            total_tokens = sum(arms[arm]["tokens"] for arm in ARMS)
            total_seconds = sum(arms[arm]["seconds"] for arm in ARMS)
            finite = all(
                math.isfinite(arms[arm]["final_nll"])
                and math.isfinite(arms[arm]["final_predictive_mse"])
                for arm in ARMS
            )
            gates_ok = all(all(arms[arm]["gates"].values()) for arm in ARMS)
            safety = {
                "finite": finite,
                "arm_gates": gates_ok,
                "memory_headroom": gpu["memory_max_gib"] <= 20.0,
                "temperature": gpu["temperature_max_celsius"] <= 80.0,
                "power_limit": gpu["power_limit_max_watts"] <= 270.5,
            }
            row.update({
                "arms": arms, "gpu": gpu,
                "combined_target_tokens_per_second": total_tokens / total_seconds,
                "safety": safety, "eligible": all(safety.values()),
            })
        results.append(row)
    eligible = [row for row in results if row["eligible"]]
    recommendation = max(
        eligible, key=lambda row: row["combined_target_tokens_per_second"], default=None,
    )
    report = {
        "claim": "S3-PREDICTIVE-STATE-ENGINE-CANARY-D12-E1",
        "results": results,
        "recommended_batch": recommendation["batch_size"] if recommendation else None,
        "decision": "supported" if recommendation else "no_safe_candidate",
        "selection_rule": "highest safe combined target tokens per second",
    }
    path = root / "throughput_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
