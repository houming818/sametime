#!/usr/bin/env python3
"""Run STONE-1 C03's capacity-versus-update-budget audit."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_private_protocol_data_dose as data_dose
import s3_stone1_canonical_codec as c02
import s3_stone1_private_protocol as c01


ARM_CONFIGS = {
    "base_28m_long": {"dim": 192, "hidden": 256, "steps": 31_250},
    "balanced_50m_equal": {"dim": 320, "hidden": 512, "steps": 15_625},
}
EXPECTED_PARAMETERS = {
    "base_28m_long": 27_620_482,
    "balanced_50m_equal": 50_267_778,
}
BASELINE_ARM = "base_28m"


def parameter_count(vocab: int, dim: int, hidden: int) -> int:
    """Closed-form parameter count for the current C02 model at heap width 64."""
    return (
        3 * vocab * dim + vocab * hidden + vocab
        + 8 * hidden * dim + 3 * hidden * hidden
        + 10 * dim * dim + 19 * dim + 6 * hidden + 1
    )


def load_baseline(path: str) -> dict:
    baseline = json.loads(Path(path).read_text(encoding="utf-8"))
    if baseline.get("experiment_id") != "s3_stone1_canonical_codec":
        raise ValueError("baseline is not the formal C02 summary")
    return baseline


def verify_frozen_platform(manifest: dict, baseline: dict, smoke: bool) -> dict:
    current = manifest["splits"]
    frozen = baseline["dataset"]["splits"]
    checks = {
        "tokenizer_sha256": (
            manifest["tokenizer"]["sha256"]
            == baseline["dataset"]["tokenizer"]["sha256"]
        ),
    }
    if not smoke:
        checks.update({
            "train_1m_sha256": (
                current["train_sha256"]["1000000"]
                == frozen["train_sha256"]["1000000"]
            ),
            "validation_sha256": (
                current["validation_sha256"] == frozen["validation_sha256"]
            ),
            "test_sha256": current["test_sha256"] == frozen["test_sha256"],
        })
    if not all(checks.values()):
        raise ValueError(f"frozen platform mismatch: {checks}")
    return checks


def final_training_window(trace_path: Path) -> float:
    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values = [row for row in rows if row.get("train_nll_window") is not None]
    return float(values[-1]["train_nll_window"])


def aggregate(results: Sequence[dict]) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for arm in ARM_CONFIGS:
        rows = [row for row in results if row["arm"] == arm]
        nll = [row["test"]["nll"] for row in rows]
        output[arm] = {
            "nll_mean": statistics.mean(nll),
            "nll_std": statistics.pstdev(nll),
            "bits_per_token_mean": statistics.mean(nll) / math.log(2),
            "bleu4_mean": statistics.mean(
                row["test"]["token_bleu4"] for row in rows
            ),
            "nonempty_mean": statistics.mean(
                row["test"]["nonempty"] for row in rows
            ),
            "severe_repetition_mean": statistics.mean(
                row["test"]["severe_repetition_rate"] for row in rows
            ),
            "final_train_nll_mean": statistics.mean(
                row["final_train_nll_window"] for row in rows
            ),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
            "parameters": rows[0]["parameters"],
            "trainable_parameters": rows[0]["trainable_parameters"],
            "peak_vram_max": max(row["peak_vram_bytes"] for row in rows),
        }
    return output


def decide(
    baseline: dict, aggregate_rows: dict, intervention: dict,
    results: Sequence[dict], smoke: bool,
):
    old = baseline["aggregate"]["canonical_learned"]
    long = aggregate_rows["base_28m_long"]
    large = aggregate_rows["balanced_50m_equal"]
    large_rows = [row for row in results if row["arm"] == "balanced_50m_equal"]
    closure_max = max(row["structure"]["closure_max_abs"] for row in large_rows)
    gates = {
        "C1_50m_nll_gain_at_least_0_08": old["nll_mean"] - large["nll_mean"] >= 0.08,
        "C2_50m_bleu_gain_at_least_0_75": large["bleu4_mean"] - old["bleu4_mean"] >= 0.75,
        "C3_50m_nll_std_at_most_0_08": large["nll_std"] <= 0.08,
        "C4_50m_within_0_02_of_28m_long": large["nll_mean"] <= long["nll_mean"] + 0.02,
        "C5_force_algebraic_damage_at_least_0_10": (
            intervention["damage_nll"]["force_algebraic"] >= 0.10
        ),
        "C6_address_damage_at_least_0_10": (
            intervention["damage_nll"]["address_swap"] >= 0.10
        ),
        "C7_depth_growth": (
            intervention["root_to_full_gain_nll"] >= 0.50
            and intervention["improving_depth_transitions"] >= 4
        ),
        "C8_closure_below_1e_5": closure_max < 1e-5,
        "C9_generation_nondegenerate": (
            large["nonempty_mean"] == 1.0
            and large["severe_repetition_mean"] <= 0.10
        ),
        "Q1_nll_at_most_3_90": large["nll_mean"] <= 3.90,
        "Q2_bleu4_at_least_13_5": large["bleu4_mean"] >= 13.5,
        "Q3_nll_std_at_most_0_05": large["nll_std"] <= 0.05,
        "E1_peak_vram_below_10gib": large["peak_vram_max"] < 10 * 2**30,
        "E2_all_gradients_finite": all(row["finite_gradients"] for row in results),
        "E3_parameter_counts_exact": all(
            row["parameters"] == EXPECTED_PARAMETERS[row["arm"]]
            for row in results
        ),
    }
    capacity = all(gates[f"C{i}_{suffix}"] for i, suffix in [
        (1, "50m_nll_gain_at_least_0_08"),
        (2, "50m_bleu_gain_at_least_0_75"),
        (3, "50m_nll_std_at_most_0_08"),
        (4, "50m_within_0_02_of_28m_long"),
        (5, "force_algebraic_damage_at_least_0_10"),
        (6, "address_damage_at_least_0_10"),
        (7, "depth_growth"), (8, "closure_below_1e_5"),
        (9, "generation_nondegenerate"),
    ])
    product = all(gates[key] for key in (
        "Q1_nll_at_most_3_90", "Q2_bleu4_at_least_13_5",
        "Q3_nll_std_at_most_0_05",
    ))
    engineering = all(value for key, value in gates.items() if key.startswith("E"))
    stage_b_authorized = (
        gates["C1_50m_nll_gain_at_least_0_08"]
        and gates["C3_50m_nll_std_at_most_0_08"]
        and sum(gates[key] for key in (
            "C5_force_algebraic_damage_at_least_0_10",
            "C6_address_damage_at_least_0_10", "C7_depth_growth",
        )) >= 2
    )
    if smoke:
        status = "smoke_only"
    elif capacity and product and engineering:
        status = "capacity_supported_stone1_complete"
    elif capacity and engineering:
        status = "capacity_supported_stone1_incomplete"
    else:
        status = "capacity_not_supported_stone1_incomplete"
    return gates, status, stage_b_authorized


def render_report(summary: dict, results: Sequence[dict]) -> str:
    lines = [
        "# STONE-1 C03 Capacity and Rate-Distortion Report", "",
        "## Experiment Card", "", "| Field | Value |", "|---|---|",
        f"| Status | `{summary['status']}` |",
        f"| Stage B authorized | `{summary['stage_b_authorized']}` |",
        f"| Host / device | `{summary['host']}` / `{summary['device_name']}` |",
        f"| Runtime | {summary['seconds'] / 3600:.2f} h |", "",
        "## Per-run Results", "",
        "| Arm | Seed | Steps | Params | Test NLL | Bits/token | BLEU-4 | Train NLL | Time | VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        test = row["test"]
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['fixed_steps']:,} | "
            f"{row['parameters']:,} | {test['nll']:.4f} | "
            f"{test['nll'] / math.log(2):.4f} | {test['token_bleu4']:.3f} | "
            f"{row['final_train_nll_window']:.4f} | {row['seconds']/60:.1f} min | "
            f"{row['peak_vram_bytes']/2**30:.2f} GiB |"
        )
    lines.extend([
        "", "## Aggregate", "", "```json",
        json.dumps(summary["aggregate"], indent=2, ensure_ascii=False), "```",
        "", "## Gates", "", "```json",
        json.dumps(summary["gates"], indent=2, ensure_ascii=False), "```", "",
        "## Boundary", "", summary["boundary"], "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--baseline-summary", default=(
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_canonical_codec/summary.json"
    ))
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--arms", nargs="+", choices=ARM_CONFIGS, default=list(ARM_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[71901, 71902, 71903])
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--baseline-max-scan", type=int, default=300_000)
    parser.add_argument("--pool-max-scan", type=int, default=3_000_000)
    parser.add_argument("--source-rows", type=int, default=14_170_275)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71900)
    parser.add_argument("--pool-seed", type=int, default=72003)
    parser.add_argument("--model-seed", type=int, default=71901)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--cli-max-new-tokens", type=int, default=64)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if set(args.arms) != set(ARM_CONFIGS):
        raise ValueError("C03 requires both registered Stage A arms")
    if args.smoke:
        args.train_samples = min(args.train_samples, 4096)
        args.valid_samples = min(args.valid_samples, 128)
        args.test_samples = min(args.test_samples, 128)
        args.baseline_max_scan = min(args.baseline_max_scan, 100_000)
        args.pool_max_scan = min(args.pool_max_scan, 200_000)
        args.seeds = args.seeds[:1]
        args.eval_interval = min(args.eval_interval, 100)
        args.latency_repeats = min(args.latency_repeats, 5)

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output / "checkpoints"
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    config = vars(args).copy()
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["arm_configs"] = ARM_CONFIGS
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    command = shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]])
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8",
    )

    started = time.time()
    baseline = load_baseline(args.baseline_summary)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    platform_checks = verify_frozen_platform(manifest, baseline, args.smoke)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)
    tokenizer_sha256 = manifest["tokenizer"]["sha256"]

    results: List[dict] = []
    best_large_nll = math.inf
    best_large_state = None
    best_large_seed = None
    best_large_args = None
    for seed in args.seeds:
        for arm in args.arms:
            run_args = copy.copy(args)
            run_args.dim = ARM_CONFIGS[arm]["dim"]
            run_args.hidden = ARM_CONFIGS[arm]["hidden"]
            run_args.fixed_steps = 500 if args.smoke else ARM_CONFIGS[arm]["steps"]
            run_dir = output / "arms" / f"{arm}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result, state = c02.train_arm(
                "canonical_learned", seed, rows[: args.train_samples],
                valid_loader, test_loader, run_args, vocab, pad, bos, eos, sp,
                run_dir, checkpoint_dir / arm, tokenizer_sha256,
            )
            result.update({
                "arm": arm, "fixed_steps": run_args.fixed_steps,
                "dim": run_args.dim, "hidden": run_args.hidden,
                "final_train_nll_window": final_training_window(run_dir / "trace.jsonl"),
            })
            expected = parameter_count(vocab, run_args.dim, run_args.hidden)
            if expected != EXPECTED_PARAMETERS[arm] or result["parameters"] != expected:
                raise ValueError(
                    f"parameter mismatch for {arm}: formula={expected}, "
                    f"model={result['parameters']}, registered={EXPECTED_PARAMETERS[arm]}"
                )
            results.append(result)
            if arm == "balanced_50m_equal" and result["best_valid_nll"] < best_large_nll:
                best_large_nll = result["best_valid_nll"]
                best_large_state = state
                best_large_seed = seed
                best_large_args = copy.copy(run_args)
            (output / "runs.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            del state
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if best_large_state is None:
        raise RuntimeError("no 50M state was retained")
    intervention = c02.audit_learned(
        best_large_state, best_large_seed, test_loader, best_large_args,
        vocab, pad, bos, eos, sp,
    )
    (output / "interventions.json").write_text(
        json.dumps(intervention, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    aggregate_rows = aggregate(results)
    gates, status, stage_b = decide(
        baseline, aggregate_rows, intervention, results, args.smoke,
    )
    summary = {
        "experiment_id": (
            "s3_stone1_capacity_rate_distortion_smoke" if args.smoke
            else "s3_stone1_capacity_rate_distortion"
        ),
        "claim": "S3-STONE1-CAPACITY-RATE-DISTORTION-C03",
        "predict": "P-S3-STONE1-CAPACITY-RATE-DISTORTION-03",
        "milestone": "STONE-1", "status": status,
        "stage_b_authorized": stage_b,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "git_commit": args.code_commit or c01.git_revision(),
        "seconds": time.time() - started,
        "config": config, "dataset": manifest,
        "platform_checks": platform_checks,
        "frozen_c02_baseline": baseline["aggregate"]["canonical_learned"],
        "aggregate": aggregate_rows, "intervention": intervention,
        "gates": gates,
        "capacity_deltas": {
            "nll_gain_50m_over_c02_28m": (
                baseline["aggregate"]["canonical_learned"]["nll_mean"]
                - aggregate_rows["balanced_50m_equal"]["nll_mean"]
            ),
            "bleu_gain_50m_over_c02_28m": (
                aggregate_rows["balanced_50m_equal"]["bleu4_mean"]
                - baseline["aggregate"]["canonical_learned"]["bleu4_mean"]
            ),
            "nll_50m_minus_28m_long": (
                aggregate_rows["balanced_50m_equal"]["nll_mean"]
                - aggregate_rows["base_28m_long"]["nll_mean"]
            ),
        },
        "boundary": (
            "C03 tests whole-system capacity under one frozen WMT and optimizer "
            "contract. It does not isolate codec-only capacity or establish "
            "general scaling laws, dialogue, or world knowledge."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        render_report(summary, results), encoding="utf-8",
    )
    print(json.dumps({
        "status": status, "stage_b_authorized": stage_b,
        "seconds": summary["seconds"], "aggregate": aggregate_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
