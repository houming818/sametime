#!/usr/bin/env python3
"""Build the D10 recovery canary report from persisted evidence."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)

    before = json.loads((root / "checkpoint_before.json").read_text(encoding="utf-8"))
    after = json.loads((root / "checkpoint_after.json").read_text(encoding="utf-8"))
    source_wake = json.loads(
        (root / "source_wake_step275000.json").read_text(encoding="utf-8")
    )
    wake = json.loads((root / "task" / "wake_latest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "task" / "summary.json").read_text(encoding="utf-8"))
    initial_nll = float(source_wake["valid"]["mean_nll"])
    final_nll = float(wake["valid"]["mean_nll"])
    gates = {
        "checkpoint_before": bool(before["passed"]),
        "checkpoint_after": bool(after["passed"]),
        "advanced_2000_steps": after["step"] - before["step"] == 2000,
        "cursor_advanced": after["cursor"] > before["cursor"],
        "finite_nll": math.isfinite(final_nll),
        "nll_damage_at_most_0_30": final_nll - initial_nll <= 0.30,
        "structure": all(
            row["native"]["route"]["owner_leaf_coverage"] == 1.0
            and row["native"]["route"]["argmax_coverage"] >= 0.999
            for row in summary["best_causal"].values()
        ),
    }
    result = {
        "claim": "S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10-RECOVERY-R1",
        "mode": "canary",
        "start_step": before["step"],
        "end_step": after["step"],
        "start_cursor": before["cursor"],
        "end_cursor": after["cursor"],
        "initial_mean_nll": initial_nll,
        "final_mean_nll": final_nll,
        "nll_delta": final_nll - initial_nll,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (root / "recovery_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
