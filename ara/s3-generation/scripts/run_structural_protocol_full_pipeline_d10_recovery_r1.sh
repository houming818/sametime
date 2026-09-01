#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

mode="${1:-canary}"
claim_root="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1"
source_root="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/full_seed11001/task"
run_root="$claim_root/${mode}_seed11001"
task_root="$run_root/task"
runner="ara/s3-generation/src/s3_structural_protocol_full_pipeline_d10.py"
auditor="ara/s3-generation/src/s3_d10_recovery_checkpoint_audit.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/full_seed11001/pretrain/checkpoint_best.pt"
text_data="/home/nio/datasets/nio/releases/NioText-ZH-Integrity-2985K-v1/data.jsonl"
parallel_data="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv"

test -f "$runner"
test -f "$auditor"
test -f "$source_checkpoint"
test -f "$warm_start"
test -f "$source_root/checkpoint_latest.pt"
test -f "$source_root/checkpoint_best.pt"
test -f "$text_data"
test -f "$parallel_data"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

mkdir -p "$task_root"
if [[ ! -f "$task_root/checkpoint_latest.pt" ]]; then
  cp --reflink=auto "$source_root/checkpoint_latest.pt" "$task_root/checkpoint_latest.pt"
  cp --reflink=auto "$source_root/checkpoint_best.pt" "$task_root/checkpoint_best.pt"
fi

python3 "$auditor" \
  --checkpoint "$task_root/checkpoint_latest.pt" \
  --expected-step 275000 --expected-cursor 4400986 \
  --output "$run_root/checkpoint_before.json"

nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$run_root/gpu_before.csv"

if [[ "$mode" == "canary" ]]; then
  max_steps=277000
  wake_every=500
else
  echo "formal segmented continuation is submitted only after canary audit" >&2
  exit 2
fi

python3 "$runner" \
  --stage task --mode full --source-checkpoint "$source_checkpoint" \
  --warm-start "$warm_start" --evidence-dir "$run_root" \
  --text-data "$text_data" --parallel-data "$parallel_data" \
  --seed 11001 --ownership-seed 11002 --eval-rows 1000 \
  --wake-every "$wake_every" --log-every 100 --min-steps 100000 \
  --max-steps "$max_steps" --resume --device cuda \
  2>&1 | tee "$run_root/canary.log"

python3 "$auditor" \
  --checkpoint "$task_root/checkpoint_latest.pt" \
  --expected-step "$max_steps" \
  --output "$run_root/checkpoint_after.json"

nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$run_root/gpu_after.csv"

python3 - "$run_root" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "checkpoint_before.json").read_text(encoding="utf-8"))
after = json.loads((root / "checkpoint_after.json").read_text(encoding="utf-8"))
wake = json.loads((root / "task" / "wake_latest.json").read_text(encoding="utf-8"))
summary = json.loads((root / "task" / "summary.json").read_text(encoding="utf-8"))
initial_nll = float(summary["initial_valid"]["mean_nll"])
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
    "gates": gates,
    "passed": all(gates.values()),
}
(root / "recovery_summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False))
if not result["passed"]:
    raise SystemExit(3)
PY

sendme -s "D10 recovery canary finished" -f "$run_root/recovery_summary.json" || true
