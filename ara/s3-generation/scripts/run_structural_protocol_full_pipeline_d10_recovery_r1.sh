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
canary_reporter="ara/s3-generation/src/s3_d10_recovery_canary_report.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/full_seed11001/pretrain/checkpoint_best.pt"
text_data="/home/nio/datasets/nio/releases/NioText-ZH-Integrity-2985K-v1/data.jsonl"
parallel_data="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv"

test -f "$runner"
test -f "$auditor"
test -f "$canary_reporter"
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

nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$run_root/gpu_before.csv"

if [[ "$mode" == "canary" ]]; then
  cp "$source_root/wake_latest.json" "$run_root/source_wake_step275000.json"
  python3 "$auditor" \
    --checkpoint "$task_root/checkpoint_latest.pt" \
    --expected-step 275000 --expected-cursor 4400986 \
    --output "$run_root/checkpoint_before.json"
  max_steps=277000
  wake_every=500
else
  canary_root="$claim_root/canary_seed11001"
  python3 - "$canary_root/recovery_summary.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if not result.get("passed"):
    raise SystemExit("canary did not pass; formal recovery is blocked")
PY
  if [[ ! -f "$run_root/recovery_initialized.json" ]]; then
    cp --reflink=auto "$canary_root/task/checkpoint_latest.pt" "$task_root/checkpoint_latest.pt"
    cp --reflink=auto "$canary_root/task/checkpoint_best.pt" "$task_root/checkpoint_best.pt"
    python3 "$auditor" \
      --checkpoint "$task_root/checkpoint_latest.pt" \
      --expected-step 277000 --expected-cursor 4432986 \
      --output "$run_root/recovery_initialized.json"
  fi
fi

if [[ "$mode" == "canary" ]]; then
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

  python3 "$canary_reporter" --run-root "$run_root"

  sendme -s "D10 recovery canary finished" -f "$run_root/recovery_summary.json" || true
  exit 0
fi

mkdir -p "$run_root/segments"
while true; do
  read -r start_step start_cursor < <(python3 - "$task_root/checkpoint_latest.pt" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload["step"]), int(payload["cursor"]))
PY
  )
  if (( start_cursor >= 7304358 )); then
    break
  fi
  end_step=$((start_step + 25000))
  segment="${start_step}_${end_step}"
  nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader > "$run_root/segments/${segment}_gpu_before.csv"
  python3 "$runner" \
    --stage task --mode full --source-checkpoint "$source_checkpoint" \
    --warm-start "$warm_start" --evidence-dir "$run_root" \
    --text-data "$text_data" --parallel-data "$parallel_data" \
    --seed 11001 --ownership-seed 11002 --eval-rows 1000 \
    --wake-every 5000 --log-every 500 --min-steps 100000 \
    --max-steps "$end_step" --resume --device cuda \
    2>&1 | tee -a "$run_root/formal.log"
  read -r actual_step actual_cursor < <(python3 - "$task_root/checkpoint_latest.pt" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload["step"]), int(payload["cursor"]))
PY
  )
  if (( actual_step <= start_step || actual_cursor <= start_cursor )); then
    echo "segment made no forward progress" >&2
    exit 4
  fi
  python3 "$auditor" \
    --checkpoint "$task_root/checkpoint_latest.pt" \
    --expected-step "$actual_step" \
    --output "$run_root/segments/${segment}_checkpoint.json"
  cp "$task_root/wake_latest.json" "$run_root/segments/${segment}_wake.json"
  cp "$task_root/summary.json" "$run_root/segments/${segment}_summary.json"
  nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader > "$run_root/segments/${segment}_gpu_after.csv"
done

python3 - "$run_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "task" / "summary.json").read_text(encoding="utf-8"))
result = {
    "claim": "S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10-RECOVERY-R1",
    "mode": "formal_segmented_continuation",
    "cursor": summary["cursor"],
    "best_step": summary["best_step"],
    "best_valid": summary["best_valid"],
    "best_generation": summary["best_generation"],
    "gates": summary["gates"],
    "decision": summary["decision"],
    "complete": summary["cursor"] >= 7304358,
}
(root / "recovery_summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False))
if not result["complete"]:
    raise SystemExit(5)
PY

sendme -s "D10 segmented recovery finished" -f "$run_root/recovery_summary.json" || true
