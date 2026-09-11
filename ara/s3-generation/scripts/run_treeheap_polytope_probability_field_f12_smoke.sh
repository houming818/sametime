#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_treeheap_polytope_probability_field_f12/smoke_seed12301"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_polytope_probability_field_f12.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --evidence-dir "$evidence"
  --device cuda
  --seed 12301
  --groups 8
  --eval-rows 16
  --batch-size 8
  --step-norms 0.5 1.0
)

printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_polytope_probability_field_f12.py \
  ara/s3-generation/logic/treeheap_polytope_probability_field_f12.zh.md \
  >"$evidence/inputs.sha256"

nvidia-smi \
  --query-gpu=timestamp,power.limit,power.draw,temperature.gpu,memory.used,utilization.gpu \
  --format=csv -l 5 >"$evidence/gpu_samples.csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

"${command[@]}" 2>&1 | tee "$evidence/run.log"
