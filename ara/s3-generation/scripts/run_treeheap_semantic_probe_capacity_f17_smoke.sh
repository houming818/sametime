#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_treeheap_semantic_probe_capacity_f17/smoke_seed12801"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_semantic_probe_capacity_f17.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --cases ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/cases.json
  --f16-summary ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/summary.json
  --evidence-dir "$evidence"
  --ridge 0.001
  --batch-size 16
  --shuffle-repeats 5
  --seed 12801
  --device cuda
)
printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_semantic_probe_capacity_f17.py \
  ara/s3-generation/logic/treeheap_semantic_probe_capacity_f17.zh.md \
  ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/cases.json \
  ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/summary.json \
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
