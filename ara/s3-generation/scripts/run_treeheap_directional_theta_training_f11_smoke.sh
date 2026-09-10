#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_treeheap_directional_theta_training_f11/smoke_seed12201"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_directional_theta_training_f11.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --specimens ara/s3-generation/data/filter_guided_theta_specimens_f04.json
  --evidence-dir "$evidence"
  --device cuda
  --seed 12201
  --rank 16
  --steps 90
  --lr 0.002
  --behavior-steps 24
  --generation-steps 64
  --eval-rows 16
  --eval-batch 8
)

printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_directional_theta_training_f11.py \
  ara/s3-generation/src/s3_filter_guided_theta_calibration_f04.py \
  ara/s3-generation/logic/treeheap_directional_theta_training_f11.zh.md \
  ara/s3-generation/data/filter_guided_theta_specimens_f04.json \
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
