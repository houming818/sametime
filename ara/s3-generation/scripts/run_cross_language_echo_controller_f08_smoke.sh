#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_cross_language_echo_controller_f08/smoke_r2_seed11901"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_cross_language_echo_controller_f08.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --theta-checkpoint ara/s3-generation/evidence/s3_filter_guided_theta_calibration_f04/smoke_r1_seed11501/filter-guided-theta.theta.pt
  --data ara/s3-generation/data/cross_language_echo_f08.json
  --evidence-dir "$evidence"
  --depth 7
  --steps 16
  --optimization-steps 80
  --learning-rate 0.05
  --max-u 0.2
  --rank 16
  --seed 11901
  --device cuda
)

printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_cross_language_echo_controller_f08.py \
  ara/s3-generation/src/s3_treeheap_structured_filter_bank_f07.py \
  ara/s3-generation/src/s3_treeheap_analytic_filter_interference_f06.py \
  ara/s3-generation/logic/cross_language_echo_controller_f08.zh.md \
  ara/s3-generation/data/cross_language_echo_f08.json \
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
