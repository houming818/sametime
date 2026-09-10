#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_treeheap_orthogonal_annealing_microscope_f10/smoke_seed12101"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_orthogonal_annealing_microscope_f10.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --data ara/s3-generation/data/cross_language_echo_f08.json
  --evidence-dir "$evidence"
  --depth 7
  --steps 16
  --seed 12101
  --device cuda
)

printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_orthogonal_annealing_microscope_f10.py \
  ara/s3-generation/src/s3_recursive_depth_pressure_protocol_training.py \
  ara/s3-generation/src/s3_filter_guided_theta_calibration_f04.py \
  ara/s3-generation/src/s3_treeheap_ancestor_jump_microscope_f09.py \
  ara/s3-generation/logic/treeheap_orthogonal_annealing_microscope_f10.zh.md \
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
