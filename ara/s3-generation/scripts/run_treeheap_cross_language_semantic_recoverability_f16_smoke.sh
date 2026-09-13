#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_cross_language_semantic_recoverability_f16.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --corpus /home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv
  --evidence-dir "$evidence"
  --per-class 40
  --candidate-cap 256
  --batch-size 16
  --shuffle-repeats 5
  --seed 12701
  --device cuda
)
printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_cross_language_semantic_recoverability_f16.py \
  ara/s3-generation/logic/treeheap_cross_language_semantic_recoverability_f16.zh.md \
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
