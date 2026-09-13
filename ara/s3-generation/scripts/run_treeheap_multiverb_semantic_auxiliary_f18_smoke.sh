#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_treeheap_multiverb_semantic_auxiliary_f18/smoke_seed12901"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_multiverb_semantic_auxiliary_f18.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --corpus /home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv
  --f16-cases ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/cases.json
  --eval-wmt-data /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv
  --evidence-dir "$evidence"
  --candidate-cap 256
  --train-per-class 32
  --eval-per-class 8
  --head-steps 50
  --joint-steps 120
  --batch-size 16
  --eval-batch 8
  --wmt-eval-rows 64
  --head-lr 0.002
  --joint-head-lr 0.0005
  --model-lr 0.00002
  --aux-weight 0.20
  --seed 12901
  --device cuda
)
printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_multiverb_semantic_auxiliary_f18.py \
  ara/s3-generation/logic/treeheap_multiverb_semantic_auxiliary_f18.zh.md \
  ara/s3-generation/evidence/s3_treeheap_cross_language_semantic_recoverability_f16/smoke_seed12701/cases.json \
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
