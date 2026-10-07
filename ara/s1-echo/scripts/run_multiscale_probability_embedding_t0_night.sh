#!/usr/bin/env bash
# Owner: Nio / Trinity
# Author: GPT-5.6 Sol
# Created: 2026-10-07
# Updated: 2026-10-07
# Responsibility: run T0 smoke, conditionally release the full budget, then
# execute a supplemental local-teacher blind audit.

set -euo pipefail

cd /home/nio/log/holds/SameTime

ROOT="ara/s1-echo/evidence/s1_multiscale_probability_embedding_t0"
SMOKE="$ROOT/smoke_20k_seed20261007"
FORMAL="$ROOT/formal_200k_seed20261007"
TEACHER="$ROOT/teacher_qwen3_8b_seed20261007"
mkdir -p "$SMOKE" "$FORMAL" "$TEACHER"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total --format=csv,noheader >"$ROOT/gpu_before.txt"

python3 ara/s1-echo/src/s1_multiscale_probability_embedding_t0.py \
  --output "$SMOKE" \
  --max-lines 20000 \
  --target-vocab 2048 \
  --context-vocab 1024 \
  --sgns-steps 500 \
  --teacher-items 20 \
  --seed 20261007 \
  --device cuda 2>&1 | tee "$SMOKE/stdout.log"

python3 ara/s1-echo/src/s1_multiscale_probability_embedding_t0.py \
  --output "$FORMAL" \
  --max-lines 200000 \
  --target-vocab 8192 \
  --context-vocab 4096 \
  --sgns-steps 5000 \
  --sgns-batch 2048 \
  --teacher-items 80 \
  --seed 20261007 \
  --device cuda 2>&1 | tee "$FORMAL/stdout.log"

python3 ara/s1-echo/src/s1_embedding_teacher_blind_eval_t0.py \
  --model /home/nio/models/Qwen3-8B \
  --embedding-evidence "$FORMAL" \
  --output "$TEACHER" \
  --calibration-items 40 \
  --blind-items 80 \
  --seed 20261007 2>&1 | tee "$TEACHER/stdout.log"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total --format=csv,noheader >"$ROOT/gpu_after.txt"
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$ROOT/SHA256SUMS"

