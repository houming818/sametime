#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
out="ara/s1-echo/evidence/s1_multiscale_context_posterior_t1/formal_200k_seed20261011"
mkdir -p "$out"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$out/gpu_before.txt"

python3 ara/s1-echo/src/s1_multiscale_context_posterior_t1.py \
  --output "$out" \
  --max-lines 200000 \
  --target-vocab 8192 \
  --context-vocab 4096 \
  --max-events 50000 \
  --random-replicates 3 \
  --bootstrap-samples 1000 \
  --batch-size 128 \
  --seed 20261011 \
  --device cuda \
  2>&1 | tee "$out/stdout.log"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$out/gpu_after.txt"

sha256sum \
  "$out/summary.json" \
  "$out/event_contract.json" \
  "$out/stdout.log" \
  "$out/gpu_before.txt" \
  "$out/gpu_after.txt" \
  > "$out/SHA256SUMS"
