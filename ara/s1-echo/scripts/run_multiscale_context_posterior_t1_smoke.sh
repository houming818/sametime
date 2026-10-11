#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
out="ara/s1-echo/evidence/s1_multiscale_context_posterior_t1/smoke_20k_seed20261011"
mkdir -p "$out"

python3 ara/s1-echo/src/s1_multiscale_context_posterior_t1.py \
  --output "$out" \
  --max-lines 20000 \
  --target-vocab 2048 \
  --context-vocab 1024 \
  --max-events 12000 \
  --random-replicates 3 \
  --bootstrap-samples 500 \
  --seed 20261011 \
  --device cuda \
  2>&1 | tee "$out/stdout.log"

sha256sum "$out/summary.json" "$out/event_contract.json" "$out/stdout.log" > "$out/SHA256SUMS"
