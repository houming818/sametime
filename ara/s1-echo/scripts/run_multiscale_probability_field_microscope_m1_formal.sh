#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
out="ara/s1-echo/evidence/s1_multiscale_probability_field_microscope_m1/formal_200k_seed20261011"
mkdir -p "$out"

python3 ara/s1-echo/tests/test_multiscale_probability_field_microscope_m1.py \
  >"$out/unit_test.log" 2>&1

python3 ara/s1-echo/src/s1_multiscale_probability_field_microscope_m1.py \
  --output "$out" \
  --max-lines 200000 \
  --target-vocab 8192 \
  --context-vocab 4096 \
  --controls 3 \
  --seed 20261011 \
  --device cuda \
  >"$out/stdout.log" 2>&1

python3 ara/s1-echo/src/s1_multiscale_posterior_pyramid_m1b.py \
  --output "$out" \
  --max-lines 200000 \
  --target-vocab 8192 \
  --context-vocab 4096 \
  --max-events 50000 \
  --controls 3 \
  --batch-size 128 \
  --spectrum-events 256 \
  --seed 20261011 \
  --device cuda \
  >"$out/posterior_stdout.log" 2>&1

(cd "$out" && sha256sum summary.json posterior_summary.json stdout.log posterior_stdout.log unit_test.log >SHA256SUMS)
cat "$out/summary.json"
cat "$out/posterior_summary.json"
