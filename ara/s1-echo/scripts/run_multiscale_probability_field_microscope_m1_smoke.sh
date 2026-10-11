#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
out="ara/s1-echo/evidence/s1_multiscale_probability_field_microscope_m1/smoke_20k_seed20261011"
mkdir -p "$out"

python3 ara/s1-echo/tests/test_multiscale_probability_field_microscope_m1.py \
  >"$out/unit_test.log" 2>&1

python3 ara/s1-echo/src/s1_multiscale_probability_field_microscope_m1.py \
  --output "$out" \
  --max-lines 20000 \
  --target-vocab 2048 \
  --context-vocab 1024 \
  --controls 3 \
  --seed 20261011 \
  --device cuda \
  >"$out/stdout.log" 2>&1

python3 ara/s1-echo/src/s1_multiscale_posterior_pyramid_m1b.py \
  --output "$out" \
  --max-lines 20000 \
  --target-vocab 2048 \
  --context-vocab 1024 \
  --max-events 5000 \
  --controls 3 \
  --seed 20261011 \
  --device cuda \
  >"$out/posterior_stdout.log" 2>&1

(cd "$out" && sha256sum summary.json posterior_summary.json stdout.log posterior_stdout.log unit_test.log >SHA256SUMS)
cat "$out/summary.json"
cat "$out/posterior_summary.json"

