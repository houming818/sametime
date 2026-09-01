#!/usr/bin/env bash
set -euo pipefail

input_root="${1:-/home/nio/treeheap-observer/input/ara}"
output_dir="${2:-/home/nio/treeheap-observer/output/c01}"
interval="${OBSERVER_INTERVAL:-900}"
cycles="${OBSERVER_CYCLES:-192}"
tmp_dir="${OBSERVER_TMPDIR:-/home/nio/treeheap-observer/tmp}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "$output_dir" "$tmp_dir"
export TMPDIR="$tmp_dir"
exec nice -n 10 python3 ara/observer/src/treeheap_evidence_miner.py \
  --input-root "$input_root" \
  --output-dir "$output_dir" \
  --interval "$interval" \
  --cycles "$cycles"
