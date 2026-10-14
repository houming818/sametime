#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

OUT="ara/s3-generation/evidence/s3_multiresolution_language_curriculum_a01/smoke_r2_seed20261013"
DATA="ara/s3-generation/evidence/s3_multiresolution_language_curriculum_t0/formal_r3_512_seed20261012/accepted_curriculum.jsonl"
CHECKPOINT="ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_latest.pt"
mkdir -p "$OUT"

nvidia-smi --query-gpu=timestamp,name,power.limit,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader >"$OUT/gpu_before.csv"

python3 ara/s3-generation/src/s3_multiresolution_language_curriculum_a01.py \
  --mode smoke \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --seed 20261013 \
  >"$OUT/stdout.log" 2>"$OUT/stderr.log"

nvidia-smi --query-gpu=timestamp,name,power.limit,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader >"$OUT/gpu_after.csv"

find "$OUT" -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | sort | while read -r file; do
  sha256sum "$OUT/$file"
done >"$OUT/SHA256SUMS"
sha256sum -c "$OUT/SHA256SUMS"
