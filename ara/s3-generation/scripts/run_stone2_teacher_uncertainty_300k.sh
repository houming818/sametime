#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
evidence="ara/s3-generation/evidence/s3_stone2_teacher_uncertainty_300k"
cache="/home/nio/datasets/distillation/opus_mt_en_zh_300k_top4_temp0p1.jsonl.gz"
mkdir -p "$evidence" "$(dirname "$cache")"
python3 ara/s3-generation/src/s3_stone2_teacher_uncertainty.py \
  --evidence-dir "$evidence" \
  --cache "$cache" \
  --code-commit "${CODE_COMMIT:-unknown}" \
  2>&1 | tee "$evidence/stdout.log"
