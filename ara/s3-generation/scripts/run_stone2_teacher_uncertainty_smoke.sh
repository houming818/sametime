#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
evidence="ara/s3-generation/evidence/s3_stone2_teacher_uncertainty_smoke"
mkdir -p "$evidence"
python3 ara/s3-generation/src/s3_stone2_teacher_uncertainty.py \
  --smoke \
  --evidence-dir "$evidence" \
  --code-commit "${CODE_COMMIT:-unknown}" \
  2>&1 | tee "$evidence/stdout.log"
