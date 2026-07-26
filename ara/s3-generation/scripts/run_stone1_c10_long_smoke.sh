#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone1_c10_long_smoke"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_stone1_c10_long_smoke.py \
  --evidence-dir "$evidence" \
  --source-width 128 \
  --target-width 128 \
  --heap-width 256 \
  --batch 4 \
  --steps 300 \
  2>&1 | tee "$evidence/stdout.log"

printf '{"completed":"%s","exit_code":0}\n' "$(date --iso-8601=seconds)" \
  > "$evidence/runner_status.json"
