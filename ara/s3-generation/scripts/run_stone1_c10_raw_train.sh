#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone1_c10_raw_full_train"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_stone1_c10_raw_train.py \
  --batch 96 \
  --resume \
  2>&1 | tee -a "$evidence/stdout.log"

printf '{"completed":"%s","exit_code":0}\n' "$(date --iso-8601=seconds)" \
  > "$evidence/runner_status.json"

if command -v sendme >/dev/null 2>&1; then
  sendme -s "STONE-1 C10 raw full pass completed" \
    "io task finished; see $evidence/summary.json"
fi
