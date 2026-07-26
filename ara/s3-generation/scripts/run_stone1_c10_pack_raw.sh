#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone1_c10_raw_pack"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_stone1_c10_pack_raw.py \
  --split train \
  --resume \
  2>&1 | tee -a "$evidence/stdout-train.log"

python3 ara/s3-generation/src/s3_stone1_c10_pack_raw.py \
  --split valid \
  --resume \
  2>&1 | tee -a "$evidence/stdout-valid.log"

printf '{"completed":"%s","exit_code":0}\n' "$(date --iso-8601=seconds)" \
  > "$evidence/runner_status.json"
