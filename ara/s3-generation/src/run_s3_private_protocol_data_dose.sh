#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_private_protocol_data_dose_full"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_private_protocol_data_dose.py \
  --evidence-dir "$evidence" \
  --models h1 flat transformer \
  --doses 30000 100000 300000 1000000 \
  --code-commit 2c37057 \
  --fixed-steps 15625 \
  --eval-interval 500 \
  > >(tee "$evidence/stdout.log") \
  2> >(tee "$evidence/stderr.log" >&2)
