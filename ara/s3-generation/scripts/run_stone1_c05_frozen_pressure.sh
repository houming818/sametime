#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone1_frozen_encoder_pressure_decoder"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_stone1_frozen_encoder_pressure_decoder.py \
  --evidence-dir "$evidence" \
  --code-commit "${CODE_COMMIT:-$(git rev-parse --short HEAD)}" \
  2>&1 | tee "$evidence/stdout.log"

printf '{"completed":"%s","exit_code":0}\n' "$(date --iso-8601=seconds)" \
  > "$evidence/runner_status.json"
