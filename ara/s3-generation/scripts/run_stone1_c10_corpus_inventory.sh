#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone1_c10_corpus_inventory"
mkdir -p "$evidence"

python3 ara/s3-generation/src/s3_stone1_c10_corpus_inventory.py \
  --evidence-dir "$evidence" \
  2>&1 | tee "$evidence/stdout.log"

printf '{"completed":"%s","exit_code":0}\n' "$(date --iso-8601=seconds)" \
  > "$evidence/runner_status.json"
