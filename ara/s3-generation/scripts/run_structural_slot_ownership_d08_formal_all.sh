#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

for seed in 10811 10812 10813; do
  bash ara/s3-generation/scripts/run_structural_slot_ownership_d08_formal_seed.sh "$seed"
done

python3 ara/s3-generation/src/s3_structural_slot_ownership_d08_aggregate.py \
  --evidence-dir ara/s3-generation/evidence/s3_structural_slot_ownership_d08
