#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_treeheap_functional_equivalence.py \
  --checkpoint-dir ara/s3-generation/evidence/s3_wmt_frontier_smoke \
  --output ara/s3-generation/evidence/s2_treeheap_functional_equivalence/summary.json \
  --groups 8 \
  --batch-size 16 \
  --device cuda
