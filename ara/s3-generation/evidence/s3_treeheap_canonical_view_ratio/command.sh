#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

python3 ara/s3-generation/src/s3_treeheap_canonical_view_ratio.py \
  --checkpoint ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt \
  --ratios 0.0,0.2,0.4,0.6 \
  --seeds 9101 \
  --train-lines 300000 \
  --block-lines 50000 \
  --eval-pairs 1000 \
  --diagnostic-rows 128 \
  --diagnostic-batch 4 \
  --lr 0.0002 \
  --reuse-optimizer \
  --notify \
  2>&1 | tee ara/s3-generation/evidence/s3_treeheap_canonical_view_ratio/stdout.log
