#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_treeheap_emergent_mask_translation.py \
  --evidence-dir ara/s3-generation/evidence/s2_treeheap_emergent_mask_translation \
  --train-samples 5000 \
  --valid-samples 500 \
  --test-samples 500 \
  --max-scan 100000 \
  --min-len 9 \
  --max-len 24 \
  --dim 192 \
  --hidden 192 \
  --rank 32 \
  --heads 4 \
  --batch-size 24 \
  --epochs 5 \
  --mask-probability 0.5 \
  --mask-cuts 2 \
  --max-mask-depth 3 \
  --precision fp32 \
  --device cuda
