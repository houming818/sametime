#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_multiresolution_internal_echo.py \
  --block-dir /home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64 \
  --evidence-dir ara/s1-echo/evidence/s1_multiresolution_internal_echo/main \
  --context 16 \
  --dim 128 \
  --batch 128 \
  --epochs 5 \
  --max-train-blocks 100000 \
  --max-valid-blocks 8192 \
  --device cuda
