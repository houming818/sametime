#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_lifting_information_pump.py \
  --block-dir /home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64 \
  --evidence-dir ara/s1-echo/evidence/s1_lifting_information_pump/main \
  --context 16 \
  --dim 128 \
  --batch 256 \
  --epochs 3 \
  --max-train-blocks 100000 \
  --max-valid-blocks 8192 \
  --codec-blocks 128 \
  --device cuda
