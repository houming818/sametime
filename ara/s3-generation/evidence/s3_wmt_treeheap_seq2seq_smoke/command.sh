#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_wmt_treeheap_seq2seq.py \
  --model treeheap \
  --device cuda \
  --evidence-dir ara/s3-generation/evidence/s3_wmt_treeheap_seq2seq_smoke \
  --train-samples 1024 \
  --valid-samples 128 \
  --test-samples 128 \
  --max-scan 10000 \
  --max-len 16 \
  --dim 96 \
  --hidden 96 \
  --batch-size 32 \
  --epochs 2 \
  --num-workers 0
