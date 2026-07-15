#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_wmt_treeheap_seq2seq.py \
  --model all \
  --device cuda \
  --evidence-dir ara/s3-generation/evidence/s3_wmt_treeheap_seq2seq \
  --seed 17 \
  --train-samples 30000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --max-scan 120000 \
  --min-len 3 \
  --max-len 24 \
  --dim 256 \
  --hidden 256 \
  --batch-size 64 \
  --epochs 10 \
  --lr 0.002 \
  --num-workers 2
