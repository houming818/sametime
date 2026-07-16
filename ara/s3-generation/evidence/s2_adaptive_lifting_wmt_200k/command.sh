#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_adaptive_lifting_wmt.py \
  --data /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv \
  --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model \
  --evidence-dir ara/s3-generation/evidence/s2_adaptive_lifting_wmt_200k \
  --stage scale \
  --train-samples 200000 \
  --valid-samples 5000 \
  --test-samples 5000 \
  --max-scan 2000000 \
  --epochs 5 \
  --dim 256 \
  --hidden 256 \
  --batch-size 64 \
  --num-workers 2 \
  --candidate-variant learned_update \
  --audit-variant learned_update \
  --variants flat_seq old_recursive learned_update
