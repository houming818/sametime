#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_adaptive_lifting_wmt.py \
  --data /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv \
  --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model \
  --evidence-dir ara/s3-generation/evidence/s2_adaptive_lifting_wmt_ablation \
  --stage ablation \
  --train-samples 30000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --max-scan 300000 \
  --epochs 5 \
  --dim 256 \
  --hidden 256 \
  --batch-size 64 \
  --num-workers 2
