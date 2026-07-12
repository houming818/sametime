#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_wmt_learned_fold_seq2seq.py \
  --data /mnt/nas/datasets/wmt17/train.zh-en \
  --spm-model /mnt/nas/datasets/wmt17/sp_bpe.model \
  --evidence-dir ara/s3-generation/evidence/s3_wmt_learned_fold_smoke \
  --model all \
  --train-samples 5000 \
  --valid-samples 500 \
  --test-samples 500 \
  --max-scan 50000 \
  --min-len 3 \
  --max-len 16 \
  --dim 192 \
  --hidden 192 \
  --batch-size 32 \
  --epochs 5 \
  --lr 0.002 \
  --device cuda \
  --num-workers 0
