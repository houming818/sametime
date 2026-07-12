#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_conditional_denoising_seq2seq.py \
  --root /home/nio/datasets/pretrain \
  --spm-model ara/s3-generation/evidence/s3_p0_world_observation_tokenizer/p0_zh_16000.model \
  --evidence-dir ara/s3-generation/evidence/s3_conditional_denoising_smoke \
  --model all \
  --length 64 \
  --target-mode full \
  --target-length 64 \
  --mask-rate 0.30 \
  --max-span 3 \
  --batch 32 \
  --steps 1000 \
  --valid-every 250 \
  --valid-batches 16 \
  --test-batches 32 \
  --dim 192 \
  --hidden 192 \
  --lr 0.002 \
  --device cuda
