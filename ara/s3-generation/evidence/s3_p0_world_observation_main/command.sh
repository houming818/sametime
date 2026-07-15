#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_treeheap_p0_pretrain.py \
  --mode train \
  --root /home/nio/datasets/pretrain \
  --spm-model ara/s3-generation/evidence/s3_p0_world_observation_tokenizer/p0_zh_16000.model \
  --evidence-dir ara/s3-generation/evidence/s3_p0_world_observation_main \
  --model treeheap \
  --context 64 \
  --target 32 \
  --batch 64 \
  --steps 50000 \
  --valid-every 1000 \
  --valid-batches 64 \
  --dim 192 \
  --hidden 192 \
  --lr 0.002 \
  --device cuda
