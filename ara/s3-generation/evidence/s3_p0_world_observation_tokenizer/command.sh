#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_treeheap_p0_pretrain.py \
  --mode tokenizer \
  --root /home/nio/datasets/pretrain \
  --evidence-dir ara/s3-generation/evidence/s3_p0_world_observation_tokenizer \
  --tokenizer-samples 50000 \
  --vocab 16000 \
  --seed 17
