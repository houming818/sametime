#!/usr/bin/env bash
set -euo pipefail

python3 ara/m0-treeheap-math/src/algebraic_operator_codec_probe.py \
  --out ara/m0-treeheap-math/evidence/algebraic_operator_codec_probe \
  --train-samples 8000 \
  --test-samples 2000 \
  --epochs 20 \
  --batch 128 \
  --hidden 192 \
  --lr 0.002 \
  --seed 53 \
  --device cuda
