#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python ara/m0-treeheap-math/src/recursive_operator_locator_probe.py \
  --out ara/m0-treeheap-math/evidence/recursive_operator_locator_probe \
  --train-samples 12000 --test-samples 2000 --epochs 24 \
  --batch 128 --hidden 192 --lr 0.002 --seed 59 --device cuda
