#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python ara/s1-echo/src/s1_treeheap_vs_mlp_metric_objective.py \
  --out ara/s1-echo/evidence/s1_treeheap_vs_mlp_metric_objective \
  --seeds 8 --base-epochs 800 --metric-epochs 400 \
  --dim 32 --prefix-slots 6 --mlp-hidden 48 --lr 0.03 \
  --echo-weight 0.1 --contrastive-weight 0.5 --temperature 0.25 \
  --seed-start 81 --device cuda
