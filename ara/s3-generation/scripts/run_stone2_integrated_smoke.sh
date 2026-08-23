#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101"
rm -rf "$evidence"

notify_exit() {
  status=$?
  if [[ $status -eq 0 ]]; then
    sendme -s "STONE-2 integrated smoke PASSED" -f "$evidence/integrated_audit.json" || true
  else
    printf 'STONE-2 integrated smoke failed on %s with exit=%s\nEvidence: %s\n' \
      "$(hostname)" "$status" "$evidence" | sendme -s "STONE-2 integrated smoke FAILED" || true
  fi
}
trap notify_exit EXIT

python3 ara/s3-generation/src/s3_stone2_integrated_pipeline.py \
    --mode smoke \
    --seed 16101 \
    --evidence-dir "$evidence" \
    --raw-root /home/nio/datasets/pretrain \
    --wmt-data /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv \
    --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model \
    --device cuda \
    --heap-width 256 \
    --pretrain-steps 120 \
    --pretrain-train-rows 1536 \
    --pretrain-valid-rows 256 \
    --task-steps 160 \
    --task-train-rows 4096 \
    --task-eval-rows 256 \
    --posterior-contexts 64 \
    --log-every 40

python3 ara/s3-generation/src/s3_stone2_integrated_audit.py \
  --checkpoint "$evidence/task/PT/checkpoint_best.pt" \
  --output "$evidence/integrated_audit.json" \
  --device cuda \
  --eval-rows 256
