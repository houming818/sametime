#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

SMOKE_BLOCK_DIR=/home/nio/datasets/derived/s3_residual_treeheap_forest/smoke_blocks64
FULL_BLOCK_DIR=/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64
TOKENIZER=ara/s3-generation/evidence/s3_p0_world_observation_tokenizer/p0_zh_16000.model
EVIDENCE=ara/s3-generation/evidence/s3_residual_treeheap_forest
SCRIPT=ara/s3-generation/src/s3_residual_treeheap_forest_pretrain.py

mkdir -p "$SMOKE_BLOCK_DIR" "$FULL_BLOCK_DIR" "$EVIDENCE"
nvidia-smi --query-gpu=name,memory.total,memory.used,power.limit --format=csv,noheader | tee "$EVIDENCE/gpu_before.txt"

# A bounded prepare/train gate catches parser, CUDA, memory, and mechanism bugs.
python3 "$SCRIPT" prepare --split valid --block-dir "$SMOKE_BLOCK_DIR" --spm-model "$TOKENIZER" --max-blocks 16384
python3 "$SCRIPT" prepare --split train --block-dir "$SMOKE_BLOCK_DIR" --spm-model "$TOKENIZER" --max-blocks 65536
for SEED in 71301 71302 71303; do
  python3 "$SCRIPT" train --block-dir "$SMOKE_BLOCK_DIR" --evidence-dir "$EVIDENCE/smoke_seed_${SEED}" \
    --seed "$SEED" --max-train-blocks 65536 --max-valid-blocks 8192 --batch 128 --epochs 1 \
    --valid-every 250 --checkpoint-every 500
done

# Prepare complete source passes in their own immutable manifest directory.
python3 "$SCRIPT" prepare --split valid --block-dir "$FULL_BLOCK_DIR" --spm-model "$TOKENIZER" --max-blocks 65536
python3 "$SCRIPT" prepare --split train --block-dir "$FULL_BLOCK_DIR" --spm-model "$TOKENIZER"
python3 "$SCRIPT" train --block-dir "$FULL_BLOCK_DIR" --evidence-dir "$EVIDENCE/full" \
  --max-train-blocks 0 --max-valid-blocks 32768 --batch 128 --epochs 1 \
  --valid-every 5000 --checkpoint-every 10000

nvidia-smi --query-gpu=name,memory.total,memory.used,power.limit --format=csv,noheader | tee "$EVIDENCE/gpu_after.txt"
sudo mkdir -p /mnt/nas/ara/s3_residual_treeheap_forest/evidence
sudo cp -a "$EVIDENCE"/. /mnt/nas/ara/s3_residual_treeheap_forest/evidence/
sendme -s "ARA S3 residual TreeHeap forest finished" \
  "io completed S3-RESIDUAL-FOREST-C01. Evidence: $EVIDENCE/full/summary.json"
