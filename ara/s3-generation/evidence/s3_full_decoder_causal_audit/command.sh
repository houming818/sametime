#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
exec python3 -u ara/s3-generation/src/s3_full_decoder_causal_audit.py   --checkpoint ara/s3-generation/evidence/s3_full_repair_seq2seq/checkpoint_160000.pt   --evidence-dir ara/s3-generation/evidence/s3_full_decoder_causal_audit   --eval-batches 2 --eval-batch 4 --threads 6
