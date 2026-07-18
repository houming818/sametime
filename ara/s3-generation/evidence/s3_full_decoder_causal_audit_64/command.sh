#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
exec python3 -u ara/s3-generation/src/s3_full_decoder_causal_audit.py   --checkpoint ara/s3-generation/evidence/s3_full_repair_seq2seq/checkpoint_160000.pt   --evidence-dir ara/s3-generation/evidence/s3_full_decoder_causal_audit_64   --eval-batches 8 --eval-batch 8 --threads 6
