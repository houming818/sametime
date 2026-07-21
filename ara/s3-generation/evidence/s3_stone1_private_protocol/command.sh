#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_private_protocol.py --evidence-dir ara/s3-generation/evidence/s3_stone1_private_protocol --checkpoint-dir ara/s3-generation/evidence/s3_stone1_private_protocol/checkpoints --train-samples 1000000 --valid-samples 2000 --test-samples 2000 --fixed-steps 15625 --eval-interval 500 --seeds 71901 71902 71903 --variants identity learned_structural frozen_random --code-commit 4c3d275
