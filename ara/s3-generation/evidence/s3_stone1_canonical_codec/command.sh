#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_canonical_codec.py --evidence-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_canonical_codec --checkpoint-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_canonical_codec/checkpoints --train-samples 1000000 --valid-samples 2000 --test-samples 2000 --fixed-steps 15625 --eval-interval 500 --seeds 71901 71902 71903 --variants canonical_algebraic canonical_learned canonical_frozen --code-commit decc78e
