#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_capacity_rate_distortion.py --evidence-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_capacity_rate_distortion_smoke_20260722a --checkpoint-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_capacity_rate_distortion_smoke_20260722a/checkpoints --code-commit d18ec85 --smoke
