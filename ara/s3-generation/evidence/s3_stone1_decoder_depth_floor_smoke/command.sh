#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_decoder_depth_floor.py --evidence-dir ara/s3-generation/evidence/s3_stone1_decoder_depth_floor_smoke --code-commit d926775 --smoke
