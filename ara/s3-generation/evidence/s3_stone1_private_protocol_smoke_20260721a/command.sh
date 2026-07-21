#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_private_protocol.py --evidence-dir ara/s3-generation/evidence/s3_stone1_private_protocol_smoke_20260721a --checkpoint-dir ara/s3-generation/evidence/s3_stone1_private_protocol_smoke_20260721a/checkpoints --code-commit 4c3d275 --smoke
