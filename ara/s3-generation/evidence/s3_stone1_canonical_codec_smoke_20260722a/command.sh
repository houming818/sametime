#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_stone1_canonical_codec.py --evidence-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_canonical_codec_smoke_20260722a --checkpoint-dir /home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_canonical_codec_smoke_20260722a/checkpoints --smoke --code-commit 22b5ef7
