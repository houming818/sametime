#!/usr/bin/env bash
set -euo pipefail
python3 /home/nio/log/holds/SameTime/ara/s3-generation/src/s3_private_protocol_data_dose.py --evidence-dir ara/s3-generation/evidence/s3_private_protocol_data_dose_full --models h1 flat transformer --doses 30000 100000 300000 1000000 --code-commit 7bbb89c --fixed-steps 15625 --eval-interval 500
