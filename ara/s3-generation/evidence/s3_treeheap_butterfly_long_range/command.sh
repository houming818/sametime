#!/usr/bin/env bash
set -euo pipefail
ara/s3-generation/src/s3_treeheap_butterfly_long_range.py --device cuda --steps 1200 --batch 256 --eval-batches 16 --output ara/s3-generation/evidence/s3_treeheap_butterfly_long_range
