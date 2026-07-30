#!/usr/bin/env bash
set -euo pipefail
ara/s3-generation/src/s2_treeheap_butterfly_wmt.py --mode formal --evidence-dir ara/s3-generation/evidence/s2_treeheap_butterfly_wmt_formal --device cuda --generate-examples --save-checkpoint
