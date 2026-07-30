#!/usr/bin/env bash
set -euo pipefail
ara/s3-generation/src/s2_treeheap_butterfly_wmt.py --mode smoke --evidence-dir ara/s3-generation/evidence/s2_treeheap_butterfly_wmt_smoke --device cuda --generate-examples
