#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
evidence="ara/s3-generation/evidence/s3_stone2_dialogue_100k"
mkdir -p "$evidence"
python3 ara/s3-generation/src/s3_stone2_dialogue.py \
  --evidence-dir "$evidence" \
  2>&1 | tee "$evidence/stdout.log"
