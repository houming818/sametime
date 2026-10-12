#!/usr/bin/env bash
# Owner: Nio / Trinity
# Author: GPT-5.6 Sol
# Created: 2026-10-12
# Responsibility: validate and generate a bounded teacher-synthetic multiresolution curriculum.

set -euo pipefail
cd /home/nio/log/holds/SameTime

ROOT="ara/s3-generation/evidence/s3_multiresolution_language_curriculum_t0"
SMOKE="$ROOT/smoke_24_seed20261012"
FORMAL="$ROOT/formal_512_seed20261012"
DATA="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv"
MODEL="/home/nio/models/Qwen3-8B"
mkdir -p "$SMOKE" "$FORMAL"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total --format=csv,noheader >"$ROOT/gpu_before.txt"
python3 ara/s3-generation/tests/test_multiresolution_language_curriculum_t0.py 2>&1 | tee "$ROOT/tests.log"

python3 ara/s3-generation/src/s3_multiresolution_language_curriculum_t0.py \
  --data "$DATA" \
  --model "$MODEL" \
  --output "$SMOKE" \
  --items 24 \
  --seed 20261012 2>&1 | tee "$SMOKE/stdout.log"

python3 - "$SMOKE/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary.get("formal_release_allowed"):
    raise SystemExit("smoke gates failed; formal generation withheld")
PY

python3 ara/s3-generation/src/s3_multiresolution_language_curriculum_t0.py \
  --data "$DATA" \
  --model "$MODEL" \
  --output "$FORMAL" \
  --items 512 \
  --seed 20261012 2>&1 | tee "$FORMAL/stdout.log"

nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total --format=csv,noheader >"$ROOT/gpu_after.txt"
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$ROOT/SHA256SUMS"

