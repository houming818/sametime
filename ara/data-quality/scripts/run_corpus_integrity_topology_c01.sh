#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/home/houming818/SameTime}"
data_root="${DATA_ROOT:-/home/houming818/datasets/nio/releases}"
evidence_dir="${EVIDENCE_DIR:-$repo_root/ara/data-quality/evidence/corpus_integrity_topology_c01_n14}"
src="$repo_root/ara/data-quality/src/corpus_integrity_topology_c01.py"
mono="$data_root/NioText-ZH-Integrity-2985K-v1/data.jsonl"
parallel="$data_root/NioClean-ZHEN-S098-7M-v2/pairs.tsv"

mkdir -p "$evidence_dir/mono" "$evidence_dir/parallel"

{
  printf 'claim=NIO-CORPUS-INTEGRITY-C01\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'kernel=%s\n' "$(uname -srvmo)"
  printf 'python=%s\n' "$(python3 --version 2>&1)"
  printf 'logical_cpus=%s\n' "$(nproc)"
  printf 'repo_root=%s\n' "$repo_root"
  printf 'data_root=%s\n' "$data_root"
  printf 'evidence_dir=%s\n' "$evidence_dir"
  stat -c 'input=%n bytes=%s mtime=%y' "$mono" "$parallel"
} >"$evidence_dir/run_environment.txt"

printf '%q ' "$0" "$@" >"$evidence_dir/command.txt"
printf '\n' >>"$evidence_dir/command.txt"

nice -n 10 python3 "$src" \
  --kind mono \
  --input "$mono" \
  --output "$evidence_dir/mono" \
  --expected-rows 2972976 \
  --expected-bytes 5147059454 \
  >"$evidence_dir/mono/stdout.log" 2>"$evidence_dir/mono/stderr.log" &
mono_pid=$!

nice -n 10 python3 "$src" \
  --kind parallel \
  --input "$parallel" \
  --output "$evidence_dir/parallel" \
  --expected-rows 7304358 \
  --expected-bytes 1535169936 \
  >"$evidence_dir/parallel/stdout.log" 2>"$evidence_dir/parallel/stderr.log" &
parallel_pid=$!

status=0
wait "$mono_pid" || status=$?
wait "$parallel_pid" || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

python3 "$src" \
  --kind merge \
  --mono-summary "$evidence_dir/mono/summary.json" \
  --parallel-summary "$evidence_dir/parallel/summary.json" \
  --output "$evidence_dir/summary.json" \
  >"$evidence_dir/stdout.log" 2>"$evidence_dir/stderr.log"
