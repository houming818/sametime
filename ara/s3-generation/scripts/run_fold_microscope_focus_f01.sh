#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
root="ara/s3-generation/evidence/s3_fold_microscope_focus_f01"
checkpoint="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt"
specimens="ara/s3-generation/data/fold_microscope_specimens_f01.jsonl"
runner="ara/s3-generation/src/treeheap_fold_matrix_decode_probe.py"

if [[ "$mode" == "smoke" ]]; then
  out="$root/smoke"
  limit=2
  max_tokens=32
  scales=(0 0.5 0.7071067811865476 1.0)
elif [[ "$mode" == "formal" ]]; then
  out="$root/formal"
  limit=0
  max_tokens=64
  scales=(0 0.125 0.25 0.375 0.5 0.625 0.7071067811865476 0.8 1.0)
else
  echo "usage: $0 [smoke|formal]" >&2
  exit 2
fi

mkdir -p "$out"
sha256sum "$checkpoint" "$specimens" "$runner" >"$out/inputs.sha256"

args=(
  python3 "$runner"
  --checkpoint "$checkpoint"
  --specimens "$specimens"
  --depth all
  --up-mode native
  --up-mode bypass
  --max-new-tokens "$max_tokens"
  --limit-specimens "$limit"
  --device cuda
  --output "$out/results.json"
)
for scale in "${scales[@]}"; do
  args+=(--scale "$scale")
done

printf '%q ' "${args[@]}" >"$out/command.txt"
printf '\n' >>"$out/command.txt"
"${args[@]}" | tee "$out/stdout.log"
