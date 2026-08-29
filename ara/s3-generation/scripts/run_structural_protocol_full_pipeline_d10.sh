#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

mode="${1:-smoke}"
runner="ara/s3-generation/src/s3_structural_protocol_full_pipeline_d10.py"
root="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/${mode}_seed11001"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
d09_checkpoint="ara/s3-generation/evidence/s3_structural_slot_ownership_d09_scale/formal_seed10901/checkpoint_best.pt"
text_data="/home/nio/datasets/nio/releases/NioText-ZH-Integrity-2985K-v1/data.jsonl"
parallel_data="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv"

test -f "$runner"
test -f "$source_checkpoint"
test -f "$d09_checkpoint"
test -f "$text_data"
test -f "$parallel_data"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

if [[ "$mode" == "full" ]]; then
  printf '%s  %s\n' \
    '04a90d88b51755561645d0fec962fc8bd5e642d099423348417de9318e22c94e' "$text_data" \
    '299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8' "$parallel_data" \
    | sha256sum -c -
  eval_rows=1000
  pretrain_wake=10000
  task_wake=25000
  timeout_note="full"
else
  eval_rows=128
  pretrain_wake=50
  task_wake=50
  timeout_note="smoke"
fi

mkdir -p "$root"
nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_before.csv"

python3 "$runner" \
  --stage pretrain --mode "$mode" --source-checkpoint "$source_checkpoint" \
  --warm-start "$d09_checkpoint" --evidence-dir "$root" \
  --text-data "$text_data" --parallel-data "$parallel_data" \
  --seed 11001 --ownership-seed 11002 --eval-rows "$eval_rows" \
  --wake-every "$pretrain_wake" --resume --device cuda \
  2>&1 | tee "$root/pretrain.log"

python3 - "$root/pretrain/summary.json" <<'PY'
import json
import sys
summary=json.load(open(sys.argv[1], encoding='utf-8'))
if summary['decision'] != 'stage_supported':
    raise SystemExit('pretrain stage did not pass; task stage is blocked')
PY

python3 "$runner" \
  --stage task --mode "$mode" --source-checkpoint "$source_checkpoint" \
  --warm-start "$root/pretrain/checkpoint_best.pt" --evidence-dir "$root" \
  --text-data "$text_data" --parallel-data "$parallel_data" \
  --seed 11001 --ownership-seed 11002 --eval-rows "$eval_rows" \
  --wake-every "$task_wake" --min-steps 100000 --resume --device cuda \
  2>&1 | tee "$root/task.log"

python3 - "$root" "$mode" <<'PY'
import json
import sys
from pathlib import Path
root=Path(sys.argv[1])
mode=sys.argv[2]
pre=json.loads((root/'pretrain/summary.json').read_text(encoding='utf-8'))
task=json.loads((root/'task/summary.json').read_text(encoding='utf-8'))
assert pre['claim']==task['claim']=='S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10'
assert pre['gates']['reload'] and task['gates']['reload']
summary={'claim':pre['claim'],'mode':mode,'pretrain':pre['decision'],'task':task['decision'],
         'pretrain_best_nll':pre['best_valid']['mean_nll'],
         'task_best_nll':task['best_valid']['mean_nll'],
         'task_bleu4_median':task['best_generation']['bleu4_median'],
         'passed':all(pre['gates'].values()) and all(task['gates'].values())}
(root/'pipeline_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'event':'pipeline_complete',**summary},ensure_ascii=False))
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_after.csv"

sendme -s "D10 ${timeout_note} finished" -f "$root/pipeline_summary.json" || true
