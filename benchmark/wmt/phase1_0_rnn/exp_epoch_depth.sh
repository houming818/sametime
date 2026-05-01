#!/bin/bash
# exp_epoch_depth.sh — 验证预测 3: epoch 精度不改变 K_lang
set -e

DOCKER_HOST=${DOCKER_HOST:-ssh://houming818@io.grepcode.cn}
IO="houming818@io.grepcode.cn"
IO_DATA="/data/homecicd/sametime/code/wmt"
IO_SAFE="/data/homecicd/sametime/results"
COMPOSE_FILE="benchmark/wmt/phase1_0_rnn/docker-compose.yml"

echo "===== epoch 深度证伪实验 ====="
date

make wmt_base

# 组 A: H=512 跑 20 epoch (baseline 收敛)
for CONFIG in "512 512" "1024 1024" "512 128"; do
    read H E <<< "$CONFIG"
    TAG="h${H}_e${E}_ep20"
    echo ""
    echo "=== $TAG ==="
    ssh $IO "rm -f ${IO_DATA}/metrics.jsonl"
    ssh $IO "mkdir -p ${IO_SAFE}"
    
    DOCKER_HOST=$DOCKER_HOST docker compose -f $COMPOSE_FILE \
        run --rm phase --hidden $H --embed $E --epochs 20 --seed 42
    
    ssh $IO "cp ${IO_DATA}/metrics.jsonl ${IO_SAFE}/metrics_${TAG}.jsonl"
    scp ${IO}:${IO_SAFE}/metrics_${TAG}.jsonl ./metrics_${TAG}.jsonl 2>/dev/null
    
    LINES=$(wc -l < ./metrics_${TAG}.jsonl 2>/dev/null || echo 0)
    # 提取关键 epoch 的 BLEU
    python3 -c "
import json
lines = [json.loads(l) for l in open('metrics_${TAG}.jsonl')]
best = max(lines, key=lambda x: x['bleu'])
for i in [0,4,9,14,19]:
    if i < len(lines):
        l = lines[i]
        print(f'  epoch={l[\"epoch\"]:2d}  loss={l[\"loss\"]:.3f}  BLEU={l[\"bleu\"]:.2f}')
print(f'  best: epoch={best[\"epoch\"]} BLEU={best[\"bleu\"]:.2f}')
print(f'  total lines: {len(lines)}')
"
done

echo ""
date
echo "===== done ====="
