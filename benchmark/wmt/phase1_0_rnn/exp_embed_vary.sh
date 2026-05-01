#!/bin/bash
# exp_embed_vary.sh — Phase 1.0 RNN 解耦合实验：固定 hidden=512, vary embed
# 验证假设: hash 碰撞在 embedding 宽度而非 hidden 容量
set -e

DOCKER_HOST=${DOCKER_HOST:-ssh://houming818@io.grepcode.cn}
IO="houming818@io.grepcode.cn"
IO_DATA="/data/homecicd/sametime/code/wmt"
IO_SAFE="/data/homecicd/sametime/results"
COMPOSE_FILE="benchmark/wmt/phase1_0_rnn/docker-compose.yml"

echo "===== 对照分析实验 start ====="
date
echo "hidden=512 fixed, embed vary"

echo ">>> rsync..."
make wmt_base

for E in 32 64 128 256 512 1024; do
    echo ""
    echo "============================================"
    echo "  HIDDEN=512 | EMBED=$E | EPOCHS=5 | SEED=42"
    echo "============================================"
    
    ssh $IO "rm -f ${IO_DATA}/metrics.jsonl"
    ssh $IO "mkdir -p ${IO_SAFE}"
    
    DOCKER_HOST=$DOCKER_HOST \
        docker compose -f $COMPOSE_FILE \
        run --rm phase --hidden 512 --embed $E --epochs 5 --seed 42
    
    ssh $IO "cp ${IO_DATA}/metrics.jsonl ${IO_SAFE}/metrics_h512_e${E}.jsonl"
    scp ${IO}:${IO_SAFE}/metrics_h512_e${E}.jsonl ./metrics_h512_e${E}.jsonl 2>/dev/null
    
    LINES=$(wc -l < ./metrics_h512_e${E}.jsonl 2>/dev/null || echo 0)
    echo "  saved: $LINES epochs"
done

echo ""
date
echo "===== 对照分析实验 end ====="

# summary
echo ""
echo "===== Summary ====="
for E in 32 64 128 256 512 1024; do
    F="metrics_h512_e${E}.jsonl"
    if [ -f "$F" ]; then
        BEST=$(python3 -c "
import json
bleus = [json.loads(l)['bleu'] for l in open('$F')]
print(f'{max(bleus):.2f}')
")
        echo "  h512_e$E: best BLEU=$BEST"
    fi
done
