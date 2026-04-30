#!/bin/bash
# exp_hidden_size.sh — Phase 1.0 RNN 隐藏层维度对照实验
# 变量: hidden={128,256,512}, 固定: embed=hidden, epochs=5, layers=2, seed=42, lr=1e-3
set -e

DOCKER_HOST=${DOCKER_HOST:-ssh://houming818@io.grepcode.cn}
IO="houming818@io.grepcode.cn"
IO_DATA="/data/homecicd/sametime/code/wmt"
COMPOSE_FILE="benchmark/wmt/phase1_0_rnn/docker-compose.yml"

echo "===== 对照分析实验 start ====="
echo "$(date)"

echo ""
echo ">>> rsync code to io..."
make wmt_base

for H in 128 256 512; do
    echo ""
    echo "============================================"
    echo "  HIDDEN=$H | EMBED=$H | EPOCHS=5 | SEED=42"
    echo "============================================"
    
    ssh $IO "rm -f ${IO_DATA}/metrics.jsonl"
    
    DOCKER_HOST=$DOCKER_HOST \
        docker compose -f $COMPOSE_FILE \
        run --rm phase --hidden $H --embed $H --epochs 5 --seed 42
    
    # 保存
    ssh $IO "cp ${IO_DATA}/metrics.jsonl ${IO_DATA}/metrics_h${H}.jsonl 2>/dev/null"
    ssh $IO "cp ${IO_DATA}/checkpoints/phase1_0_rnn.pt ${IO_DATA}/checkpoints/phase1_0_h${H}.pt 2>/dev/null || true"
    
    echo "  HIDDEN=$H done"
done

echo ""
echo ">>> fetching results..."
for H in 128 256 512; do
    scp ${IO}:${IO_DATA}/metrics_h${H}.jsonl ./metrics_h${H}.jsonl 2>/dev/null
    LINES=$(wc -l < ./metrics_h${H}.jsonl 2>/dev/null || echo 0)
    echo "  metrics_h${H}.jsonl: $LINES epochs"
done

echo ""
echo "===== 对照分析实验 end ====="
echo "$(date)"
