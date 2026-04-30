#!/bin/bash
# exp_hidden_size.sh — Phase 1.0 RNN 隐藏层维度对照实验
# 控制变量: --hidden {128,256,512}, 固定其他所有参数
set -e

DOCKER_HOST=${DOCKER_HOST:-ssh://houming818@io.grepcode.cn}
IO="houming818@io.grepcode.cn"
IO_DATA="/data/homecicd/sametime/code/wmt"

echo "===== 对照分析实验 start ====="
echo "$(date)"
echo "DOCKER_HOST=$DOCKER_HOST"

# Step 1: 同步代码到 io
echo ""
echo ">>> rsync code to io..."
make wmt_base

# Step 2: 三组实验
for H in 128 256 512; do
    echo ""
    echo "============================================"
    echo "  HIDDEN=$H | EMBED=$H | EPOCHS=5 | SEED=42"
    echo "============================================"
    
    # 清空上次 metrics
    ssh $IO "rm -f ${IO_DATA}/metrics.jsonl"
    
    # 训练
    DOCKER_HOST=$DOCKER_HOST \
        docker compose -f benchmark/wmt/phase1_0_rnn/docker-compose.yml \
        run --rm phase --hidden $H --embed $H --epochs 5 --seed 42
    
    # 保存结果
    ssh $IO "cp ${IO_DATA}/metrics.jsonl ${IO_DATA}/metrics_h${H}.jsonl"
    ssh $IO "cp ${IO_DATA}/checkpoints/phase1_0_rnn.pt ${IO_DATA}/checkpoints/phase1_0_h${H}.pt 2>/dev/null || true"
    
    echo "  HIDDEN=$H done"
done

# Step 3: 拉取结果到本地
echo ""
echo ">>> fetching results..."
for H in 128 256 512; do
    scp ${IO}:${IO_DATA}/metrics_h${H}.jsonl ./metrics_h${H}.jsonl 2>/dev/null
    echo "  metrics_h${H}.jsonl: $(cat ./metrics_h${H}.jsonl | wc -l) lines"
done

echo ""
echo "===== 对照分析实验 end ====="
echo "$(date)"
