#!/bin/bash
# exp_hidden_scale.sh — Phase 1.0 RNN 隐藏层维度指数对照实验
# 尺度: 2^4=16 → 2^10=1024 (8192/65536 OOM, 跳过)
# 指标: params, loss, BLEU, time/epoch
set -e

DOCKER_HOST=${DOCKER_HOST:-ssh://houming818@io.grepcode.cn}
IO="houming818@io.grepcode.cn"
IO_DATA="/data/homecicd/sametime/code/wmt"
IO_SAFE="/data/homecicd/sametime/results"
COMPOSE_FILE="benchmark/wmt/phase1_0_rnn/docker-compose.yml"
SUMMARY="./metrics_hidden_scale.jsonl"

echo "===== 对照分析实验 start ====="
date

echo ""
echo ">>> rsync code to io..."
make wmt_base

for H in 16 32 64 128 256 512 1024; do
    T0=$(date +%s)
    echo ""
    echo "============================================"
    echo "  HIDDEN=$H | EMBED=$H | EPOCHS=5 | SEED=42"
    echo "============================================"
    
    ssh $IO "rm -f ${IO_DATA}/metrics.jsonl"
    
    if DOCKER_HOST=$DOCKER_HOST \
        docker compose -f $COMPOSE_FILE \
        run --rm phase --hidden $H --embed $H --epochs 5 --seed 42 2>&1 | tee /tmp/exp_${H}.log; then
        STATUS="ok"
    else
        STATUS="fail"
        echo "  HIDDEN=$H failed, saving partial"
    fi
    
    T1=$(date +%s)
    DURATION=$((T1 - T0))
    
    # 保存到安全目录（不被 make wmt_base 的 rm -rf 清掉）
    ssh $IO "mkdir -p ${IO_SAFE} && cp ${IO_DATA}/metrics.jsonl ${IO_SAFE}/metrics_h${H}.jsonl 2>/dev/null || true"
    scp ${IO}:${IO_SAFE}/metrics_h${H}.jsonl ./metrics_h${H}.jsonl 2>/dev/null || true
    
    # 写入摘要
    EPOCHS_OK=$(wc -l < ./metrics_h${H}.jsonl 2>/dev/null || echo 0)
    echo "{\"hidden\":$H,\"status\":\"$STATUS\",\"epochs\":$EPOCHS_OK,\"duration_sec\":$DURATION,\"timestamp\":\"$(date -Iseconds)\"}" >> $SUMMARY
    echo "  saved: $EPOCHS_OK epochs, ${DURATION}s"
done

echo ""
echo "===== 对照分析实验 end ====="
date
echo "summary: $SUMMARY"
cat $SUMMARY
