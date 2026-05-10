#!/bin/bash
# C-series: WMT14 De-En Transformer sweep
# Serial execution — reads queue_status.txt for checkpoint resume
CODE=/data/homecicd/sametime/code/wmt
IMG=reg.grepcode.cn/sati/sametime-base:cu121-py310
ENV="-e PYTHONUNBUFFERED=1 -e HF_DATASETS_CACHE=/data/datasets -e HF_HOME=/data/huggingface -e HF_TRUST_REMOTE_CODE=1 -e HTTP_PROXY=http://192.168.1.101:10809 -e HTTPS_PROXY=http://192.168.1.101:10809 -e NO_PROXY='localhost,127.0.0.0/24,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.grepcode.cn,*.sedcode.cn'"
VOL="-v $CODE:/workspace -v /data/homecicd/sametime:/data"
STATUS="$CODE/queue_status.txt"
RUN="docker run --rm --gpus all --memory=12g --memory-swap=12g $ENV $VOL -w /workspace $IMG python3 -u phase6_transformer/train_wmt14.py"

check() { grep -q "$1 DONE" $STATUS 2>/dev/null; }
mark_start() { echo "$1 START $(date -Iseconds)" | tee -a $STATUS; }
mark_done() { echo "$1 DONE $(date -Iseconds) BLEU=$2" | tee -a $STATUS; }

echo "=== T0: word d256/3L WMT14 ===" | tee $CODE/log_t0.txt
check T0 || { mark_start T0; $RUN --exp-name T0 --epochs 20 --d-model 256 --n-layers 3 --n-heads 4 --d-ff 1024 --word-level --output checkpoints/t0_word.pt 2>&1 | tee -a $CODE/log_t0.txt; B=$(grep 'best_bleu=' $CODE/log_t0.txt | tail -1 | grep -oP '[\d.]+'); mark_done T0 $B; }

echo "=== T1: BPE d256/3L WMT14 ===" | tee $CODE/log_t1.txt
check T1 || { mark_start T1; $RUN --exp-name T1 --epochs 20 --d-model 256 --n-layers 3 --n-heads 4 --d-ff 1024 --output checkpoints/t1_bpe_d256_3l.pt 2>&1 | tee -a $CODE/log_t1.txt; B=$(grep 'best_bleu=' $CODE/log_t1.txt | tail -1 | grep -oP '[\d.]+'); mark_done T1 $B; }

echo "=== T2: BPE d512/6L/8H WMT14 (paper config) ===" | tee $CODE/log_t2.txt
check T2 || { mark_start T2; $RUN --exp-name T2 --epochs 20 --d-model 512 --n-layers 6 --n-heads 8 --d-ff 2048 --output checkpoints/t2_paper.pt 2>&1 | tee -a $CODE/log_t2.txt; B=$(grep 'best_bleu=' $CODE/log_t2.txt | tail -1 | grep -oP '[\d.]+'); mark_done T2 $B; }

echo "=== T3: BPE d256/6L WMT14 (our best arch) ===" | tee $CODE/log_t3.txt
check T3 || { mark_start T3; $RUN --exp-name T3 --epochs 20 --d-model 256 --n-layers 6 --n-heads 4 --d-ff 1024 --output checkpoints/t3_d256_6l.pt 2>&1 | tee -a $CODE/log_t3.txt; B=$(grep 'best_bleu=' $CODE/log_t3.txt | tail -1 | grep -oP '[\d.]+'); mark_done T3 $B; }

echo "=== T4: C3 best + beam=4 ===" | tee $CODE/log_t4.txt
check T4 || { mark_start T4; $RUN --exp-name T4 --epochs 20 --d-model 256 --n-layers 6 --n-heads 4 --d-ff 1024 --beam-size 4 --output checkpoints/t4_beam.pt 2>&1 | tee -a $CODE/log_t4.txt; B=$(grep 'best_bleu=' $CODE/log_t4.txt | tail -1 | grep -oP '[\d.]+'); mark_done T4 $B; }

echo "ALL DONE" | tee -a $STATUS
