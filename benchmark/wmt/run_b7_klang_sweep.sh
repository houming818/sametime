#!/bin/bash
# B7_CE + B7_k*: K_lang sweep on BPE d256/6L (B3 architecture), 20 epochs
# k=0 (CE-only), 24, 50, 100, 170
# Serial execution

CODE=/data/homecicd/sametime/code/wmt
IMAGE=reg.grepcode.cn/sati/sametime-base:cu121-py310
BASE="docker run --rm --gpus all --memory=12g --memory-swap=12g -e PYTHONUNBUFFERED=1 \
    -e HF_DATASETS_CACHE=/data/datasets -e HF_HOME=/data/huggingface \
    -e HF_TRUST_REMOTE_CODE=1 \
    -e HTTP_PROXY=http://192.168.1.101:10809 \
    -e HTTPS_PROXY=http://192.168.1.101:10809 \
    -e NO_PROXY='localhost,127.0.0.0/24,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.grepcode.cn,*.sedcode.cn' \
    -v $CODE:/workspace -v /data/homecicd/sametime:/data \
    -w /workspace $IMAGE python3 -u phase6_transformer/train_bpe.py \
    --epochs 20 --d-model 256 --n-layers 6 --n-heads 4 --d-ff 1024"

echo "===== B7_CE: CE-only baseline =====" | tee $CODE/log_b7_ce.log
$BASE --exp-name B7_CE --output checkpoints/b7_ce.pt 2>&1 | tee -a $CODE/log_b7_ce.log

echo "===== B7_k24: k=24 (K_lang*8000) =====" | tee $CODE/log_b7_k24.log
$BASE --exp-name B7_k24 --dual-head --restricted-bleu --k-lang 24 --sb-alpha 0.8 --output checkpoints/b7_k24.pt 2>&1 | tee -a $CODE/log_b7_k24.log

echo "===== B7_k50: k=50 =====" | tee $CODE/log_b7_k50.log
$BASE --exp-name B7_k50 --dual-head --restricted-bleu --k-lang 50 --sb-alpha 0.8 --output checkpoints/b7_k50.pt 2>&1 | tee -a $CODE/log_b7_k50.log

echo "===== B7_k100: k=100 =====" | tee $CODE/log_b7_k100.log
$BASE --exp-name B7_k100 --dual-head --restricted-bleu --k-lang 100 --sb-alpha 0.8 --output checkpoints/b7_k100.pt 2>&1 | tee -a $CODE/log_b7_k100.log

echo "===== B7_k170: k=170 (original) =====" | tee $CODE/log_b7_k170.log
$BASE --exp-name B7_k170 --dual-head --restricted-bleu --k-lang 170 --sb-alpha 0.8 --output checkpoints/b7_k170.pt 2>&1 | tee -a $CODE/log_b7_k170.log

echo "===== B7_k500: k=500 (too wide) =====" | tee $CODE/log_b7_k500.log
$BASE --exp-name B7_k500 --dual-head --restricted-bleu --k-lang 500 --sb-alpha 0.8 --output checkpoints/b7_k500.pt 2>&1 | tee -a $CODE/log_b7_k500.log

echo "===== B7_full: k=8000 (full vocab) =====" | tee $CODE/log_b7_full.log
$BASE --exp-name B7_full --dual-head --restricted-bleu --k-lang 8000 --sb-alpha 0.8 --output checkpoints/b7_full.pt 2>&1 | tee -a $CODE/log_b7_full.log

echo "ALL DONE" | tee $CODE/log_b7_done.txt
