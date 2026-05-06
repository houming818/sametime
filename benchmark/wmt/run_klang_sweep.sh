#!/bin/bash
# A9-A12: Restricted SoftBLEU K_lang sweep
# Run sequentially on io

CODE=/data/homecicd/sametime/code/wmt
IMAGE=reg.grepcode.cn/sati/sametime-base:cu121-py310
BASE="docker run --rm --gpus all \
    -e HF_DATASETS_CACHE=/data/datasets \
    -e HF_HOME=/data/huggingface \
    -e HF_TRUST_REMOTE_CODE=1 \
    -e HTTP_PROXY=http://192.168.1.101:10809 \
    -e HTTPS_PROXY=http://192.168.1.101:10809 \
    -e NO_PROXY=localhost,127.0.0.0/24,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.grepcode.cn,*.sedcode.cn \
    -v $CODE:/workspace -v /data/homecicd/sametime:/data \
    -w /workspace $IMAGE python3 phase2_bahdanau/train.py --epochs 4"

echo "===== A8: CE-only (baseline) =====" | tee $CODE/log_a8.txt
$BASE --output checkpoints/a8_ce_only.pt 2>&1 | tee -a $CODE/log_a8.txt

echo "===== A9: DH k=50 =====" | tee $CODE/log_a9.txt
$BASE --dual-head --restricted-bleu --k-lang 50 --output checkpoints/a9_k50.pt 2>&1 | tee -a $CODE/log_a9.txt

echo "===== A10: DH k=170 =====" | tee $CODE/log_a10.txt
$BASE --dual-head --restricted-bleu --k-lang 170 --output checkpoints/a10_k170.pt 2>&1 | tee -a $CODE/log_a10.txt

echo "===== A11: DH k=500 =====" | tee $CODE/log_a11.txt
$BASE --dual-head --restricted-bleu --k-lang 500 --output checkpoints/a11_k500.pt 2>&1 | tee -a $CODE/log_a11.txt

echo "===== A12: DH full-softmax =====" | tee $CODE/log_a12.txt
$BASE --dual-head --output checkpoints/a12_dh_full.pt 2>&1 | tee -a $CODE/log_a12.txt

echo "ALL DONE" | tee $CODE/log_done.txt
