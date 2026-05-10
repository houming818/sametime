#!/bin/bash
# A13-A15: SB gradient decay fix — multiply_log mode test
# k=170 (K_lang sweet spot) fixed, vary mode + alpha

CODE=/data/homecicd/sametime/code/wmt
IMAGE=reg.grepcode.cn/sati/sametime-base:cu121-py310
BASE="docker run --rm --gpus all --memory=12g --memory-swap=12g \
    -e HF_DATASETS_CACHE=/data/datasets \
    -e HF_HOME=/data/huggingface \
    -e HF_TRUST_REMOTE_CODE=1 \
    -e HTTP_PROXY=http://192.168.1.101:10809 \
    -e HTTPS_PROXY=http://192.168.1.101:10809 \
    -e NO_PROXY=localhost,127.0.0.0/24,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.grepcode.cn,*.sedcode.cn \
    -v $CODE:/workspace -v /data/homecicd/sametime:/data \
    -w /workspace $IMAGE python3 phase2_bahdanau/train.py --epochs 5 --dual-head --restricted-bleu --k-lang 170"

echo "===== A13: multiply_sqrt beta=0.5 alpha=0.5 =====" | tee $CODE/log_a13.txt
$BASE --sb-mode multiply_sqrt --sb-beta 0.5 --sb-alpha 0.5 --output checkpoints/a13_sqrt05.pt 2>&1 | tee -a $CODE/log_a13.txt

echo "===== A14: multiply_sqrt beta=0.3 alpha=0.3 =====" | tee $CODE/log_a14.txt
$BASE --sb-mode multiply_sqrt --sb-beta 0.3 --sb-alpha 0.3 --output checkpoints/a14_sqrt03.pt 2>&1 | tee -a $CODE/log_a14.txt

echo "===== A15: multiply_linear alpha=0.8 (baseline) =====" | tee $CODE/log_a15.txt
$BASE --sb-mode multiply_linear --sb-alpha 0.8 --output checkpoints/a15_lin08.pt 2>&1 | tee -a $CODE/log_a15.txt

echo "ALL DONE" | tee $CODE/log_done2.txt
