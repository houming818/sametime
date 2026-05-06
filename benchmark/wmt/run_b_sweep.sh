#!/bin/bash
# B3-B5: Transformer architecture sweep (BPE, 30 epochs)
# Serial execution on io

CODE=/data/homecicd/sametime/code/wmt
IMAGE=reg.grepcode.cn/sati/sametime-base:cu121-py310
BASE="docker run --rm --gpus all -e PYTHONUNBUFFERED=1 \
    -e HF_DATASETS_CACHE=/data/datasets -e HF_HOME=/data/huggingface \
    -e HF_TRUST_REMOTE_CODE=1 \
    -e HTTP_PROXY=http://192.168.1.101:10809 \
    -e HTTPS_PROXY=http://192.168.1.101:10809 \
    -e NO_PROXY='localhost,127.0.0.0/24,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.grepcode.cn,*.sedcode.cn' \
    -v $CODE:/workspace -v /data/homecicd/sametime:/data \
    -w /workspace $IMAGE python3 -u phase6_transformer/train_bpe.py --epochs 30"

echo "===== B3: BPE d256/6L/4H dff=1024 =====" | tee $CODE/log_b3.txt
$BASE --exp-name B3 --d-model 256 --n-layers 6 --n-heads 4 --d-ff 1024 --output checkpoints/b3_d256_6l.pt 2>&1 | tee -a $CODE/log_b3.txt

echo "===== B4: BPE d512/3L/8H dff=2048 =====" | tee $CODE/log_b4.txt
$BASE --exp-name B4 --d-model 512 --n-layers 3 --n-heads 8 --d-ff 2048 --output checkpoints/b4_d512_3l.pt 2>&1 | tee -a $CODE/log_b4.txt

echo "===== B5: BPE d256/4L/4H dff=2048 =====" | tee $CODE/log_b5.txt
$BASE --exp-name B5 --d-model 256 --n-layers 4 --n-heads 4 --d-ff 2048 --output checkpoints/b5_d256_4l.pt 2>&1 | tee -a $CODE/log_b5.txt

echo "ALL DONE" | tee $CODE/log_b_done.txt
