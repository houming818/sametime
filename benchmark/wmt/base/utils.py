"""
base/utils.py — 工具函数：随机种子、Checkpoint、日志
"""

import os
import json
import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, model, optimizer, epoch, bleu, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "bleu": bleu,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def log_metrics(epoch, loss, bleu, lr, log_path="metrics.jsonl"):
    with open(log_path, "a") as f:
        f.write(json.dumps({"epoch": epoch, "loss": loss, "bleu": bleu, "lr": lr}) + "\n")
