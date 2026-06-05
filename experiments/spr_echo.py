import os, sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json

from spr_tree_layer import HeapTreeLayer

def train_echo(depth, dim, agg_method, run_name, vocab_size=32000, batch_size=1024, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] 开始 Echo 训练 | Depth: {depth}, Dim: {dim}, Agg: {agg_method} | Device: {device}")
    
    # 1. 构建模型
    tree_layer = HeapTreeLayer(vocab_size=vocab_size, embed_dim=dim, depth=depth, agg_method=agg_method)
    
    # Decoder 仅使用单层 Linear，验证树层的表征能力
    decoder = nn.Linear(dim, vocab_size)
    
    model = nn.Sequential(tree_layer, decoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 2. 构建自编码 Dummy 数据集 (全词表)
    # 对于 Echo 任务，只需让模型能够 "吃 Token，吐 Token"
    all_tokens = torch.arange(vocab_size)
    dataset = TensorDataset(all_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 训练循环
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == x).sum().item()
            total += x.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        best_acc = max(best_acc, epoch_acc)
        
        print(f"[{run_name}] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
        # 简单早停逻辑 (如果达到完美重构则提前结束)
        if epoch_acc >= 0.999:
            print(f"[{run_name}] 完美重构达成，提前结束训练。")
            break
            
    # 4. 记录结果
    end_time = time.time()
    result = {
        "run_name": run_name,
        "depth": depth,
        "dim": dim,
        "agg_method": agg_method,
        "best_acc": best_acc,
        "final_loss": epoch_loss,
        "time_sec": end_time - start_time
    }
    
    # 追加到结果文件 (同时写到 workspace 和 /data 持久卷)
    for d in ["results", "/data/results"]:
        try:
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/echo_matrix_results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
        except:
            pass
        
    print(f"[{run_name}] 训练完成！Best Acc: {best_acc:.4f}。结果已保存。")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--agg_method", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_echo(args.depth, args.dim, args.agg_method, args.run_name, epochs=args.epochs)
