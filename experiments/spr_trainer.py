import os
import sys
import argparse
import torch
import torch.optim as optim
import time
import json

from spr_dataset import load_tokenizer, build_anchors, collate_anchors_to_padded
from spr_model import SemanticPrefixRoutingModel
from spr_eval import evaluate_alignment

def train_alignment(depth, dim, agg_method, run_name, epochs=100, lr=3e-3, temp=0.07, warm_from=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] 开始 Translation 对齐训练 | Depth: {depth}, Dim: {dim}, Agg: {agg_method} | Device: {device}")
    
    # 1. 加载 Tokenizer 与词表大小
    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    print(f"[{run_name}] Tokenizer 加载成功。词表大小 V: {vocab_size}")
    
    # 2. 构建去重训练锚点 (仅使用 LaBSE 引导，完全隔离 Gold 手工词对)
    train_anchors = build_anchors(sp, use_labse_only=True)
    N_train = len(train_anchors)
    print(f"[{run_name}] 训练对齐锚点对数量: {N_train}")
    
    # 3. 准备全量训练数据的 Padded Tensor
    en_train_list = [e for e, _ in train_anchors]
    zh_train_list = [z for _, z in train_anchors]
    en_pad, en_mask = collate_anchors_to_padded(en_train_list, device)
    zh_pad, zh_mask = collate_anchors_to_padded(zh_train_list, device)
    
    # 4. 构建对齐模型
    model = SemanticPrefixRoutingModel(vocab_size=vocab_size, embed_dim=dim, depth=depth, agg_method=agg_method)
    
    # 支持从预训练 Echo 恢复权重 (若提供)
    if warm_from:
        if os.path.exists(warm_from):
            print(f"[{run_name}] 尝试从 Echo 预训练权重恢复: {warm_from}")
            try:
                ckpt = torch.load(warm_from, map_location="cpu")
                # 兼容 Sequential 结构与直接层结构
                if '0.L0.weight' in ckpt:
                    # 原本是 Sequential(HeapTreeLayer, Decoder)
                    state_dict = {}
                    for k, v in ckpt.items():
                        if k.startswith('0.'):
                            state_dict[k[2:]] = v
                    model.tree_layer.load_state_dict(state_dict, strict=False)
                elif 'tree_layer' in ckpt:
                    model.load_state_dict(ckpt, strict=False)
                else:
                    model.tree_layer.load_state_dict(ckpt, strict=False)
                print(f"[{run_name}] 成功加载预训练 L0 与 Heap Tree 路由权重。")
            except Exception as e:
                print(f"[{run_name}] 加载预训练权重失败: {e}，将采用随机初始化。")
        else:
            print(f"[{run_name}] 预训练路径不存在: {warm_from}，将采用随机初始化。")
            
    model = model.to(device)
    
    # 5. 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 6. 训练循环
    best_gold_acc = 0.0
    best_train_acc = 0.0
    start_time = time.time()
    
    # 每个 epoch 跑 200 个 gradient steps 快速更新收敛
    steps_per_epoch = 150
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = model.compute_infonce_loss(en_pad, en_mask, zh_pad, zh_mask, temp=temp)
            loss.backward()
            # 梯度裁剪以提高稳定度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / steps_per_epoch
        
        # 每个 epoch 结束时进行全面评估
        metrics = evaluate_alignment(model, sp, train_anchors, device)
        train_acc = metrics['train_acc']
        train_cos = metrics['train_cos']
        gold_acc = metrics['gold_acc']
        gold_cos = metrics['gold_cos']
        
        best_gold_acc = max(best_gold_acc, gold_acc)
        best_train_acc = max(best_train_acc, train_acc)
        
        print(f"[{run_name}] Epoch {epoch+1:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% (Cos: {train_cos:.3f}) | "
              f"Gold Acc: {gold_acc:.1f}% (Cos: {gold_cos:.3f})")
              
        # 极佳收敛则可提前终止
        if train_acc >= 99.9 and gold_acc >= 95.0:
            print(f"[{run_name}] 达到极佳的双语对齐效果，提前结束。")
            break
            
    elapsed_time = time.time() - start_time
    print(f"[{run_name}] 训练完成！Best Train Acc: {best_train_acc:.1f}%, Best Gold Acc: {best_gold_acc:.1f}%, 总耗时: {elapsed_time:.1f}s")
    
    # 7. 保存对齐后的完整模型模型
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{run_name}_alignment.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'depth': depth,
        'dim': dim,
        'agg_method': agg_method,
        'vocab_size': vocab_size,
        'best_gold_acc': best_gold_acc,
        'best_train_acc': best_train_acc
    }, save_path)
    print(f"[{run_name}] 模型已保存至 {save_path}")
    
    # 保存结果指标
    result = {
        "run_name": run_name,
        "depth": depth,
        "dim": dim,
        "agg_method": agg_method,
        "best_train_acc": best_train_acc,
        "best_gold_acc": best_gold_acc,
        "time_sec": elapsed_time
    }
    
    for d in ["results", "/data/results"]:
        try:
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/alignment_results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
        except:
            pass
            
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--agg_method", type=str, default="complex_mul")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warm_from", type=str, default="")
    args = parser.parse_args()
    
    train_alignment(
        depth=args.depth,
        dim=args.dim,
        agg_method=args.agg_method,
        run_name=args.run_name,
        epochs=args.epochs,
        lr=args.lr,
        warm_from=args.warm_from
    )
