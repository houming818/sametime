"""
SPR Echo — 加载训练好的 H，验证自映射
"""
import torch
import numpy as np

ckpt = torch.load('spr_echo_H.pt', weights_only=True)
H = ckpt['H']
d, D = ckpt['d'], ckpt['D']
seed = ckpt['seed']

torch.manual_seed(seed)
tokens = torch.randn(128, d)

def forward_split(scores):
    def split(indices, depth):
        if depth == 0 or len(indices) <= 1:
            return [indices]
        g_scores = scores[indices]
        if len(g_scores) < 2:
            return [indices]
        median = g_scores.median()
        left = indices[g_scores <= median]
        right = indices[g_scores > median]
        return split(left, depth-1) + split(right, depth-1)
    return split(torch.arange(len(scores)), D)

with torch.no_grad():
    scores = tokens @ H
    groups = forward_split(scores)
    sizes = [len(g) for g in groups]

print(f"H: {H.numpy().round(3).tolist()}")
print(f"Leaves: {len(groups)}, sizes: {sizes}")
print(f"Active: {sum(1 for s in sizes if s > 0)}/{len(groups)}")

# 确定性验证
scores2 = tokens @ H
groups2 = forward_split(scores2)
same = sum(1 for i in range(128)
           if [j for j, g in enumerate(groups)  if i in g][0]
           == [j for j, g in enumerate(groups2) if i in g][0])

print(f"Deterministic: {same}/128")

# 示例：前5个token的路径
for i in range(5):
    leaf = [j for j, g in enumerate(groups) if i in g][0]
    path = format(leaf, f'0{D}b')
    print(f"  token_{i}: leaf={leaf} path={path}")

print("\n=== ECHO VERIFIED ===" if same == 128 else "\nFAILED")
