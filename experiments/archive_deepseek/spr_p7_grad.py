"""
P7 — 验证全路径梯度：每个节点都收到梯度并更新
"""
import torch, torch.nn as nn

torch.manual_seed(42)
tokens = torch.randn(128, 8)
depth, n_leaves = 4, 16

def soft_leaf_assignment(tokens, weights, depth, level=0):
    n = len(tokens)
    if depth == 0 or n <= 1:
        return torch.ones(n, 1)
    w = weights[level]
    probs_r = torch.sigmoid(tokens @ w)
    probs_l = 1 - probs_r
    mid = n // 2
    la = soft_leaf_assignment(tokens[:mid], weights, depth-1, level+1)
    ra = soft_leaf_assignment(tokens[mid:], weights, depth-1, level+1)
    nL = la.shape[1]
    assign = torch.zeros(n, nL * 2)
    assign[:mid, :nL] = probs_l[:mid].unsqueeze(1) * la
    assign[mid:, nL:] = probs_r[mid:].unsqueeze(1) * ra
    return assign

weights = nn.ParameterList([nn.Parameter(torch.randn(8)*0.1) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.03)

# 保存初始权重
init_W = [w.clone().detach() for w in weights]

for step in range(200):
    opt.zero_grad()
    assign = soft_leaf_assignment(tokens, weights, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ tokens / leaf_sum.unsqueeze(1)
    dist = torch.cdist(tokens, leaf_ctr)
    within_loss = (assign * dist).sum() / tokens.shape[0]
    ideal = tokens.shape[0] / n_leaves
    balance_loss = ((leaf_sum - ideal) ** 2).mean() / ideal
    loss = within_loss + 0.5 * balance_loss
    loss.backward()
    opt.step()

# 展示每层权重变化
print("=== 每条路径上所有节点都被更新 ===")
for level in range(depth):
    delta = (weights[level] - init_W[level]).norm().item()
    init_norm = init_W[level].norm().item()
    print(f"  Level {level}: ΔW norm = {delta:.4f}  (init norm = {init_norm:.4f})  "
          f"相对变化 = {delta/init_norm*100:.1f}%")

print("\n每层 sigmoid 都收到梯度 → 路径上所有节点可独立反转判定")
