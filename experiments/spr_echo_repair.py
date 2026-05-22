"""
SPR Echo Self-Repair — sigmoid 软路由 + 平衡损失
鼓励 tokens 分散到多片叶子
"""
import torch
import torch.nn as nn

torch.manual_seed(42)
tokens = torch.randn(128, 8)
depth, n_leaves = 4, 16

def soft_leaf_assignment(tokens, weights, depth, level=0):
    n = len(tokens)
    if depth == 0 or n <= 1:
        return torch.ones(n, 1)
    w = weights[level]
    scores = tokens @ w
    probs_r = torch.sigmoid(scores)
    probs_l = 1 - probs_r
    mid = n // 2
    left_a = soft_leaf_assignment(tokens[:mid], weights, depth-1, level+1)
    right_a = soft_leaf_assignment(tokens[mid:], weights, depth-1, level+1)
    nL = left_a.shape[1]
    assign = torch.zeros(n, nL * 2)
    assign[:mid, :nL] = probs_l[:mid].unsqueeze(1) * left_a
    assign[mid:, nL:] = probs_r[mid:].unsqueeze(1) * right_a
    return assign

weights = nn.ParameterList([nn.Parameter(torch.randn(8) * 0.1) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.03)

for step in range(500):
    opt.zero_grad()
    assign = soft_leaf_assignment(tokens, weights, depth)  # (128, 16)
    leaf_sum = assign.sum(dim=0) + 1e-8                    # (16,)
    
    # 组内方差
    leaf_ctr = assign.T @ tokens / leaf_sum.unsqueeze(1)
    dist = torch.cdist(tokens, leaf_ctr)
    within_loss = (assign * dist).sum() / tokens.shape[0]
    
    # 平衡正则：惩罚叶子使用不均衡
    ideal = tokens.shape[0] / n_leaves
    balance_loss = ((leaf_sum - ideal) ** 2).mean() / ideal
    loss = within_loss + 0.5 * balance_loss
    
    loss.backward()
    opt.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            hard = assign.argmax(dim=1)
            counts = torch.bincount(hard, minlength=n_leaves)
            active = (counts > 0).sum().item()
            maxp = assign.max(dim=1).values.mean().item()
        print(f"  step {step:3d}: loss={loss.item():.4f}  within={within_loss.item():.4f}  "
              f"active={active}/{n_leaves}  max_prob={maxp:.3f}")

# 硬化
with torch.no_grad():
    assign_f = soft_leaf_assignment(tokens, [w.detach() for w in weights], depth)
    hard = assign_f.argmax(dim=1)
    counts = torch.bincount(hard, minlength=n_leaves)
    assign_f2 = soft_leaf_assignment(tokens, [w.detach() for w in weights], depth)
    same = (hard == assign_f2.argmax(dim=1)).sum().item()
    
print(f"\nactive leaves: {(counts>0).sum().item()}/{n_leaves}")
print(f"leaf sizes   : {counts.tolist()}")
print(f"same leaf    : {same}/{tokens.shape[0]}")
print("ECHO OK" if same == tokens.shape[0] else "NON-DETERMINISTIC")
