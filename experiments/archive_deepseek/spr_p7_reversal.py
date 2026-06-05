"""
P7 实验 v4 — 真正的语义路由
每个 token 独立判定 sigmoid(W @ token) → 软左/右
"""
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(42)
tokens = torch.randn(128, 8)
depth, n_leaves = 4, 16

def semantic_assign(token_probs_l, token_probs_r, depth, level, max_depth):
    """
    token_probs_l: (N,) — 每个 token 在本层走左的概率
    token_probs_r: (N,) — 每个 token 在本层走右的概率  
    递归：每个 token 带着自己的概率进入下一层
    返回: (N, 2^depth) 分配矩阵
    """
    n = token_probs_l.shape[0]
    if depth == 0 or n <= 1:
        return torch.ones(n, 1)
    
    # 所有 token 都进入左右子树（乘以各自的概率）
    la = semantic_assign(token_probs_l, token_probs_l, depth-1, level+1, max_depth)
    ra = semantic_assign(token_probs_r, token_probs_r, depth-1, level+1, max_depth)
    nL = la.shape[1]
    assign = torch.zeros(n, nL * 2)
    assign[:, :nL] = token_probs_l.unsqueeze(1) * la
    assign[:, nL:] = token_probs_r.unsqueeze(1) * ra
    return assign

def route(tokens, weights, depth, level=0):
    n = len(tokens)
    if depth == 0 or n <= 1:
        return torch.ones(n, 1)
    w = weights[level]
    scores = tokens @ w
    probs_r = torch.sigmoid(scores)
    probs_l = 1 - probs_r
    child = route(tokens, weights, depth-1, level+1)  # 只调一次！
    L = child.shape[1]
    assign = torch.zeros(n, L * 2)
    assign[:, :L] = probs_l.unsqueeze(1) * child
    assign[:, L:] = probs_r.unsqueeze(1) * child
    return assign

weights = nn.ParameterList([nn.Parameter(torch.randn(8)*0.05) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.05)

# 初始状态
with torch.no_grad():
    assign_b = route(tokens, [w.detach() for w in weights], depth)
    cur_b = assign_b.argmax(dim=1)
    ls_b = assign_b.sum(dim=0) + 1e-8
    lc_b = assign_b.T @ tokens / ls_b.unsqueeze(1)
    dst_b = torch.cdist(tokens, lc_b)
    best_b = dst_b.argmin(dim=1)
    matched_b = (cur_b == best_b).sum().item()

print(f"初始匹配: {matched_b}/128\n")

for step in range(300):
    opt.zero_grad()
    assign = route(tokens, weights, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ tokens / leaf_sum.unsqueeze(1)
    dists = torch.cdist(tokens, leaf_ctr)
    
    with torch.no_grad():
        best = dists.argmin(dim=1)
        target = torch.zeros_like(assign)
        target[range(128), best] = 1.0
    
    loss = F.mse_loss(assign, target)
    loss.backward()
    opt.step()
    
    if step % 60 == 0:
        with torch.no_grad():
            a = route(tokens, [w.detach() for w in weights], depth)
            cur = a.argmax(dim=1)
            ls = a.sum(dim=0) + 1e-8
            lc = a.T @ tokens / ls.unsqueeze(1)
            d = torch.cdist(tokens, lc)
            best_now = d.argmin(dim=1)
            matched = (cur == best_now).sum().item()
        print(f"  step {step:3d}: loss={loss.item():.4f}  matched={matched}/128")

with torch.no_grad():
    assign_f = route(tokens, [w.detach() for w in weights], depth)
    cur = assign_f.argmax(dim=1)
    ls = assign_f.sum(dim=0) + 1e-8
    lc = assign_f.T @ tokens / ls.unsqueeze(1)
    d = torch.cdist(tokens, lc)
    best_now = d.argmin(dim=1)
    matched_f = (cur == best_now).sum().item()
    assign_f2 = route(tokens, [w.detach() for w in weights], depth)
    same = (cur == assign_f2.argmax(dim=1)).sum().item()
    active = (ls > 0).sum().item()

print(f"\nfinal: matched={matched_f}/128  active={active}/{n_leaves}  deterministic={same}/128")
print("REVERSAL PROVED" if same == 128 and matched_f > matched_b else "needs work")
