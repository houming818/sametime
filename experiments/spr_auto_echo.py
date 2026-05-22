"""
SPR Auto Echo — 简单版：fake token 做 auto echo proof
"""
import torch, torch.nn as nn, torch.nn.functional as F

# 假造句子的 token embedding (4 个不同的 "词")
tokens = torch.tensor([
    [0.1, 0.2, 0.3, 0.1],  # token "the"
    [0.2, 0.1, 0.2, 0.3],  # token "cat"  
    [0.3, 0.3, 0.1, 0.2],  # token "slept"
    [0.1, 0.2, 0.3, 0.1],  # token "the" (重复)
], dtype=torch.float32)

depth, d = 2, 4
n_leaves = 2**depth

weights = nn.ParameterList([nn.Parameter(torch.randn(d)*0.05) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.05)

def route(tokens, weights, depth):
    if depth == 1:
        w = weights[0]
        probs_r = torch.sigmoid(tokens @ w)
        probs_l = 1 - probs_r
        assign = torch.zeros(len(tokens), 2)
        assign[:, 0], assign[:, 1] = probs_l, probs_r
        return assign
    w = weights[0]
    probs_r = torch.sigmoid(tokens @ w)
    probs_l = 1 - probs_r
    child = route(tokens, weights[1:], depth-1)
    L = child.shape[1]
    assign = torch.zeros(len(tokens), L*2)
    assign[:, :L] = probs_l.unsqueeze(1) * child
    assign[:, L:] = probs_r.unsqueeze(1) * child
    return assign

# 训练前
with torch.no_grad():
    a0 = route(tokens, [w.detach() for w in weights], depth)
    l0 = a0.argmax(dim=1)
    ls = a0.sum(dim=0) + 1e-8
    lc = a0.T @ tokens / ls.unsqueeze(1)
    d0 = torch.cdist(tokens, lc)
    b0 = d0.argmin(dim=1)
    matched0 = (l0 == b0).sum().item()

print(f"tokens: {tokens.tolist()}")
print(f"initial: matched={matched0}/4  leaves={l0.tolist()}\n")

for step in range(100):
    opt.zero_grad()
    assign = route(tokens, weights, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ tokens / leaf_sum.unsqueeze(1)
    dists = torch.cdist(tokens, leaf_ctr)

    with torch.no_grad():
        best = dists.argmin(dim=1)
        target = torch.zeros_like(assign)
        target[range(4), best] = 1.0
    loss = F.mse_loss(assign, target)
    loss.backward()
    opt.step()

    if step % 25 == 0:
        with torch.no_grad():
            a = route(tokens, [w.detach() for w in weights], depth)
            cur = a.argmax(dim=1)
            ls = a.sum(dim=0) + 1e-8
            lc = a.T @ tokens / ls.unsqueeze(1)
            d = torch.cdist(tokens, lc)
            b = d.argmin(dim=1)
            m = (cur == b).sum().item()
        print(f"  step {step:3d}: loss={loss.item():.4f}  matched={m}/4  leaves={cur.tolist()}")

with torch.no_grad():
    af = route(tokens, [w.detach() for w in weights], depth)
    lf = af.argmax(dim=1)
    af2 = route(tokens, [w.detach() for w in weights], depth)
    same = (lf == af2.argmax(dim=1)).sum().item()

print(f"\nfinal: matched={m}/4  deterministic={same}/4")
print("AUTO ECHO OK" if same == 4 else "needs work")
