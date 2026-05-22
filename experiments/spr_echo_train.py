"""
SPR Echo Train — 训练并保存 H 参数
"""
import torch
import torch.nn.functional as F
import numpy as np

V, D, d = 128, 4, 8
torch.manual_seed(42)
tokens = torch.randn(V, d)

H = torch.randn(d, requires_grad=True)
opt = torch.optim.Adam([H], lr=0.05)

for step in range(200):
    opt.zero_grad()
    scores = tokens @ H
    emb_dist = torch.cdist(tokens, tokens)
    score_dist = torch.abs(scores[:, None] - scores[None, :])
    emb_dist_n = emb_dist / (emb_dist.max() + 1e-8)
    score_dist_n = score_dist / (score_dist.max() + 1e-8)
    loss = F.mse_loss(score_dist_n, emb_dist_n)
    loss.backward()
    opt.step()

print(f"final loss: {loss.item():.4f}")
print(f"H: {H.detach().numpy().round(3).tolist()}")

# 保存参数
torch.save({'H': H.detach(), 'd': d, 'D': D, 'seed': 42}, 'spr_echo_H.pt')
print("saved to spr_echo_H.pt")
