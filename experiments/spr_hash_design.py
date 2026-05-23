"""
SPR Hash Function Design
H_parent = combine(H_left, H_right)  — recursive composition
对比 plain mean vs weighted vs LSTM merge
"""
import torch, torch.nn as nn

torch.manual_seed(42)
E = torch.randn(10, 4) * 0.5
tA, tB, tC = E[0], E[1], E[2]

def r(t):
    return [round(x.item(), 2) for x in t]

def hash_mean(l, r):
    return (l + r) / 2

def hash_weighted(l, r, w=0.6):
    return w * l + (1-w) * r

lstm = nn.LSTMCell(4, 4)
def hash_lstm(l, r):
    h = torch.zeros(4); c = torch.zeros(4)
    h, _ = lstm(l.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
    h, _ = lstm(r.unsqueeze(0), (h.squeeze(0).unsqueeze(0), c.unsqueeze(0)))
    return h.squeeze(0)

print("=== Merkle hash: (A+B)+C ===")
for label, hf in [("mean", hash_mean), ("weighted", hash_weighted), ("LSTM", hash_lstm)]:
    H_AB = hf(tA, tB)
    H_ABC = hf(H_AB, tC)
    scores = H_ABC @ E.T
    top = scores.argsort(descending=True)[:3]
    top_s = [scores[i].item() for i in top]
    print(f"{label:8s}: H={r(H_ABC)}  top3={top.tolist()}  scores={[round(s,2) for s in top_s]}")

print("\n=== A-then-B vs B-then-A ===")
for label, hf in [("mean", hash_mean), ("weighted", hash_weighted)]:
    H_AB = hf(tA, tB)
    H_BA = hf(tB, tA)
    print(f"{label:8s}: same={torch.allclose(H_AB, H_BA)}")

print("\n=== 设计结论 ===")
print("mean    : 顺序不敏感，天然可组合，echo 够用")
print("weighted: 顺序敏感（单参数），翻译需要")
print("LSTM    : 顺序敏感 + 可训练，SPR-007 候选")
