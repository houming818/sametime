"""
算清楚：路径向量是什么，cos 怎么来的
"""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm
import numpy as np

device='cuda'; d=128; td=5; V=16000
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')

t_nodes = nn.ModuleList([nn.Embedding(2**i, d).to(device) for i in range(td)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])

print("="*70)
print("1. 路径向量是什么？")
print("   path_vector = node[L0][0] + node[L1][i1] + node[L2][i2] + node[L3][i3] + node[L4][i4]")

# tid=0 的路径
path_0 = (0, 0, 0, 0, 0)
v0 = sum(t_nodes[l](torch.tensor([path_0[l]], device=device)).squeeze(0) for l in range(td))
v0n = F.normalize(v0, dim=-1)

path_15 = (0, 1, 3, 7, 15)
v15 = sum(t_nodes[l](torch.tensor([path_15[l]], device=device)).squeeze(0) for l in range(td))
v15n = F.normalize(v15, dim=-1)

print(f"\n2. path_0  = {path_0}")
print(f"   path_15 = {path_15}")
print(f"   cos(path_0, path_15) = {(v0n * v15n).sum().item():.4f}")

print(f"\n3. 各层节点对比:")
for l in range(td):
    n0 = t_nodes[l](torch.tensor([path_0[l]], device=device)).squeeze(0)
    n15 = t_nodes[l](torch.tensor([path_15[l]], device=device)).squeeze(0)
    c = F.cosine_similarity(n0.unsqueeze(0), n15.unsqueeze(0)).item()
    print(f"   L{l}: {path_0[l]}↔{path_15[l]}  |n0|={n0.norm():.2f} |n15|={n15.norm():.2f}  cos={c:.3f}")

print(f"\n4. root 占路径的比例:")
v0_children = sum(t_nodes[l](torch.tensor([path_0[l]], device=device)).squeeze(0) for l in range(1, td))
v15_children = sum(t_nodes[l](torch.tensor([path_15[l]], device=device)).squeeze(0) for l in range(1, td))
root = t_nodes[0](torch.tensor([0], device=device)).squeeze(0)
print(f"   |root| / |children_0|  = {root.norm():.2f} / {v0_children.norm():.2f} = {root.norm()/v0_children.norm():.2f}")
print(f"   |root| / |children_15| = {root.norm():.2f} / {v15_children.norm():.2f} = {root.norm()/v15_children.norm():.2f}")
print(f"   去掉 root 后 cos(children_0, children_15) = {F.cosine_similarity(F.normalize(v0_children, dim=-1).unsqueeze(0), F.normalize(v15_children, dim=-1).unsqueeze(0)).item():.4f}")

# 16 条路径全部 pairwise cos
print(f"\n5. 16 条路径 pairwise cos 分布:")
all_paths = []
for L4 in range(16):
    L3 = L4 // 2
    L2 = L3 // 2
    L1 = L2 // 2
    p = [sum(t_nodes[l](torch.tensor([[0,L1,L2,L3,L4][l]], device=device)).squeeze(0) for l in range(td))]
    all_paths.append(F.normalize(p[0], dim=-1))
all_paths = torch.stack(all_paths)
cos_mat = all_paths @ all_paths.T
mask = ~torch.eye(16, dtype=torch.bool, device=device)
off_diag = cos_mat[mask]
print(f"   mean={off_diag.mean():.4f}  max={off_diag.max():.4f}  min={off_diag.min():.4f}")
print(f"   {off_diag.max().item() - off_diag.min().item():.4f} 的跨度 — 区分度空间")

# heap_world 的作用
print(f"\n6. heap_world 怎么区分的 — CMul 的作用:")
L0 = nn.Embedding(V, d).to(device); L0.load_state_dict(ckpt['L0'])
t_merge = nn.Linear(d, d).to(device); t_merge.load_state_dict(ckpt['t_merge'])
tid_cat = sp.encode_as_ids("cat")[0]
tid_mao = sp.encode_as_ids("猫")[0]
hw_cat = F.normalize(t_merge(torch.cat([
    L0(torch.tensor(tid_cat, device=device))[:d//2] * all_paths[0,:d//2] - L0(torch.tensor(tid_cat, device=device))[d//2:] * all_paths[0,d//2:],
    L0(torch.tensor(tid_cat, device=device))[:d//2] * all_paths[0,d//2:] + L0(torch.tensor(tid_cat, device=device))[d//2:] * all_paths[0,:d//2]
], -1)), dim=-1)
hw_mao = F.normalize(t_merge(torch.cat([
    L0(torch.tensor(tid_mao, device=device))[:d//2] * all_paths[0,:d//2] - L0(torch.tensor(tid_mao, device=device))[d//2:] * all_paths[0,d//2:],
    L0(torch.tensor(tid_mao, device=device))[:d//2] * all_paths[0,d//2:] + L0(torch.tensor(tid_mao, device=device))[d//2:] * all_paths[0,:d//2]
], -1)), dim=-1)
print(f"   L0[cat] 本身 cos = {F.cosine_similarity(L0.weight[tid_cat].unsqueeze(0), L0.weight[tid_mao].unsqueeze(0)).item():.4f}")
print(f"   经过相同 path=0 heap_world 后 cos = {F.cosine_similarity(hw_cat.unsqueeze(0), hw_mao.unsqueeze(0)).item():.4f}")

# 同一个 token 换不同 path
hw_cat_p0 = hw_cat
hw_cat_p15 = F.normalize(t_merge(torch.cat([
    L0(torch.tensor(tid_cat, device=device))[:d//2] * all_paths[15,:d//2] - L0(torch.tensor(tid_cat, device=device))[d//2:] * all_paths[15,d//2:],
    L0(torch.tensor(tid_cat, device=device))[:d//2] * all_paths[15,d//2:] + L0(torch.tensor(tid_cat, device=device))[d//2:] * all_paths[15,:d//2]
], -1)), dim=-1)
print(f"   cat 用 path_0  vs cat 用 path_15  cos = {F.cosine_similarity(hw_cat_p0.unsqueeze(0), hw_cat_p15.unsqueeze(0)).item():.4f}")
print(f"   猫 用 path_0  vs cat 用 path_15  cos = {F.cosine_similarity(hw_mao.unsqueeze(0), hw_cat_p15.unsqueeze(0)).item():.4f}")
