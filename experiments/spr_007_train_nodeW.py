"""
SPR-007: Train node_W + E for semantic leaf cohesion
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 64:
            raw.append(l.split("\t", 1)[1].strip().lower().split())

PAD = 8
word2id = {}
sent_tokens = []
for s in raw:
    ids = [0]*PAD
    for i, w in enumerate(s[:PAD]):
        if w not in word2id: word2id[w] = len(word2id)
        ids[i] = word2id[w]
    sent_tokens.append(ids)

V, d = len(word2id), 16
depth = 5
n_nodes, n_leaves = (1<<depth)-1, 1<<depth
print(f"sents={len(raw)} vocab={V} depth={depth} nodes={n_nodes} leaves={n_leaves}")

torch.manual_seed(42)
E = nn.Parameter(torch.randn(V, d) * 0.5)
node_W = nn.Parameter(torch.randn(n_nodes, d) * 0.05)
opt = torch.optim.Adam([node_W, E], lr=0.03)

def soft_assign(emb, node_W, depth, node_idx=0):
    if depth == 0:
        return torch.ones(len(emb), 1)
    scores = emb @ node_W[node_idx]
    probs_r = torch.sigmoid(scores)
    probs_l = 1 - probs_r
    child = soft_assign(emb, node_W, depth-1, 2*node_idx+1)
    L = child.shape[1]
    assign = torch.zeros(len(emb), L*2)
    assign[:, :L] = probs_l.unsqueeze(1) * child
    assign[:, L:] = probs_r.unsqueeze(1) * child
    return assign

def cohesion(hard, emb):
    """emb must be detached"""
    scores = []
    for leaf in range(n_leaves):
        ids = (hard == leaf).nonzero(as_tuple=True)[0]
        if len(ids) > 1:
            e = emb[ids]
            sim = F.cosine_similarity(e.unsqueeze(1), e.unsqueeze(0), dim=2)
            mask = torch.eye(len(ids)) == 0
            if mask.any():
                scores.append(sim[mask].mean().item())
    return np.mean(scores) if scores else 0.0

all_ids = list(range(V))

# 训练前
with torch.no_grad():
    a0 = soft_assign(E.detach(), node_W.detach(), depth)
    h0 = a0.argmax(dim=1)
    coh0 = cohesion(h0, E.detach())
    leaves0 = len(torch.unique(h0))
    print(f"before: used={leaves0}/{n_leaves}  cohesion={coh0:.4f}")

for step in range(500):
    opt.zero_grad()
    assign = soft_assign(E, node_W, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ E / leaf_sum.unsqueeze(1)
    
    # cohesion loss: tok ns to their leaf center
    hard = assign.argmax(dim=1)
    cohesion_loss = 0.0
    for leaf in range(n_leaves):
        ids = (hard == leaf).nonzero(as_tuple=True)[0]
        if len(ids) > 1:
            cohesion_loss += F.mse_loss(E[ids], leaf_ctr[leaf].unsqueeze(0).expand(len(ids), -1))
    
    ideal = V / n_leaves
    balance_loss = ((leaf_sum - ideal) ** 2).mean() / ideal
    loss = cohesion_loss + 0.3 * balance_loss
    loss.backward()
    opt.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            a = soft_assign(E.detach(), node_W.detach(), depth)
            h = a.argmax(dim=1)
            coh = cohesion(h, E.detach())
            used = len(torch.unique(h))
        print(f"  step {step:3d}: loss={loss.item():.3f}  coh={coh:.4f}  leaves={used}/{n_leaves}")

with torch.no_grad():
    a_f = soft_assign(E.detach(), node_W.detach(), depth)
    h_f = a_f.argmax(dim=1)
    coh_f = cohesion(h_f, E.detach())
    used_f = len(torch.unique(h_f))
    
    leaf_words = [[] for _ in range(n_leaves)]
    for wid in range(V):
        leaf_words[h_f[wid].item()].append(wid)
    multi = sum(1 for lw in leaf_words if len(lw) > 1)
    solo = sum(1 for lw in leaf_words if len(lw) == 1)

    # 组间分离度
    lc = a_f.T @ E.detach() / (a_f.sum(dim=0) + 1e-8).unsqueeze(1)
    inter_dist = torch.cdist(lc, lc).mean().item()
    intra_dist = torch.cdist(E.detach(), E.detach()).mean().item()

print(f"\nfinal: coh={coh_f:.4f} (start={coh0:.4f})  Δ={coh_f-coh0:+.4f}  leaves={used_f}/{n_leaves}")
print(f"       solo={solo} multi={multi}  params={n_nodes*d + V*d}")
print(f"       inter-leaf dist={inter_dist:.4f}  intra-token dist={intra_dist:.4f}  collapse={'YES' if intra_dist < inter_dist else 'NO'}")

id2word = {v:k for k,v in word2id.items()}
if multi > 0:
    print("\n=== 叶内碰撞组 ===")
    for lid, words in sorted(enumerate(leaf_words), key=lambda x: -len(x[1]))[:5]:
        ws = [id2word[w] for w in words[:6]]
        print(f"  leaf {lid} ({len(words)}w): {ws}")
