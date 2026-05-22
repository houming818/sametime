"""
SPR P7 BLEU — 树做词分类器，重建句子算 BLEU
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math
from collections import Counter

# ── IWSLT14 数据 ──
cache = "/data/datasets/iwslt14/iwslt14.valid.de-en"
if os.path.exists(cache):
    with open(cache, "r", encoding="utf-8") as f:
        lines = [l.strip().split("\t", 1)[1] for l in f if "\t" in l][:512]
    sents = [l.split() for l in lines]
    print(f"IWSLT14 validation: {len(sents)} en sentences")
else:
    import random; random.seed(42)
    words = [f"w{i}" for i in range(200)]
    sents = [random.sample(words, random.randint(5, 12)) for _ in range(128)]
    print(f"using {len(sents)} random sentences")

# 词表
word2id, id2word = {}, {}
for s in sents:
    for w in s:
        if w not in word2id:
            idx = len(word2id)
            word2id[w] = idx
            id2word[idx] = w
V = len(word2id)
print(f"vocabulary: {V} words")

# ── 构建 token embedding ──
d = 16
torch.manual_seed(42)
embed = torch.randn(V, d) * 0.5

# ── 树模型 ──
depth = 9  # 512 leaves >> 200 words → perfect echo
n_leaves = 2**depth
weights = nn.ParameterList([nn.Parameter(torch.randn(d)*0.05) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.03)

def route(tokens, weights, depth):
    if depth == 1:
        probs_r = torch.sigmoid(tokens @ weights[0])
        assign = torch.zeros(len(tokens), 2)
        assign[:, 0] = 1 - probs_r
        assign[:, 1] = probs_r
        return assign
    probs_r = torch.sigmoid(tokens @ weights[0])
    child = route(tokens, weights[1:], depth-1)
    L = child.shape[1]
    assign = torch.zeros(len(tokens), L*2)
    assign[:, :L] = (1 - probs_r).unsqueeze(1) * child
    assign[:, L:] = probs_r.unsqueeze(1) * child
    return assign

# ── 训练：所有词走树，分类到叶子 ──
all_ids = list(range(V))
all_emb = embed[all_ids]

print(f"training {V} words into {n_leaves} leaves...")
for step in range(800):
    opt.zero_grad()
    assign = route(all_emb, weights, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ all_emb / leaf_sum.unsqueeze(1)
    dists = torch.cdist(all_emb, leaf_ctr)
    with torch.no_grad():
        best = dists.argmin(dim=1)
        target = torch.zeros_like(assign)
        target[range(V), best] = 1.0
    loss = F.mse_loss(assign, target)
    loss.backward()
    opt.step()

# ── 构建叶子查表：每片叶子最常见的词 ──
with torch.no_grad():
    assign_final = route(all_emb, [w.detach() for w in weights], depth)
    leaf_for_word = assign_final.argmax(dim=1)  # shape (V,)
    leaf_words = [[] for _ in range(n_leaves)]
    for wid in range(V):
        leaf_words[leaf_for_word[wid].item()].append(wid)
    leaf_top_word = {}
    for lid in range(n_leaves):
        if leaf_words[lid]:
            counts = Counter(leaf_words[lid])
            leaf_top_word[lid] = counts.most_common(1)[0][0]

print(f"trained. active leaves: {sum(1 for lw in leaf_words if lw)}/{n_leaves}")
print(f"avg words/leaf: {np.mean([len(lw) for lw in leaf_words if lw]):.1f}")

# ── BLEU 评估 ──
def compute_bleu(refs, hyps, max_n=4):
    """简化 BLEU 计算"""
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    precisions = []
    for n in range(1, max_n+1):
        match, total = 0, 0
        for ref, hyp in zip(refs, hyps):
            ref_ng = Counter(ngrams(ref, n))
            hyp_ng = Counter(ngrams(hyp, n))
            total += sum(hyp_ng.values())
            match += sum(min(hyp_ng[k], ref_ng.get(k, 0)) for k in hyp_ng)
        precisions.append(match / max(total, 1))
    
    bp_vals = [1 - len(ref) / max(len(hyp), 1) for ref, hyp in zip(refs, hyps) if len(hyp) > 0]
    bp = min(1.0, math.exp(max(bp_vals) if bp_vals else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in precisions) / max_n)

refs, hyps = [], []
for s in sents[:128]:
    ref_ids = [word2id[w] for w in s]  # 原句词 ID
    pred_ids = []
    for w in s:
        lid = leaf_for_word[word2id[w]].item()
        pred_ids.append(leaf_top_word.get(lid, 0))
    refs.append(ref_ids)
    hyps.append(pred_ids)

bleu = compute_bleu(refs, hyps)
refs_text = [[id2word[r] for r in ref[:10]] for ref in refs[:3]]
hyps_text = [[id2word[h] if h in id2word else '?' for h in hyp[:10]] for hyp in hyps[:3]]

print(f"\n=== BLEU-4 = {bleu:.4f} ===")
print(f"sample reconstructions:")
for i in range(3):
    print(f"  ref: {' '.join(refs_text[i])}")
    print(f"  hyp: {' '.join(hyps_text[i])}")
    print()
