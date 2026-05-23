"""
SPR Decoder v4 — WMT14 50K training sentences
bilingual co-occ embeddings → nearest-neighbor translation
"""
import torch, numpy as np, math, os
from collections import Counter

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load_pairs(path, n):
    pairs = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l:
                de, en = l.strip().split("\t", 1)
                pairs.append((de.split(), en.split()))
    return pairs

print("loading training data...")
train_pairs = load_pairs(train_file, 50000)
val_pairs = load_pairs(val_file, 1000)
print(f"train={len(train_pairs)} val={len(val_pairs)}")

# build vocab from train only
word2id_en, id2word_en = {}, {}
word2id_de, id2word_de = {}, {}
for de, en in train_pairs:
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w

V_en, V_de, d = len(word2id_en), len(word2id_de), 128
V_all = V_en + V_de
print(f"en vocab={V_en} de vocab={V_de} d={d}")

# bilingual co-occ
coocc = np.zeros((V_all, d), dtype=np.float32)

def en_idx(w): return word2id_en[w]
def de_idx(w): return V_en + word2id_de[w]

for di, (de, en) in enumerate(train_pairs):
    # en-en context (window=3)
    for i, w in enumerate(en):
        wid = en_idx(w)
        for j in range(max(0,i-3), min(len(en), i+4)):
            if i != j and en[j] in word2id_en:
                coocc[wid, en_idx(en[j]) % d] += 1
    # de-de context
    for i, w in enumerate(de):
        wid = de_idx(w)
        for j in range(max(0,i-3), min(len(de), i+4)):
            if i != j and de[j] in word2id_de:
                coocc[wid, de_idx(de[j]) % d] += 1
    # en-de cross pairs
    for en_w in en:
        for de_w in de:
            coocc[en_idx(en_w), de_idx(de_w) % d] += 1
            coocc[de_idx(de_w), en_idx(en_w) % d] += 1
    if di % 10000 == 9999:
        print(f"  processed {di+1} sentences")

norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E_all = torch.tensor(coocc / norms, dtype=torch.float32)
E_en = E_all[:V_en]
E_de = E_all[V_en:]

# similarity check
pairs_check = [('the','der'),('president','Präsident'),('strategy','Strategie'),('court','Gericht'),('european','europäischen'),('commission','Kommission')]
print(f"\nbilingual sims:")
for en_w, de_w in pairs_check:
    if en_w in word2id_en and de_w in word2id_de:
        s = float(torch.cosine_similarity(E_en[word2id_en[en_w]].unsqueeze(0), E_de[word2id_de[de_w]].unsqueeze(0)).item())
        print(f"  '{en_w}'↔'{de_w}': {s:.3f}")

# per-word nearest neighbor BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]

# ── tree routing ──
depth = 8; n_leaves = 1<<depth; n_nodes = n_leaves-1

def soft_assign(emb, nw, depth, idx=0):
    if depth==0: return torch.ones(len(emb),1)
    sc = emb @ nw[idx]; pr = torch.sigmoid(sc)
    ch = soft_assign(emb, nw, depth-1, 2*idx+1)
    L = ch.shape[1]; a = torch.zeros(len(emb), L*2)
    a[:,:L] = (1-pr).unsqueeze(1)*ch; a[:,L:] = pr.unsqueeze(1)*ch
    return a

with torch.no_grad():
    leaf_en = soft_assign(E_en, torch.randn(n_nodes,d)*0.05, depth).argmax(dim=1)

# leaf→de word frequency table (from 50K training sentences)
leaf_de = [Counter() for _ in range(n_leaves)]
for de, en in train_pairs:
    en_ids = [word2id_en[w] for w in en if w in word2id_en]
    de_ids = [word2id_de[w] for w in de if w in word2id_de]
    if not en_ids: continue
    leaf_votes = Counter([leaf_en[eid].item() for eid in en_ids])
    best_leaf = leaf_votes.most_common(1)[0][0]
    for did in de_ids:
        leaf_de[best_leaf][did] += 1

leaf_trans_best = {}
for lid in range(n_leaves):
    if leaf_de[lid]:
        leaf_trans_best[lid] = leaf_de[lid].most_common(1)[0][0]

active = sum(1 for l in leaf_de if l)
print(f"\ntree: leaves={n_leaves} active={active}")

# BLEU
refs, hyps = [], []
for de, en in val_pairs:
    de_ref = [word2id_de[w] for w in de if w in word2id_de]
    hyp = []
    for w in en:
        if w in word2id_en:
            eid = word2id_en[w]
            lid = leaf_en[eid].item()
            hyp.append(leaf_trans_best.get(lid, 0))
    refs.append(de_ref)
    hyps.append(hyp[:len(de_ref)])

def bleu(rf, hp):
    C=Counter; ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(rf,hp):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1))
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

b = bleu(refs, hyps)
print(f"\nBLEU-4 = {b:.2f}")

# samples
for i in [0, 3, 7]:
    de, en = val_pairs[i]
    hyp_words = []
    for w in en[:8]:
        if w in word2id_en:
            eid = word2id_en[w]
            lid = leaf_en[eid].item()
            hyp_words.append(id2word_de.get(leaf_trans_best.get(lid, 0), '?'))
    print(f"\n  src: {' '.join(en[:6])}")
    print(f"  ref: {' '.join(de[:6])}")
    print(f"  hyp: {' '.join(hyp_words[:6])}")
