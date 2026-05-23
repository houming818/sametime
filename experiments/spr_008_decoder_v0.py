"""
SPR Decoder v3 — bilingual co-occ embeddings
en/de words co-embedded in shared space via sentence co-occurrence
"""
import torch, numpy as np, math, os
from collections import Counter

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
pairs = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(pairs) < 512:
            de, en = l.strip().split("\t", 1)
            pairs.append((de.split(), en.split()))

word2id_en, id2word_en = {}, {}
word2id_de, id2word_de = {}, {}
for de, en in pairs:
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w

V_en, V_de, d = len(word2id_en), len(word2id_de), 64
train_n = int(len(pairs) * 0.8)

# bilingual co-occ: en-en, de-de, AND en-de all in same matrix
V_all = V_en + V_de
coocc = np.zeros((V_all, d), dtype=np.float32)

def en_idx(w): return word2id_en[w]
def de_idx(w): return V_en + word2id_de[w]

for de, en in pairs[:train_n]:
    # en-en context
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

norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E_all = torch.tensor(coocc / norms, dtype=torch.float32)
E_en = E_all[:V_en]
E_de = E_all[V_en:]

# translation: en word → nearest de word
print(f"en={V_en} de={V_de} d={d}")
print(f"\nsimilarity samples:")
pairs_to_check = [('president','Präsident'),('strategy','Strategie'),('court','Gericht'),('cat','Katze'),('the','der')]
for en_w, de_w in pairs_to_check:
    if en_w in word2id_en and de_w in word2id_de:
        sim = float(torch.cosine_similarity(E_en[word2id_en[en_w]].unsqueeze(0), E_de[word2id_de[de_w]].unsqueeze(0)).item())
        print(f"  '{en_w}'↔'{de_w}': {sim:.3f}")

# tree on en
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

# Nearest neighbor: for each en word, H_leaf → nearest de word
refs, hyps = [], []
for i in range(train_n, len(pairs)):
    de, en = pairs[i]
    de_ref = [word2id_de[w] for w in de if w in word2id_de]
    hyp = []
    for w in en:
        if w in word2id_en:
            eid = word2id_en[w]
            scores = E_en[eid] @ E_de.T  # per-word nearest neighbor
            hyp.append(scores.argmax().item())
    refs.append(de_ref)
    hyps.append(hyp[:len(de_ref)])

def bleu(rf, hp):
    def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
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
print(f"BLEU-4 = {b:.2f}")

# samples
for i in [0, 2]:
    de, en = pairs[train_n + i]
    hyp_words = []
    for w in en[:8]:
        if w in word2id_en:
            scores = E_en[word2id_en[w]] @ E_de.T
            hyp_words.append(id2word_de.get(scores.argmax().item(), '?'))
    print(f"\n  src: {' '.join(en[:6])}")
    print(f"  ref: {' '.join(de[:6])}")
    print(f"  hyp: {' '.join(hyp_words[:6])}")
