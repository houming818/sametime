"""
SPR-007 → Translation: bilingual tree + leaf decoder
German sentence → tree → leaf hash → predict English word
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os
from collections import Counter, defaultdict

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

print("loading...")
train_pairs = load_pairs(train_file, 50000)
val_pairs = load_pairs(val_file, 500)

# bilingual vocab
word2id_de, id2word_de = {}, {}
word2id_en, id2word_en = {}, {}
for de, en in train_pairs + val_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w

V_de, V_en, d = len(word2id_de), len(word2id_en), 64
V_all = V_de + V_en
print(f"de={V_de} en={V_en} d={d}")

# bilingual co-occ: de-en cross pairs in same space
coocc = np.zeros((V_all, d), dtype=np.float32)
for de, en in train_pairs:
    # en-en context
    for i, w in enumerate(en):
        wid = word2id_en[w]
        for j in range(max(0,i-3), min(len(en), i+4)):
            if i != j and en[j] in word2id_en:
                coocc[wid, word2id_en[en[j]] % d] += 1
    # de-de context
    for i, w in enumerate(de):
        wid = V_en + word2id_de[w]
        for j in range(max(0,i-3), min(len(de), i+4)):
            if i != j and de[j] in word2id_de:
                coocc[wid, (V_en + word2id_de[de[j]]) % d] += 1
    # de-en cross
    for de_w in de:
        for en_w in en:
            coocc[V_en + word2id_de[de_w], word2id_en[en_w] % d] += 1
            coocc[word2id_en[en_w], (V_en + word2id_de[de_w]) % d] += 1

norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32).cuda()
E_de = E[V_en:V_all]
E_en = E[:V_en]

# tree on English words → leaves
depth = 8; n_leaves = 1<<depth; n_nodes = n_leaves-1

def soft_assign(emb, nw, depth, idx=0):
    if depth==0: return torch.ones(len(emb), 1, device=emb.device)
    sc = emb @ nw[idx]; pr = torch.sigmoid(sc)
    ch = soft_assign(emb, nw, depth-1, 2*idx+1)
    L = ch.shape[1]
    a = torch.zeros(len(emb), L*2, device=emb.device)
    a[:,:L] = (1-pr).unsqueeze(1)*ch; a[:,L:] = pr.unsqueeze(1)*ch
    return a

with torch.no_grad():
    leaf_en = soft_assign(E_en, torch.randn(n_nodes, d).cuda()*0.05, depth).argmax(dim=1)

# Build: for each leaf, which German words co-occur?
leaf_to_de = [Counter() for _ in range(n_leaves)]
for de, en in train_pairs:
    en_ids = [word2id_en[w] for w in en if w in word2id_en]
    de_ids = [word2id_de[w] for w in de if w in word2id_de]
    if not en_ids: continue
    leaf_votes = Counter([leaf_en[eid].item() for eid in en_ids])
    best_leaf = leaf_votes.most_common(1)[0][0]
    for did in de_ids:
        leaf_to_de[best_leaf][did] += 1

# per-leaf: top English word + top German word
leaf_top_en = {}
leaf_top_de = {}
active = 0
for lid in range(n_leaves):
    if leaf_to_de[lid]:
        active += 1
        # collect EN words in this leaf
        en_words = Counter()
        for de, en in train_pairs:
            for w in en:
                if w in word2id_en and leaf_en[word2id_en[w]].item() == lid:
                    en_words[word2id_en[w]] += 1
        if en_words:
            leaf_top_en[lid] = en_words.most_common(1)[0][0]
        leaf_top_de[lid] = leaf_to_de[lid].most_common(1)[0][0]

print(f"active leaves: {active}/{n_leaves}")

# Translate: en sentence → leaf → de word (reverse direction: de→en also)
# For each German sentence, route each word → find nearest EN leaf → output
refs, hyps = [], []
for de, en in val_pairs[:200]:
    de_ref = [word2id_en[w] for w in en if w in word2id_en]  # reference: English words
    # Translate German → English: for each de word → find similar en leaf → output en
    hyp = []
    for de_w in de:
        if de_w not in word2id_de: continue
        de_emb = E_de[word2id_de[de_w]]
        # nearest EN word's leaf
        scores = de_emb @ E_en.T  # (V_en,)
        best_en_id = scores.argmax().item()
        best_leaf = leaf_en[best_en_id].item()
        if best_leaf in leaf_top_en:
            hyp.append(leaf_top_en[best_leaf])
    refs.append(de_ref)
    hyps.append(hyp[:len(de_ref)])

# BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
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
print(f"BLEU-4 = {b:.2f}")

# samples
for i in [0, 2, 5]:
    de, en = val_pairs[i]
    hyp_words = []
    for w in de[:6]:
        if w not in word2id_de: continue
        de_emb = E_de[word2id_de[w]]
        scores = de_emb @ E_en.T
        best_en = scores.argmax().item()
        best_leaf = leaf_en[best_en].item()
        hyp_words.append(id2word_en.get(leaf_top_en.get(best_leaf, 0), '?'))
    print(f"\n  src(de): {' '.join(de[:6])}")
    print(f"  ref(en): {' '.join(en[:6])}")
    print(f"  hyp:     {' '.join(hyp_words[:6])}")
