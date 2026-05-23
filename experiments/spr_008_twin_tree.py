"""
SPR Translation v3 — WMT14 50K, bilingual tree, GPU training
Encoder (cyclic shift) + Bridge + Trainable Decoder → BLEU
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os
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

print("loading data...")
train_pairs = load_pairs(train_file, 50000)
val_pairs = load_pairs(val_file, 500)
print(f"train={len(train_pairs)} val={len(val_pairs)}")

# bilingual vocabulary
word2id_de, id2word_de = {}, {}
word2id_en, id2word_en = {}, {}
for de, en in train_pairs + val_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w

V_de, V_en, d = len(word2id_de), len(word2id_en), 128
print(f"de={V_de} en={V_en} d={d}")

# bilingual co-occ embeddings
print("building bilingual co-occ...")
coocc = np.zeros((V_en + V_de, d), dtype=np.float32)
for di, (de, en) in enumerate(train_pairs):
    for i, w in enumerate(en):
        wid = word2id_en[w]
        for j in range(max(0,i-3), min(len(en), i+4)):
            if i != j and en[j] in word2id_en:
                coocc[wid, word2id_en[en[j]] % d] += 1
    for i, w in enumerate(de):
        wid = V_en + word2id_de[w]
        for j in range(max(0,i-3), min(len(de), i+4)):
            if i != j and de[j] in word2id_de:
                coocc[wid, (V_en + word2id_de[de[j]]) % d] += 1
    for de_w in de:
        for en_w in en:
            coocc[V_en + word2id_de[de_w], word2id_en[en_w] % d] += 1
            coocc[word2id_en[en_w], (V_en + word2id_de[de_w]) % d] += 1
    if di % 10000 == 9999: print(f"  processed {di+1} sentences")

norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E_all = torch.tensor(coocc / norms, dtype=torch.float32).cuda()
E_de = E_all[V_en:]
E_en = E_all[:V_en]

# ── trainable modules ──
depth = 3  # pad to 8 tokens
n_leaves = 1 << depth

class Encoder:
    @staticmethod
    def sign_alt(x):
        mask = torch.tensor([1., -1.] * (d//2 + 1), device=x.device)[:d]
        return x * mask
    
    def encode(self, tokens, emb, depth=0):
        if len(tokens) <= 1:
            return emb[tokens[0]] if len(tokens) == 1 else torch.zeros(d, device=emb.device)
        mid = len(tokens) // 2
        HL = self.encode(tokens[:mid], emb, depth + 1)
        HR = self.encode(tokens[mid:], emb, depth + 1)
        return HL + self.sign_alt(torch.roll(HR, shifts=depth + 1, dims=-1))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_split = nn.ParameterList([
            nn.Parameter(torch.randn(d, d).cuda() * 0.1) for _ in range(depth)
        ])
    
    def sign_alt(self, x):
        mask = torch.tensor([1., -1.] * (d//2 + 1), device=x.device)[:d]
        return x * mask
    
    def forward(self, H, level=0):
        if level >= depth:
            return [H]  # leaf
        HL = H @ self.W_split[level]
        residual = H - HL
        HR = self.sign_alt(residual)
        HR = torch.roll(HR, shifts=-(level + 1), dims=-1)
        left = self.forward(HL, level + 1)
        right = self.forward(HR, level + 1)
        return left + right

encoder = Encoder()
decoder = Decoder().cuda()
bridge = nn.Linear(d, d).cuda()
opt = torch.optim.Adam(list(decoder.parameters()) + list(bridge.parameters()), lr=0.01)

# pad sentences to 8 tokens
def pad_ids(ids):
    ids = ids[:8]
    while len(ids) < 8: ids.append(0)
    return ids

print(f"\ntraining (samples={len(train_pairs[:2000])})...")
for step in range(500):
    opt.zero_grad()
    loss = torch.tensor(0.0, device='cuda')
    n_tokens = 0
    
    # batch: random subset of training pairs
    batch_idx = np.random.choice(min(2000, len(train_pairs)), 16, replace=False)
    for i in batch_idx:
        de, en = train_pairs[i]
        de_ids = pad_ids([word2id_de[w] for w in de if w in word2id_de])
        en_ids = pad_ids([word2id_en[w] for w in en if w in word2id_en])
        
        # Encode DE → root hash
        H_src = encoder.encode(de_ids, E_de)
        
        # Bridge: DE space → EN space
        H_tgt = bridge(H_src)
        
        # Decode → leaf hashes
        leaf_preds = decoder.forward(H_tgt)
        
        # Target: encode EN sentence → leaf hashes
        tgt_emb = E_en[en_ids]
        for i, (lh, th) in enumerate(zip(leaf_preds, tgt_emb)):
            loss += ((lh - th) ** 2).mean()
            n_tokens += 1
    
    loss = loss / max(n_tokens, 1)
    loss.backward()
    opt.step()
    
    if step % 100 == 0:
        # quick BLEU check on a validation sentence
        with torch.no_grad():
            for de, en in val_pairs[:1]:
                de_ids = pad_ids([word2id_de[w] for w in de[:8] if w in word2id_de])
                H_src = encoder.encode(de_ids, E_de)
                H_tgt = bridge(H_src)
                leaves = decoder.forward(H_tgt)
                pred_words = []
                for lh in leaves:
                    scores = lh @ E_en.T
                    pred_words.append(id2word_en.get(scores.argmax().item(), '?'))
                ref_words = en[:8]
        print(f"  step {step:3d}: loss={loss.item():.4f}  "
              f"pred={' '.join(pred_words[:4])}  ref={' '.join(ref_words[:4])}")

# ── BLEU on validation ──
print(f"\n=== BLEU evaluation ===")
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
refs, hyps = [], []
with torch.no_grad():
    for de, en in val_pairs[:100]:
        de_ids = pad_ids([word2id_de[w] for w in de[:8] if w in word2id_de])
        H_src = encoder.encode(de_ids, E_de)
        H_tgt = bridge(H_src)
        leaves = decoder.forward(H_tgt)
        hyp_ids = []
        for lh in leaves:
            scores = lh @ E_en.T
            hyp_ids.append(scores.argmax().item())
        ref_ids = [word2id_en[w] for w in en if w in word2id_en]
        if ref_ids:
            refs.append(ref_ids[:8])
            hyps.append(hyp_ids[:len(ref_ids[:8])])

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

print(f"\n=== samples ===")
with torch.no_grad():
    for i in [0, 1, 2, 5]:
        de, en = val_pairs[i]
        de_ids = pad_ids([word2id_de[w] for w in de[:8] if w in word2id_de])
        H_src = encoder.encode(de_ids, E_de)
        H_tgt = bridge(H_src)
        leaves = decoder.forward(H_tgt)
        pred = []
        for lh in leaves:
            scores = lh @ E_en.T
            pred.append(id2word_en.get(scores.argmax().item(), '?'))
        print(f"  src(de): {' '.join(de[:6])}")
        print(f"  ref(en): {' '.join(en[:6])}")
        print(f"  pred:    {' '.join(pred[:6])}")
        print()
