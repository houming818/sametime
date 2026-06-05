"""
SPR Translation v4 — Fixed: decoder math, <pad>, target alignment
50K sentences, min_freq=3, depth=6, rapid test
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os, time
from collections import Counter

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load(path, n):
    pairs = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: pairs.append(l.strip().split("\t", 1))
    return [(d.split(), e.split()) for d, e in pairs]

print("loading...")
train_pairs = load(train_file, 50000)
val_pairs = load(val_file, 300)
print(f"train={len(train_pairs)} val={len(val_pairs)}")

# Vocabulary: <pad>=0 + frequency filter
freq_de, freq_en = Counter(), Counter()
for de, en in train_pairs:
    for w in de: freq_de[w] += 1
    for w in en: freq_en[w] += 1
min_freq = 3
word2id_de, id2word_de = {"<pad>": 0}, {0: "<pad>"}
word2id_en, id2word_en = {"<pad>": 0}, {0: "<pad>"}
for w, c in freq_de.most_common():
    if c >= min_freq: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w
for w, c in freq_en.most_common():
    if c >= min_freq: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w

V_de, V_en, d = len(word2id_de), len(word2id_en), 128
depth, n_leaves = 4, 1<<4  # shallower for faster test
pad_len = n_leaves
print(f"de={V_de} en={V_en} d={d} depth={depth} leaves={n_leaves}")

# co-occ on GPU (5K sentence subset for speed)
print("building co-occ...")
t0 = time.time()
coocc = torch.zeros(V_en + V_de, d, device='cuda')
for di, (de, en) in enumerate(train_pairs[:5000]):
    for i, w in enumerate(en):
        if w not in word2id_en: continue
        wid = word2id_en[w]
        for j in range(max(0,i-3), min(len(en), i+4)):
            if i != j and en[j] in word2id_en: coocc[wid, word2id_en[en[j]] % d] += 1
    for i, w in enumerate(de):
        if w not in word2id_de: continue
        for j in range(max(0,i-3), min(len(de), i+4)):
            if i != j and de[j] in word2id_de: coocc[V_en + word2id_de[w], (V_en + word2id_de[de[j]]) % d] += 1
    for de_w in de:
        if de_w not in word2id_de: continue
        for en_w in en:
            if en_w not in word2id_en: continue
            coocc[V_en + word2id_de[de_w], word2id_en[en_w] % d] += 1
            coocc[word2id_en[en_w], (V_en + word2id_de[de_w]) % d] += 1
norms = coocc.norm(dim=1, keepdim=True) + 1e-8
E_all = coocc / norms
E_de = E_all[V_en:]
E_en = E_all[:V_en]
print(f"co-occ built ({time.time()-t0:.0f}s)")

# Trainable embeddings
E_de_train = nn.Parameter(E_de.clone())
E_en_train = nn.Parameter(E_en.clone())

# Encoder
def sign_alt(x):
    mask = torch.tensor([1., -1.] * (d//2 + 1), device=x.device)[:d]
    return x * mask

def encode(tokens, emb, d=0):
    if len(tokens) <= 1:
        return emb[tokens[0]] if len(tokens) == 1 else torch.zeros(emb.shape[1], device=emb.device)
    mid = len(tokens) // 2
    HL = encode(tokens[:mid], emb, d+1)
    HR = encode(tokens[mid:], emb, d+1)
    return HL + sign_alt(torch.roll(HR, shifts=d+1, dims=-1))

def get_leaves(tokens, emb, d=0):
    if d >= depth or len(tokens) == 0:
        return [emb[tokens[0]] if len(tokens) > 0 else torch.zeros(emb.shape[1], device=emb.device)]
    mid = len(tokens) // 2
    return get_leaves(tokens[:mid], emb, d+1) + get_leaves(tokens[mid:], emb, d+1)

# Decoder (with fix: sign_alt BEFORE roll)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_split = nn.ParameterList([
            nn.Parameter(torch.randn(d, d).cuda() * 0.02) for _ in range(depth)
        ])
    def forward(self, H, level=0):
        if level >= depth: return [H]
        HL = H @ self.W_split[level]
        HR = torch.roll(sign_alt(H - HL), shifts=-(level + 1), dims=-1)
        return self.forward(HL, level + 1) + self.forward(HR, level + 1)

decoder = Decoder().cuda()
bridge = nn.Linear(d, d).cuda()
opt = torch.optim.Adam(list(decoder.parameters()) + list(bridge.parameters()) + [E_de_train, E_en_train], lr=0.01)

def pad(ids, n):
    ids = ids[:n]
    return ids + [0] * (n - len(ids))

# Pre-encode val
print("pre-encoding validation...")
val_enc = []
for de, en in val_pairs[:100]:
    di = pad([word2id_de.get(w, 0) for w in de], pad_len)
    ei = pad([word2id_en.get(w, 0) for w in en], pad_len)
    with torch.no_grad():
        H_src = encode(di, E_de_train)
    val_enc.append((di, ei, H_src, de, en))

# Train
print(f"training 2000 steps, batch=64...")
t0 = time.time()
for step in range(2000):
    opt.zero_grad()
    loss = torch.tensor(0.0, device='cuda')
    n_tokens = 0
    
    batch = np.random.choice(len(train_pairs), 64, replace=False)
    for i in batch:
        de, en = train_pairs[i]
        di = pad([word2id_de.get(w, 0) for w in de], pad_len)
        ei = pad([word2id_en.get(w, 0) for w in en], pad_len)
        
        H_src = encode(di, E_de_train)
        H_tgt = bridge(H_src)
        pred_leaves = decoder.forward(H_tgt)
        tgt_leaves = get_leaves(ei, E_en_train)  # tree-order alignment
        
        loss += sum(F.mse_loss(pl, tl) for pl, tl in zip(pred_leaves, tgt_leaves))
        n_tokens += len(pred_leaves)
    
    loss = loss / max(n_tokens, 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0) if hasattr(opt, 'param_groups') else None
    opt.step()
    
    if step % 200 == 0:
        with torch.no_grad():
            di, ei, _, de, en = val_enc[0]
            H_pred = bridge(encode(di, E_de_train))
            leaves = decoder.forward(H_pred)
            pred = [id2word_en.get((lh @ E_en_train.T).argmax().item(), '?') for lh in leaves]
            pred = [p for p in pred if p not in ('<pad>', '?')]  # skip pad
        print(f"  step {step:4d}: loss={loss.item():.4f}  "
              f"pred={' '.join(pred[:5])}  ref={' '.join([w for w in en[:5] if w in word2id_en])}  "
              f"time={time.time()-t0:.0f}s")

# BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
refs, hyps = [], []
with torch.no_grad():
    for di, ei, _, de, en in val_enc[:50]:
        H_pred = bridge(encode(di, E_de_train))
        leaves = decoder.forward(H_pred)
        hyp_ids = [int((lh @ E_en_train.T).argmax().item()) for lh in leaves]
        ref_ids = [word2id_en.get(w, 0) for w in en if w in word2id_en]
        if ref_ids:
            refs.append(ref_ids)
            hyps.append(hyp_ids[:len(ref_ids)])

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
print(f"\nBLEU-4 = {b:.2f}  total_time={time.time()-t0:.0f}s")

# samples
print(f"\n=== samples ===")
with torch.no_grad():
    for i in [0, 1, 2]:
        di, ei, _, de, en = val_enc[i]
        H_pred = bridge(encode(di, E_de_train))
        leaves = decoder.forward(H_pred)
        pred = [id2word_en.get((lh @ E_en_train.T).argmax().item(), '?') for lh in leaves]
        pred = [p for p in pred if p != '<pad>']
        print(f"  src: {' '.join(de[:6])}")
        print(f"  ref: {' '.join(en[:6])}")
        print(f"  hyp: {' '.join(pred[:6])}")
        print()
