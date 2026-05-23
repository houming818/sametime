"""
SPR Translation — Production Scale
100K WMT14, depth=6, d=128, trainable E + Bridge + Decoder
Maximum GPU throughput
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os, time
from collections import Counter

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

print("=== SPR Translation — Production Mode ===")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
t_start = time.time()

# Load 100K training + 500 val
def load_pairs(path, n):
    pairs = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l:
                de, en = l.strip().split("\t", 1)
                pairs.append((de.split(), en.split()))
    return pairs

print("loading 100K sentences...")
train_pairs = load_pairs(train_file, 100000)
val_pairs = load_pairs(val_file, 500)
print(f"train={len(train_pairs)} val={len(val_pairs)} ({time.time()-t_start:.0f}s)")

# Build vocabulary (frequency-filtered)
min_freq = 3
freq_de, freq_en = Counter(), Counter()
for de, en in train_pairs:
    for w in de: freq_de[w] += 1
    for w in en: freq_en[w] += 1

word2id_de, id2word_de = {"<pad>": 0}, {0: "<pad>"}
word2id_en, id2word_en = {"<pad>": 0}, {0: "<pad>"}
for de, en in train_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de); id2word_de[len(id2word_de)] = w
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en); id2word_en[len(id2word_en)] = w

V_de, V_en, d = len(word2id_de), len(word2id_en), 128
depth, n_leaves = 6, 1 << 6
pad_len = n_leaves
print(f"de={V_de} en={V_en} d={d} depth={depth} leaves={n_leaves}")

# Bilingual co-occ embeddings (on GPU)
print("building co-occ...")
coocc = torch.zeros(V_en + V_de, d, device='cuda')
for di, (de, en) in enumerate(train_pairs):
    for i, w in enumerate(en):
        if w not in word2id_en: continue
        wid = word2id_en[w]
        for j in range(max(0,i-3), min(len(en), i+4)):
            if i != j and en[j] in word2id_en:
                coocc[wid, word2id_en[en[j]] % d] += 1
    for i, w in enumerate(de):
        if w not in word2id_de: continue
        wid = V_en + word2id_de[w]
        for j in range(max(0,i-3), min(len(de), i+4)):
            if i != j and de[j] in word2id_de:
                coocc[wid, (V_en + word2id_de[de[j]]) % d] += 1
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
print(f"co-occ built ({time.time()-t_start:.0f}s)")

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
    HL = encode(tokens[:mid], emb, d + 1)
    HR = encode(tokens[mid:], emb, d + 1)
    return HL + sign_alt(torch.roll(HR, shifts=d + 1, dims=-1))

def get_leaves(tokens, emb, d=0):
    """Tree-order leaf vectors for target alignment"""
    if d >= depth or len(tokens) == 0:
        return [emb[tokens[0]] if len(tokens) > 0 else torch.zeros(emb.shape[1], device=emb.device)]
    mid = len(tokens) // 2
    return get_leaves(tokens[:mid], emb, d + 1) + get_leaves(tokens[mid:], emb, d + 1)

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_split = nn.ParameterList([nn.Parameter(torch.randn(d, d).cuda() * 0.05) for _ in range(depth)])
    def forward(self, H, level=0):
        if level >= depth: return [H]
        HL = H @ self.W_split[level]
        # FIX: sign_alt first, THEN unroll — match encoder order
        HR = torch.roll(self.sign_alt(H - HL), shifts=-(level + 1), dims=-1)
        return self.forward(HL, level + 1) + self.forward(HR, level + 1)

decoder = Decoder().cuda()
bridge = nn.Linear(d, d).cuda()

params = list(decoder.parameters()) + list(bridge.parameters()) + [E_de_train, E_en_train]
opt = torch.optim.Adam(params, lr=0.02)

# Pre-encode sentences for speed
def pad_ids(ids, n):
    ids = ids[:n]
    return ids + [0] * (n - len(ids))

print("pre-encoding validation sentences...")
val_encoded = []
for de, en in val_pairs[:200]:
    de_ids = pad_ids([word2id_de[w] for w in de if w in word2id_de], pad_len)
    en_ids = pad_ids([word2id_en[w] for w in en if w in word2id_en], pad_len)
    with torch.no_grad():
        H_src = encode(de_ids, E_de_train.detach())
        H_tgt = encode(en_ids, E_en_train.detach())
    val_encoded.append((de_ids, en_ids, H_src, H_tgt, de, en))

# Training
print(f"\ntraining 3000 steps, batch=64...")
best_loss = float('inf')
for step in range(3000):
    opt.zero_grad()
    loss = torch.tensor(0.0, device='cuda')
    n_tokens = 0
    
    batch_idx = np.random.choice(len(train_pairs), 64, replace=False)
    for i in batch_idx:
        de, en = train_pairs[i]
        de_ids = pad_ids([word2id_de[w] for w in de if w in word2id_de], pad_len)
        en_ids = pad_ids([word2id_en[w] for w in en if w in word2id_en], pad_len)
        
        H_src = encode(de_ids, E_de_train)
        H_tgt = bridge(H_src)
        leaf_preds = decoder.forward(H_tgt)
        # FIX: tree-order targets from get_leaves, not linear E[en_ids]
        tgt_leaves = get_leaves(en_ids, E_en_train)
        loss += sum(((lh - th) ** 2).mean() for lh, th in zip(leaf_preds, tgt_leaves))
        n_tokens += len(leaf_preds)
    
    loss = loss / max(n_tokens, 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    
    if step % 200 == 0:
        elapsed = time.time() - t_start
        with torch.no_grad():
            # validation sample
            de_ids, en_ids, H_src, H_tgt, de, en = val_encoded[0]
            H_pred = bridge(encode(de_ids, E_de_train))
            leaves = decoder.forward(H_pred)
            pred = [id2word_en.get((lh @ E_en_train.T).argmax().item(), '?') for lh in leaves]
        print(f"  step {step:4d}: loss={loss.item():.4f}  "
              f"pred={' '.join(pred[:5])}  ref={' '.join(en[:5])}  "
              f"time={elapsed:.0f}s")

# Final validation
print(f"\n=== Validation BLEU ===")
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
refs, hyps = [], []
with torch.no_grad():
    for de_ids, en_ids, _, _, de, en in val_encoded[:100]:
        H_pred = bridge(encode(de_ids, E_de_train))
        leaves = decoder.forward(H_pred)
        hyp_ids = [(lh @ E_en_train.T).argmax().item() for lh in leaves]
        ref_ids = [word2id_en[w] for w in en if w in word2id_en]
        if ref_ids:
            refs.append(ref_ids[:pad_len])
            hyps.append(hyp_ids[:len(ref_ids[:pad_len])])

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
total_t = time.time() - t_start
print(f"BLEU-4 = {b:.2f}  total_time={total_t:.0f}s ({total_t/60:.1f}min)")
print(f"de={V_de} en={V_en} d={d} depth={depth} leaves={n_leaves}")
print(f"trainable_params: decoder(6×{d}²={6*d*d}) + bridge({d}²={d*d}) + E({(V_de+V_en)*d})")
print(f"               = {6*d*d + d*d + (V_de+V_en)*d:,} total")

# Best sample
print(f"\n=== samples ===")
with torch.no_grad():
    for i in [0, 1, 2]:
        de_ids, en_ids, _, _, de, en = val_encoded[i]
        H_pred = bridge(encode(de_ids, E_de_train))
        leaves = decoder.forward(H_pred)
        pred = [id2word_en.get((lh @ E_en_train.T).argmax().item(), '?') for lh in leaves]
        print(f"  src: {' '.join(de[:6])}")
        print(f"  ref: {' '.join(en[:6])}")
        print(f"  hyp: {' '.join(pred[:6])}")
        print()
