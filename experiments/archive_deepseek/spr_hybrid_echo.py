"""
SPR Hybrid Echo — Per-token routing + Tree path context + GRU momentum
  - Per-token W_split routing → leaf_hash (word identity)
  - Syntactic distance tree → path[t] (syntactic position)  
  - GRU decoder: [leaf_hash[t] + path_emb[t] + prev_token] → token[t]
  Translation-ready: bridge maps leaf_hash_src→tgt, tree paths shared
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t", 1)[1].strip().lower().split())
    return sents

print("loading...")
train_sents = load_sents(train_file, 20000)
val_sents = load_sents(val_file, 500)

word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for s in train_sents:
    for w in s: freq[w] += 1
for w, c in freq.most_common():
    if c >= 2: word2id[w] = len(word2id)
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 128
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d} train={len(train_sents)} val={len(val_sents)}")

torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]

# ──── Encoder 1: Per-token W_split routing → leaf_hash ────
class PerTokenRouter(nn.Module):
    """Each token independently routed through learned W_split tree"""
    def __init__(self, V, d, depth=5, n_modules=4):
        super().__init__()
        self.V = V; self.d = d; self.depth = depth
        self.n_leaves = 1 << depth
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, 0, 0.02)
        
        # Per-depth splitter: token → scalar for left/right
        self.W_split = nn.ModuleList([
            nn.Linear(d, 1, bias=False) for _ in range(n_modules)
        ])
        for w in self.W_split:
            nn.init.normal_(w.weight, 0, 0.02)
        
        # Leaf prototypes
        self.E_leaf = nn.Parameter(torch.randn(self.n_leaves, d) * 0.02)
    
    def forward(self, ids):
        """Route tokens → leaf_hashes [T, d]"""
        T = len(ids)
        emb = self.E(ids)  # [T, d]
        
        leaf_probs = torch.ones(T, self.n_leaves, device=emb.device)
        for level in range(self.depth):
            idx = level % len(self.W_split)
            score = self.W_split[idx](emb).squeeze(-1)  # [T]
            left_prob = torch.sigmoid(score)
            right_prob = 1.0 - left_prob
            bit_pos = self.depth - 1 - level
            mask = ((torch.arange(self.n_leaves, device=emb.device) >> bit_pos) & 1).float().view(1, -1)
            level_prob = mask * right_prob.unsqueeze(-1) + (1.0 - mask) * left_prob.unsqueeze(-1)
            leaf_probs = leaf_probs * level_prob
        
        leaf_vec = leaf_probs @ self.E_leaf  # [T, n_leaves] @ [n_leaves, d] = [T, d]
        return leaf_vec


# ──── Encoder 2: Syntactic distance tree → per-token paths ────
def syntactic_distances(embs):
    T = embs.shape[0]
    dists = torch.zeros(T-1, device=device)
    current = embs.clone()
    for dp in range(1, 4):
        rolled = torch.roll(current, shifts=dp, dims=-1) * SIGN_MASK
        scores = (embs * rolled).sum(dim=-1)
        dists += (scores[:-1] - scores[1:]).abs()
    return dists / 3.0

def build_paths(embs, dists):
    T = len(embs)
    if T <= 1: return [''] if T >= 1 else []
    if len(dists) == 0: return ['']
    split_pos = dists.argmax().item(); L = split_pos + 1
    left_paths = build_paths(embs[:L], dists[:split_pos] if split_pos > 0 else dists[:0])
    right_paths = build_paths(embs[L:], dists[L:] if L < len(dists) else dists[:0])
    return ['L' + p for p in left_paths] + ['R' + p for p in right_paths]


# ──── Decoder: Hybrid GRU = leaf_hash + path + momentum ────
class HybridDecoder(nn.Module):
    """Predict token[t] from leaf_hash[t] + path_emb[t] + GRU momentum"""
    def __init__(self, d, V, max_path_len=32):
        super().__init__()
        self.d = d; self.V = V
        
        self.path_enc_L = nn.Embedding(max_path_len, d)
        self.path_enc_R = nn.Embedding(max_path_len, d)
        self.path_proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        
        self.gru = nn.GRUCell(d, d)
        self.W_out = nn.Linear(d, V)
    
    def encode_path(self, path):
        v = torch.zeros(self.d, device=self.path_enc_L.weight.device)
        for depth, c in enumerate(path):
            if depth >= self.path_enc_L.num_embeddings: break
            idx = torch.tensor(depth, device=v.device)
            v += self.path_enc_L(idx) if c == 'L' else self.path_enc_R(idx)
        return v / max(len(path), 1)
    
    def forward(self, leaf_hashes, paths, gold_ids):
        """Teacher forcing: leaf[t] + path[t] + prev_gold → logits[t]"""
        T = len(paths)
        h = torch.zeros(self.d, device=leaf_hashes.device)
        logits_list = []
        for t in range(T):
            path_vec = self.path_proj(self.encode_path(paths[t]))
            inp = leaf_hashes[t] + path_vec
            if t > 0:
                prev_emb = leaf_hashes[t-1]  # momentum via leaf hash, not raw token
                inp = inp + 0.3 * prev_emb
            h = self.gru(inp, h)
            logits_list.append(self.W_out(h))
        return torch.stack(logits_list, dim=0)


router = PerTokenRouter(V, d, depth=6, n_modules=4).to(device)
decoder = HybridDecoder(d, V).to(device)
opt = torch.optim.Adam(list(router.parameters()) + list(decoder.parameters()), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values())
            mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bp = min(2.0, math.exp(max(0, max(1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0))))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents[:200] if len(s) >= 2]

n_params = sum(p.numel() for p in list(router.parameters()) + list(decoder.parameters()))
print(f"params={n_params/1e6:.1f}M")

print(f"\n{'='*60}")
print("Training Hybrid Echo: per-token leaf + tree path + GRU momentum")
print(f"  epochs=100 batch=16 lr=0.003")
t0 = time.time()

for epoch in range(100):
    router.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    for bi in range(0, 3000, 16):
        batch_sents = train_sents[bi:bi+16]
        if not batch_sents: continue
        
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        n_sents = 0
        
        for s in batch_sents:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            
            ids_t = torch.tensor(ids, device=device)
            
            # Per-token routing
            leaf_hashes = router(ids_t)
            
            # Tree paths
            with torch.no_grad():
                dists = syntactic_distances(router.E(ids_t))
                paths = build_paths(router.E(ids_t), dists)
            
            if len(paths) != len(ids): continue
            
            # Decode with teacher forcing
            logits = decoder(leaf_hashes, paths, ids_t)
            loss = F.cross_entropy(logits, ids_t)
            batch_loss = batch_loss + loss
            n_sents += 1
        
        if n_sents == 0: continue
        batch_loss = batch_loss / n_sents
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(router.parameters()) + list(decoder.parameters()), 2.0)
        opt.step()
        
        ti += 1; tl += batch_loss.item(); tt += 1
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 99:
        router.eval(); decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                if len(ids) < 3: continue
                ids_t = torch.tensor(ids, device=device)
                leaf_hashes = router(ids_t)
                dists = syntactic_distances(router.E(ids_t))
                paths = build_paths(router.E(ids_t), dists)
                if len(paths) != len(ids): continue
                logits = decoder(leaf_hashes, paths, ids_t)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids); hyps.append(pred)
        
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        tok_tot = sum(len(r) for r in refs)
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} "
              f"tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% "
              f"time={time.time()-t0:.0f}s")
        router.train(); decoder.train()

# Final
router.eval(); decoder.eval(); refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        if len(ids) < 3: continue
        ids_t = torch.tensor(ids, device=device)
        leaf_hashes = router(ids_t)
        dists = syntactic_distances(router.E(ids_t))
        paths = build_paths(router.E(ids_t), dists)
        if len(paths) != len(ids): continue
        logits = decoder(leaf_hashes, paths, ids_t)
        pred = logits.argmax(dim=-1).cpu().tolist()
        refs.append(ids); hyps.append(pred)
print(f"\nFinal BLEU-4 = {compute_bleu(refs, hyps):.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device)
    leaf_hashes = router(ids_t)
    dists = syntactic_distances(router.E(ids_t))
    paths = build_paths(router.E(ids_t), dists)
    if len(paths) != len(ids): continue
    logits = decoder(leaf_hashes, paths, ids_t)
    pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:8])}")
    print(f"  hyp: {' '.join(pred[:8])}")
    print()
