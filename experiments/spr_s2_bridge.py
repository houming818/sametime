"""
SPR Session 2 — Translation Bridge (dictionary echo)
Architecture:
  DE: E_de + 4 templates → argmax root_hash_de
  Bridge: Residual MLP (Linear→GeLU→Linear→LayerNorm) → root_hash_en_pred
  LeafPredictor: small MLP maps (root_en, pos_emb) → leaf_hash → GRU decode
  EN: GRU autoregressive (from S1 echo) → English tokens

Training: 3-phase
  Phase 1: Freeze E_en, decoder_en. Train bridge, E_de, LeafPredictor (30 epochs)
  Phase 2: Unfreeze all, lr=1e-4, joint fine-tune (20 epochs)
  Phase 3: Eval — root_en_pred backwards-selects EN template, wild inference
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sys, os
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")
print("SPR S2 TRANSLATION BRIDGE — German→English")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data ────
def load_pairs(path, n):
    pairs = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: pairs.append(tuple(c.strip().lower().split() for c in l.split("\t")[:2]))
    return pairs

print("loading...")
MAX_TRAIN = 50000
train_pairs = load_pairs(train_file, MAX_TRAIN)
val_pairs = load_pairs(val_file, 500)
print(f"train={len(train_pairs)} val={len(val_pairs)}")

# Vocab: independent DE and EN word tables
word2id_de = {"<pad>": 0, "<unk>": 1}
word2id_en = {"<pad>": 0, "<unk>": 1}
freq_de, freq_en = Counter(), Counter()
for de, en in train_pairs:
    for w in de: freq_de[w] += 1
    for w in en: freq_en[w] += 1
for w, c in freq_de.most_common():
    if c >= 2: word2id_de[w] = len(word2id_de)
for w, c in freq_en.most_common():
    if c >= 2: word2id_en[w] = len(word2id_en)
for de, en in val_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de)
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en)

V_de, V_en, d = len(word2id_de), len(word2id_en), 128
id2word_de = {v:k for k,v in word2id_de.items()}
id2word_en = {v:k for k,v in word2id_en.items()}
print(f"vocab DE={V_de} EN={V_en} d={d}")

# ──── Fixed Templates (shared code) ────
torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]

def gen_paths(T, split_fn):
    def _gen(embs, _, depth=1):
        n = len(embs)
        if n <= 1: return [('', 0)] if n == 1 else []
        s = split_fn(n, depth); L = max(1, min(s, n-1))
        left = _gen(embs[:L], None, depth+1)
        right = _gen(embs[L:], None, depth+1)
        return [('L'+p, i) for p,i in left] + [('R'+p, i+L) for p,i in right]
    dummy = list(range(T))
    results = _gen(dummy, None)
    paths = [''] * T
    for p, idx in results: paths[idx] = p
    return paths

TEMPLATES = {
    'Left_Heavy':  lambda n, d: 1,
    'Right_Heavy': lambda n, d: max(1, n-1),
    'Balanced':    lambda n, d: max(1, n//2),
    'Spec_Head':   lambda n, d: 3 if n >= 6 else max(1, n//2),
}
TEMPLATE_NAMES = list(TEMPLATES.keys())
MAX_LEN = 50
template_paths = {t: {T: gen_paths(T, fn) for T in range(2, MAX_LEN+1)} for t, fn in TEMPLATES.items()}

def _build_tree(embs, paths):
    T = len(embs)
    if len(paths) != T: paths = [''] * T
    cur = {}
    for t in range(T):
        if t < len(paths):
            cur[('leaf', paths[t])] = embs[t]
    md = max(len(p) for p in paths) if paths else 0
    for depth in range(md, 0, -1):
        for pfx in set(p[:depth-1] if depth>1 else '' for p in paths if len(p)>=max(depth-1,0)):
            lk = ('leaf', pfx+'L') if depth>1 else ('leaf','L')
            rk = ('leaf', pfx+'R') if depth>1 else ('leaf','R')
            if lk in cur and rk in cur:
                lft = cur.pop(lk); rgt = cur.pop(rk)
                merged = lft + SIGN_MASK * torch.roll(rgt, shifts=depth)
                cur[('node', pfx)] = merged / (merged.norm()+1e-8)
    return next(iter(cur.values())).squeeze() if cur else torch.zeros(d, device=embs.device)

def compute_root_hash(E, ids, tname):
    T = len(ids)
    if T < 2: 
        idx = ids[0] if T>=1 else 0
        idx = min(idx, E.weight.shape[0]-1)
        return E(torch.tensor([idx], device=device))
    # Ensure all ids are valid
    ids = [min(i, E.weight.shape[0]-1) for i in ids]
    paths = template_paths.get(tname, {}).get(min(T, MAX_LEN))
    if paths is None or len(paths) != T:
        for k in [min(T, MAX_LEN), T]:
            try:
                paths = gen_paths(k, TEMPLATES[tname])
                if len(paths) == T: break
            except: 
                paths = gen_paths(max(2, k-1), TEMPLATES[tname])
                if len(paths) == T: break
    if not paths or len(paths) != T:
        paths = [''] * T  # fallback: flat leaf
    return _build_tree(E(torch.tensor(ids, device=device)), paths)


# ──── Modules ────
class Bridge(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(),
            nn.Linear(d*2, d), nn.LayerNorm(d)
        )
    def forward(self, root_de):
        return root_de + self.net(root_de)

class LeafPredictor(nn.Module):
    def __init__(self, d, max_len=100):
        super().__init__()
        self.pos_enc = nn.Embedding(max_len, d)
        self.net = nn.Sequential(nn.Linear(d*2, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, root_hash, T):
        T = min(T, self.pos_enc.weight.shape[0])
        pos_ids = torch.arange(T, device=root_hash.device)
        pos = self.pos_enc(pos_ids)
        root_exp = root_hash.unsqueeze(0).expand(T, -1)
        inp = torch.cat([root_exp, pos], dim=-1)
        return self.net(inp)

class GRUDecoder(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.d = d
        self.gru = nn.GRUCell(d, d)
        self.W_out = nn.Linear(d, V)
    def forward(self, leaf_hashes, gold_ids=None, p_teacher=1.0):
        T = leaf_hashes.shape[0]
        h = torch.zeros(d, device=leaf_hashes.device)
        logits = []
        prev_emb = torch.zeros(d, device=leaf_hashes.device)
        for t in range(T):
            inp = leaf_hashes[t]
            if t > 0:
                if gold_ids is not None and random.random() < p_teacher:
                    inp = inp + 0.3 * leaf_hashes[t-1]  # gold context leaf
                else:
                    inp = inp + 0.3 * prev_emb
            h = self.gru(inp, h)
            out = self.W_out(h)
            logits.append(out)
            # Next step's prev: use current leaf at t (or gold leaf for teacher forcing)
            with torch.no_grad():
                pid = out.argmax(dim=-1).item()
            idx = min(t, T-1) if gold_ids is not None else min(pid, T-1)
            prev_emb = leaf_hashes[idx]
        return torch.stack(logits, dim=0)

# ──── Init modules ────
E_de = nn.Embedding(V_de, d).to(device)
E_en = nn.Embedding(V_en, d).to(device)
bridge = Bridge(d).to(device)
leaf_pred = LeafPredictor(d).to(device)
decoder = GRUDecoder(V_en).to(device)

nn.init.normal_(E_de.weight, 0, 0.02)
nn.init.normal_(E_en.weight, 0, 0.02)

def get_pos_emb(T):
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = 10000 ** (torch.arange(0, d, 2, device=device).float()/d)
    phase = pos / div
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

def get_best_root(E, ids, verbose=False):
    ids = ids[:MAX_LEN]  # cap sentence length
    if len(ids) < 2: return E(torch.tensor([ids[0] if ids else 0], device=device))
    best_score, best_root, best_tpl = -1e9, None, None
    for tname in TEMPLATE_NAMES:
        root = compute_root_hash(E, ids, tname)
        sv = root.norm().item()
        if sv > best_score: best_score, best_root, best_tpl = sv, root, tname
    return best_root

# ──── BLEU ────
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(refs,hyps):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1) if ttl>0 else 1.0)
    bp = min(2.0, math.exp(max(0,max(1-len(r)/max(len(h),1) for r,h in zip(refs,hyps) if len(h)>0))))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

# ──── Prepare val data ────
val_de = [[word2id_de.get(w,1) for w in d] for d,e in val_pairs[:300] if len(d)>=2 and len(e)>=2]
val_en = [[word2id_en.get(w,1) for w in e] for d,e in val_pairs[:300] if len(d)>=2 and len(e)>=2]
print(f"val pairs: {len(val_de)}")

n_params = sum(p.numel() for p in list(E_de.parameters())+list(E_en.parameters())+list(bridge.parameters())+list(leaf_pred.parameters())+list(decoder.parameters()))
print(f"params={n_params/1e6:.1f}M")

# ════════════════════════════════════════════════
# PHASE 1: Freeze EN, train bridge + E_de + LeafPredictor
# ════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 1: Freeze EN, train bridge + E_de + LeafPredictor")
print(f"  epochs=30 batch=8 lr=0.003")
t0 = time.time()

for p in E_en.parameters(): p.requires_grad = False
for p in decoder.parameters(): p.requires_grad = False

opt1 = torch.optim.Adam(list(E_de.parameters()) + list(bridge.parameters()) + list(leaf_pred.parameters()), lr=0.003)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=30)

for epoch in range(30):
    random.shuffle(train_pairs)
    ti, tl = 0, 0
    p_t = max(0.2, 1.0 - epoch/20.0) if epoch > 3 else 1.0
    
    for bi in range(0, 5000, 8):
        batch = train_pairs[bi:bi+8]
        if not batch: continue
        opt1.zero_grad()
        b_loss = torch.tensor(0.0, device=device); n = 0
        
        for de, en in batch:
            ids_de = [word2id_de.get(w,1) for w in de][:MAX_LEN]
            ids_en = [word2id_en.get(w,1) for w in en][:MAX_LEN]
            if len(ids_de) < 3 or len(ids_en) < 3: continue
            
            with torch.no_grad():
                root_de = get_best_root(E_de, ids_de)
            root_en_pred = bridge(root_de)
            
            # English leaf_hashes (gold, for teacher forcing)
            ids_en_t = torch.tensor(ids_en, device=device)
            pos_emb = get_pos_emb(len(ids_en))
            leaf_en = E_en(ids_en_t) + 0.5 * pos_emb
            leaf_en = leaf_en / (leaf_en.norm(dim=-1, keepdim=True)+1e-8)
            
            # Also compute LeafPredictor output for training
            leaf_pred_en = leaf_pred(root_en_pred, len(ids_en))
            
            # Combine: use gold leaf for teacher forcing, predicted leaf as auxiliary
            combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
            logits = decoder(combined, ids_en_t, p_teacher=p_t)
            loss = F.cross_entropy(logits, ids_en_t)
            
            # Leaf predictor loss
            loss_leaf = F.mse_loss(leaf_pred_en, leaf_en.detach())
            loss = loss + 0.5 * loss_leaf
            
            b_loss += loss; n += 1
        
        if n == 0: continue
        (b_loss/n).backward()
        torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]['params'], 2.0)
        opt1.step()
        ti += 1; tl += (b_loss/n).item()
    
    if ti == 0: continue
    sched1.step()
    
    if epoch % 5 == 0 or epoch == 29:
        E_de.eval(); E_en.eval(); bridge.eval(); leaf_pred.eval(); decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for ids_de, ids_en in zip(val_de[:50], val_en[:50]):
                root_de = get_best_root(E_de, ids_de)
                root_en_pred = bridge(root_de)
                leaf_en = leaf_pred(root_en_pred, len(ids_en))
                combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
                logits = decoder(combined)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids_en); hyps.append(pred[:len(ids_en)])
        bleu = compute_bleu(refs, hyps)
        tok_acc = 100*sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in refs))
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% time={time.time()-t0:.0f}s")
        E_de.train(); bridge.train(); leaf_pred.train()
    E_en.train(); decoder.train()

# ════════════════════════════════════════════════
# PHASE 2: Unfreeze all, joint fine-tune
# ════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: Unfreeze all, joint fine-tune (lr=1e-4)")
print(f"  epochs=20 batch=8 lr=0.0001")

for p in E_en.parameters(): p.requires_grad = True
for p in decoder.parameters(): p.requires_grad = True

opt2 = torch.optim.Adam(list(E_de.parameters())+list(E_en.parameters())+list(bridge.parameters())+list(leaf_pred.parameters())+list(decoder.parameters()), lr=1e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20)

for epoch in range(20):
    random.shuffle(train_pairs)
    ti, tl = 0, 0
    p_t = max(0.2, 1.0 - (epoch+30)/30.0) if epoch > 2 else 1.0
    
    for bi in range(0, 3000, 8):
        batch = train_pairs[bi:bi+8]
        if not batch: continue
        opt2.zero_grad()
        b_loss = torch.tensor(0.0, device=device); n = 0
        
        for de, en in batch:
            ids_de = [word2id_de.get(w,1) for w in de][:MAX_LEN]; ids_en = [word2id_en.get(w,1) for w in en][:MAX_LEN]
            if len(ids_de) < 3 or len(ids_en) < 3: continue
            
            with torch.no_grad():
                root_de = get_best_root(E_de, ids_de)
            root_en_pred = bridge(root_de)
            
            ids_en_t = torch.tensor(ids_en, device=device)
            pos_emb = get_pos_emb(len(ids_en))
            leaf_en = E_en(ids_en_t) + 0.5 * pos_emb
            leaf_en = leaf_en / (leaf_en.norm(dim=-1, keepdim=True)+1e-8)
            
            leaf_pred_en = leaf_pred(root_en_pred, len(ids_en))
            combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
            logits = decoder(combined, ids_en_t, p_teacher=p_t)
            loss = F.cross_entropy(logits, ids_en_t)
            loss_leaf = F.mse_loss(leaf_pred_en, leaf_en.detach())
            loss = loss + 0.5 * loss_leaf
            
            b_loss += loss; n += 1
        
        if n == 0: continue
        (b_loss/n).backward()
        torch.nn.utils.clip_grad_norm_(opt2.param_groups[0]['params'], 2.0)
        opt2.step()
        ti += 1; tl += (b_loss/n).item()
    
    if ti == 0: continue
    sched2.step()
    
    if epoch % 5 == 0 or epoch == 19:
        E_de.eval(); E_en.eval(); bridge.eval(); leaf_pred.eval(); decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for ids_de, ids_en in zip(val_de, val_en):
                root_de = get_best_root(E_de, ids_de)
                root_en_pred = bridge(root_de)
                leaf_en = leaf_pred(root_en_pred, len(ids_en))
                combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
                logits = decoder(combined)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids_en); hyps.append(pred[:len(ids_en)])
        bleu = compute_bleu(refs, hyps)
        tok_acc = 100*sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in refs))
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% time={time.time()-t0:.0f}s")
        E_de.train(); E_en.train(); bridge.train(); leaf_pred.train(); decoder.train()

# ════════════════════════════════════════════════
# PHASE 3: Final eval + samples
# ════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 3: Final Evaluation (wild inference)")
print(f"{'='*60}")

E_de.eval(); E_en.eval(); bridge.eval(); leaf_pred.eval(); decoder.eval()
refs, hyps = [], []
with torch.no_grad():
    for ids_de, ids_en in zip(val_de, val_en):
        root_de = get_best_root(E_de, ids_de)
        root_en_pred = bridge(root_de)
        T = len(ids_en)
        leaf_en = leaf_pred(root_en_pred, T)
        combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
        logits = decoder(combined)
        pred = logits.argmax(dim=-1).cpu().tolist()
        refs.append(ids_en); hyps.append(pred[:T])

bleu = compute_bleu(refs, hyps)
tok_acc = 100*sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in refs))
print(f"\nFINAL: BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%")
print(f"Total time: {time.time()-t0:.0f}s")

print(f"\n=== translation samples ===")
for i in range(5):
    de, en = val_pairs[i]
    ids_de = [word2id_de.get(w,1) for w in de]
    ids_en = [word2id_en.get(w,1) for w in en]
    if len(ids_de)<3 or len(ids_en)<3: continue
    with torch.no_grad():
        root_de = get_best_root(E_de, ids_de)
        root_en_pred = bridge(root_de)
        T = len(ids_en)
        leaf_en = leaf_pred(root_en_pred, T)
        combined = leaf_en + 0.1 * root_en_pred.unsqueeze(0).expand_as(leaf_en)
        logits = decoder(combined)
        pred = [id2word_en.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  DE: {' '.join(de[:8])}")
    print(f"  EN: {' '.join(en[:8])}")
    print(f"  PR: {' '.join(pred[:8])}")
    print()

torch.save({'E_de':E_de.state_dict(),'E_en':E_en.state_dict(),'bridge':bridge.state_dict(),'leaf_pred':leaf_pred.state_dict(),'decoder':decoder.state_dict(),'word2id_de':word2id_de,'word2id_en':word2id_en},'/tmp/spr_s2_bridge.pt')
print(f"Checkpoint: /tmp/spr_s2_bridge.pt")
