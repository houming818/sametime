"""
SPR Fixed Topology Templates — Grammar as deterministic finite automata
No distance computation, no tree learning. Just K fixed binary tree templates.
For each sentence: try all templates, pick the one with max geometric resonance.
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
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)

# ──── Fixed Topology Templates ────
def gen_paths_for_template(T, split_fn):
    """Generate per-token paths [L,R]* for a given split heuristic"""
    def _gen(embs, dists_hint, depth=1):
        n = len(embs)
        if n <= 1:
            if n == 1: return [('', 0)]  # (path, original_index)
            return []
        split = split_fn(n, depth)
        L = max(1, min(split, n-1))
        left = _gen(embs[:L], None, depth+1)
        right = _gen(embs[L:], None, depth+1)
        return [('L'+p, i) for p,i in left] + [('R'+p, i+L) for p,i in right]
    
    # Create dummy embs for path gen
    dummy = list(range(T))
    results = _gen(dummy, None)
    paths = [''] * T
    for p, idx in results:
        paths[idx] = p
    return paths

# 4 fundamental grammar templates
TEMPLATES = {
    'Left_Heavy':  lambda n, d: 1,                    # split off 1 token on left
    'Right_Heavy': lambda n, d: max(1, n-1),           # split off 1 token on right
    'Balanced':    lambda n, d: max(1, n // 2),         # balanced binary split
    'Spec_Head':   lambda n, d: 3 if n >= 6 else max(1, n//2),  # subject-predicate split
}

# Pre-compute path templates for common lengths
MAX_LEN = 50
template_paths = {}  # template_name → {T: [paths]}
for name, fn in TEMPLATES.items():
    template_paths[name] = {}
    for T in range(2, MAX_LEN+1):
        template_paths[name][T] = gen_paths_for_template(T, fn)

print(f"Templates: {list(TEMPLATES.keys())}, max_len={MAX_LEN}")


# ──── Template-based Root Hash ────
def compute_root_hash(ids, template_name):
    """Route sentence through fixed template → root hash (pure geometry)"""
    T = len(ids)
    if T < 2: return E(torch.tensor([ids[0] if T>=1 else 0], device=device))
    
    key_T = min(T, MAX_LEN)
    paths = template_paths[template_name][key_T]
    if len(paths) != T:
        key_T = max(2, min(T-1, MAX_LEN))
        paths = gen_paths_for_template(key_T, TEMPLATES[template_name])
        if len(paths) != T:
            key_T = min(T, MAX_LEN)
            paths = gen_paths_for_template(key_T, TEMPLATES[template_name])
    if len(paths) != T:
        return E(torch.tensor([ids[0]], device=device))
    
    embs = E(torch.tensor(ids, device=device))
    return _build_tree_from_paths(embs, paths)


def _build_tree_from_paths(embs, paths):
    """Bottom-up merge using fixed paths — one hash per node"""
    T = len(embs)
    # Start with token embeddings
    current = dict()
    for t in range(T):
        current[('leaf', paths[t])] = embs[t]
    
    # Merge from deepest to shallowest
    max_depth = max(len(p) for p in paths) if paths else 0
    
    for depth in range(max_depth, 0, -1):
        # Find pairs of L/R at this depth
        prefixes = set()
        for p in paths:
            if len(p) >= depth-1:
                prefixes.add(p[:depth-1] if depth > 1 else '')
        
        for prefix in prefixes:
            left_key = ('leaf', prefix + 'L') if depth > 1 else ('leaf', 'L')
            right_key = ('leaf', prefix + 'R') if depth > 1 else ('leaf', 'R')
            
            if left_key in current and right_key in current:
                left = current.pop(left_key)
                right = current.pop(right_key)
                merged = left + SIGN_MASK * torch.roll(right, shifts=depth)
                merged = merged / (merged.norm() + 1e-8)
                current[('node', prefix)] = merged
    
    # Root hash is the last remaining
    root = next(iter(current.values())) if current else torch.zeros(d, device=embs.device)
    return root.squeeze()


# ──── Decoder: per-token routing + GRU momentum + Scheduled Sampling ────
class PerTokenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(d, d)
        self.W_out = nn.Linear(d, V)
    
    def forward(self, leaf_hashes, gold_ids=None, p_teacher=1.0):
        T = len(leaf_hashes)
        h = torch.zeros(d, device=leaf_hashes.device)
        logits_list = []
        prev_pred_emb = torch.zeros(d, device=leaf_hashes.device)
        for t in range(T):
            inp = leaf_hashes[t]
            if t > 0:
                # Scheduled Sampling: teacher or self-prediction?
                if gold_ids is not None and random.random() < p_teacher:
                    runtime_prev = E.weight[gold_ids[t-1]]
                else:
                    runtime_prev = prev_pred_emb
                inp = inp + 0.3 * runtime_prev
            h = self.gru(inp, h)
            logits = self.W_out(h)
            logits_list.append(logits)
            # Record model's own prediction for next step (no grad on this branch)
            with torch.no_grad():
                pred_id = logits.argmax(dim=-1)
            prev_pred_emb = E.weight[pred_id].detach()
        return torch.stack(logits_list, dim=0)


decoder = PerTokenDecoder().to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
print(f"params={sum(p.numel() for p in opt.param_groups[0]['params'])/1e6:.1f}M")

# ──── Sinusoidal position encoding ────
def get_pos_emb(T):
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = 10000 ** (torch.arange(0, d, 2, device=device).float() / d)
    phase = pos / div
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

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

print(f"\n{'='*60}")
print(f"Training Fixed-Template Echo: per-token routing + template root hash context")
print(f"  epochs=100 batch=16 lr=0.003")
t0 = time.time()

for epoch in range(100):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    # Scheduled Sampling annealing: 1.0 → 0.2 over epochs
    p_teacher = max(0.2, 1.0 - epoch / 40.0) if epoch > 5 else 1.0
    
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
            T = len(ids)
            
            # Per-token leaf hashes: E[word] + position_emb
            pos_emb = get_pos_emb(T)
            leaf_hashes = E(ids_t) + 0.5 * pos_emb
            leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Try all templates, pick best by geometric score (grad kept for winner only)
            best_score_val = -1e9
            best_root = None
            for tname in TEMPLATES:
                root = compute_root_hash(ids, tname)
                score_val = root.norm().item()  # float compare, detach losers
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_root = root
            
            # Use root hash as context for decoder
            root_context = best_root.reshape(-1).unsqueeze(0).expand(T, -1)
            combined = leaf_hashes + 0.1 * root_context
            
            logits = decoder(combined, ids_t, p_teacher=p_teacher)
            loss = F.cross_entropy(logits, ids_t)
            batch_loss = batch_loss + loss
            n_sents += 1
        
        if n_sents == 0: continue
        batch_loss = batch_loss / n_sents
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(E.parameters()) + list(decoder.parameters()), 2.0)
        opt.step()
        
        ti += 1; tl += batch_loss.item(); tt += 1
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 99:
        E.eval(); decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                if len(ids) < 3: continue
                ids_t = torch.tensor(ids, device=device)
                T = len(ids)
                
                pos_emb = get_pos_emb(T)
                leaf_hashes = E(ids_t) + 0.5 * pos_emb
                leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
                
                best_score = -1
                best_root = None
                for tname in TEMPLATES:
                    root = compute_root_hash(ids, tname)
                    score = root.norm().item()
                    if score > best_score: best_score, best_root = score, root
                
                combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
                logits = decoder(combined)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids); hyps.append(pred)
        
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        tok_tot = sum(len(r) for r in refs)
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} "
              f"tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% "
              f"time={time.time()-t0:.0f}s")
        E.train(); decoder.train()

# Final
E.eval(); decoder.eval(); refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        if len(ids) < 3: continue
        ids_t = torch.tensor(ids, device=device)
        T = len(ids)
        pos_emb = get_pos_emb(T)
        leaf_hashes = E(ids_t) + 0.5 * pos_emb
        leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
        best_score, best_root = -1, None
        for tname in TEMPLATES:
            root = compute_root_hash(ids, tname)
            score = root.norm().item()
            if score > best_score: best_score, best_root = score, root
        combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
        logits = decoder(combined)
        pred = logits.argmax(dim=-1).cpu().tolist()
        refs.append(ids); hyps.append(pred)
print(f"\nFinal BLEU-4 = {compute_bleu(refs, hyps):.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples ===")
for i in range(5):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device); T = len(ids)
    pos_emb = get_pos_emb(T)
    leaf_hashes = E(ids_t) + 0.5 * pos_emb
    leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
    best_score, best_root, best_tpl = -1, None, None
    for tname in TEMPLATES:
        root = compute_root_hash(ids, tname)
        score = root.norm().item()
        if score > best_score: best_score, best_root, best_tpl = score, root, tname
    combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
    logits = decoder(combined)
    pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:8])}")
    print(f"  hyp: {' '.join(pred[:8])}")
    print(f"  template: {best_tpl} score={best_score:.3f}")
    print()
