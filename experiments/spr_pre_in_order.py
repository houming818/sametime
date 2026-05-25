"""
SPR Pre+In-order Echo — Fully soft, per-token learned routing
Pre-order = W_split decision sequence (tree structure)
In-order  = leaf content probabilities
Depth=9 → 512 leaves > 300 vocab → unique assignment possible
No tanh, no hard routing, no gradient breaks
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")

n_train, n_val, vocab, max_len, d, depth = 500, 100, 300, 12, 128, 9
n_leaves = 1 << depth
print(f"train={n_train} val={n_val} vocab={vocab} max_len={max_len} d={d}")
print(f"depth={depth} leaves={n_leaves} (> {vocab} vocab → unique possible)")

np.random.seed(42)
torch.manual_seed(42)

def gen_sents(n):
    return [list(np.random.randint(1, vocab, size=np.random.randint(3, max_len+1))) 
            for _ in range(n)]

train_sents = gen_sents(n_train)
val_sents = gen_sents(n_val)

# ──── Model ────
class SPREchoTree(nn.Module):
    def __init__(self, V, d, depth):
        super().__init__()
        self.V = V; self.d = d; self.depth = depth
        self.n_leaves = 1 << depth
        
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, 0, 0.02)
        
        # Per-depth splitter: emb → scalar for left/right
        self.W_split = nn.ModuleList([
            nn.Linear(d, 1, bias=False) for _ in range(depth)
        ])
        # Init with small random weights
        for w in self.W_split:
            nn.init.normal_(w.weight, 0, 0.02)
        
        # Leaf prototypes
        self.E_leaf = nn.Parameter(torch.randn(self.n_leaves, d) * 0.02)
        
        # Output: leaf vector → vocab scores
        self.W_out = nn.Linear(d, V)
        nn.init.normal_(self.W_out.weight, 0, 0.02)
        nn.init.zeros_(self.W_out.bias)
    
    def forward(self, ids):
        T = len(ids)
        emb = self.E(ids)  # [T, d]
        
        leaf_probs = torch.ones(T, self.n_leaves, device=emb.device)  # [T, n_leaves]
        
        for level in range(self.depth):
            score = self.W_split[level](emb).squeeze(-1)  # [T]
            left_prob = torch.sigmoid(score)  # [T]
            right_prob = 1.0 - left_prob  # [T]
            
            bit_pos = self.depth - 1 - level
            mask = ((torch.arange(self.n_leaves, device=emb.device) >> bit_pos) & 1).float()  # [n_leaves]
            
            level_prob = mask * right_prob.unsqueeze(1) + (1.0 - mask) * left_prob.unsqueeze(1)
            leaf_probs = leaf_probs * level_prob  # [T, n_leaves]
        
        # leaf_vec: weighted sum of E_leaf for each token
        leaf_vec = leaf_probs @ self.E_leaf  # [T, n_leaves] @ [n_leaves, d] = [T, d]
        leaf_vec = F.layer_norm(leaf_vec, [leaf_vec.shape[-1]])
        
        logits = self.W_out(leaf_vec)  # [T, V]
        
        # For analysis: argmax leaf per token
        leaf_argmax = leaf_probs.argmax(dim=-1)  # [T]
        
        # For analysis: pre-order decisions
        pre_info = torch.zeros(depth, T, dtype=torch.bool, device=emb.device)
        for level in range(self.depth):
            score = self.W_split[level](emb).squeeze(-1)
            pre_info[level] = score > 0
        
        return logits, leaf_argmax, pre_info, leaf_probs


model = SPREchoTree(vocab, d, depth).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

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
    bp = min(1.0, math.exp(max(0, max(1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0))))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

print(f"\n{'='*60}")
print(f"Training {len(train_sents)} sentences, {sum(len(s) for s in train_sents)} tokens")
t0 = time.time()

for epoch in range(200):
    model.train()
    tl, tt = 0, 0
    for ids in train_sents:
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        logits, _, _, _ = model(ids_t)
        loss = F.cross_entropy(logits[:len(ids)], ids_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tl += loss.item() * len(ids)
        tt += len(ids)
    scheduler.step()
    
    if epoch % 40 == 0 or epoch == 199:
        model.eval()
        refs, hyps, all_leafs = [], [], []
        with torch.no_grad():
            for ids in val_sents:
                ids_t = torch.tensor(ids, device=device, dtype=torch.long)
                logits, leaf_idx, pre_info, _ = model(ids_t)
                pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
                refs.append(ids)
                hyps.append(pred)
                all_leafs.extend(leaf_idx.cpu().tolist())
        
        bleu = compute_bleu(refs, hyps)
        leaf_counter = Counter(all_leafs)
        n_used = len(leaf_counter)
        n_collide = sum(1 for v in leaf_counter.values() if v > 1)
        max_per_leaf = max(leaf_counter.values()) if leaf_counter else 0
        
        # Pre-order diversity
        all_pre = torch.cat([pre_info for _, _, pre_info, _ in [
            (None, None, None, None)  # placeholder
        ]]).float().mean() if False else 0.0
        
        print(f"  ep {epoch:3d} loss={tl/tt:.4f} BLEU={bleu:5.1f} "
              f"leaves={n_used}/{n_leaves} collide={n_collide} max/leaf={max_per_leaf} "
              f"time={time.time()-t0:.0f}s")
        model.train()

# ──── Final ────
print(f"\n{'='*60}")
print("Final evaluation...")
model.eval()
refs, hyps = [], []
with torch.no_grad():
    for ids in val_sents:
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        logits, leaf_idx, pre_info, leaf_probs = model(ids_t)
        pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
        refs.append(ids)
        hyps.append(pred)

bleu = compute_bleu(refs, hyps)
print(f"Final BLEU-4 = {bleu:.1f}")

# ──── Per-token accuracy ────
correct = sum(1 for r, h in zip(refs, hyps) 
              for ri, hi in zip(r, h) if ri == hi)
total = sum(len(r) for r in refs)
print(f"Token accuracy = {correct}/{total} = {100*correct/total:.1f}%")

# ──── Samples ────
print(f"\n=== samples ===")
for i in range(5):
    ids = val_sents[i]
    ids_t = torch.tensor(ids, device=device, dtype=torch.long)
    logits, leaf_idx, pre_info, _ = model(ids_t)
    pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
    path_str = ''.join(['L' if p.item()==0 else 'R' for p in pre_info[:, 0]])
    print(f"  ref: {ids}")
    print(f"  hyp: {pred}")
    print(f"  path={path_str}")
    print()

# ──── BLEU by length ────
print(f"=== BLEU by length ===")
len_groups = defaultdict(lambda: ([], []))
for r, h in zip(refs, hyps):
    len_groups[len(r)][0].append(r)
    len_groups[len(r)][1].append(h)
for L in sorted(len_groups.keys()):
    r, h = len_groups[L]
    b = compute_bleu(r, h)
    n = len(r)
    print(f"  len={L:2d} n={n:3d} BLEU={b:5.1f}")

# ──── Leaf collision analysis ────
print(f"\n=== Leaf collision (val set) ===")
all_tokens = defaultdict(set)
for ids in val_sents:
    ids_t = torch.tensor(ids, device=device, dtype=torch.long)
    _, leaf_idx, _, _ = model(ids_t)
    for t, lid in enumerate(leaf_idx.cpu().tolist()):
        all_tokens[lid].add(ids[t])

collide_leaves = {k: v for k, v in all_tokens.items() if len(v) > 1}
print(f"  Total val tokens per unique leaf: avg={np.mean([len(v) for v in all_tokens.values()]):.1f}")
print(f"  Collision leaves: {len(collide_leaves)}/{len(all_tokens)}")
for lid, tokens in sorted(collide_leaves.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"    leaf {lid}: {len(tokens)} tokens → {sorted(list(tokens))[:10]}")
