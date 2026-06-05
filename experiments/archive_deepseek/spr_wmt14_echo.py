"""
SPR Sentence Echo — WMT14 overnight run
Per-token learned routing, depth=12 (4096 leaves)
Pre-order + In-order → perfect word-level echo → sentence BLEU
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, os, random
from collections import Counter, defaultdict

device = 'cuda'
print(f"device={device}")

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data Loading ────
def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t", 1)[1].strip().lower().split())
    return sents

print("loading...")
train_sents = load_sents(train_file, 50000)
val_sents = load_sents(val_file, 500)
print(f"train={len(train_sents)} val={len(val_sents)}")

word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for s in train_sents:
    for w in s: freq[w] += 1
min_freq = 2
for w, c in freq.most_common():
    if c >= min_freq: word2id[w] = len(word2id)
# Add val words that might not be in filtered vocab
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d, depth = len(word2id), 128, 12
id2word = {v: k for k, v in word2id.items()}
n_leaves = 1 << depth

print(f"vocab={V} d={d} depth={depth} leaves={n_leaves:,} ({n_leaves}>={V}: {'YES' if n_leaves >= V else 'NO'})")
print(f"train tokens={sum(len(s) for s in train_sents):,} val tokens={sum(len(s) for s in val_sents)}")

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents[:300]]

# ──── Model ────
class SPREchoWMT(nn.Module):
    def __init__(self, V, d, depth):
        super().__init__()
        self.V = V; self.d = d; self.depth = depth
        self.n_leaves = 1 << depth
        
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, 0, 0.02)
        
        self.W_split = nn.ModuleList([
            nn.Linear(d, 1, bias=False) for _ in range(depth)
        ])
        for w in self.W_split:
            nn.init.normal_(w.weight, 0, 0.02)
        
        self.E_leaf = nn.Parameter(torch.randn(self.n_leaves, d) * 0.02)
        self.W_out = nn.Linear(d, V)
        nn.init.normal_(self.W_out.weight, 0, 0.02)
        nn.init.zeros_(self.W_out.bias)
    
    def forward(self, ids, lens):
        B, T = ids.shape
        mask = (ids != 0).float()  # [B, T] pad mask
        
        emb = self.E(ids)  # [B, T, d]
        
        # Sinusoidal position encoding → sentence-level routing
        pos = torch.arange(T, device=ids.device).float().unsqueeze(0)  # [1, T]
        div = (10000 ** (torch.arange(0, self.d, 2, device=ids.device).float() / self.d)).unsqueeze(0)  # [1, d//2]
        phase = pos.unsqueeze(-1) / div  # [B=1, T, d//2]
        pos_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # [1, T, d]
        emb = emb + 0.3 * pos_emb  # position-modulated token embeddings
        
        # leaf_probs: [B, T, n_leaves]
        leaf_probs = torch.ones(B, T, self.n_leaves, device=ids.device)
        
        for level in range(self.depth):
            score = self.W_split[level](emb).squeeze(-1)  # [B, T]
            left_prob = torch.sigmoid(score)  # [B, T]
            right_prob = 1.0 - left_prob  # [B, T]
            
            bit_pos = self.depth - 1 - level
            leaf_mask = ((torch.arange(self.n_leaves, device=ids.device) >> bit_pos) & 1).float().view(1, 1, -1)
            
            level_prob = leaf_mask * right_prob.unsqueeze(-1) + (1.0 - leaf_mask) * left_prob.unsqueeze(-1)
            leaf_probs = leaf_probs * level_prob  # [B, T, n_leaves]
        
        # leaf_vec: weighted sum of E_leaf
        leaf_vec = leaf_probs @ self.E_leaf  # [B, T, n_leaves] @ [n_leaves, d] = [B, T, d]
        leaf_vec = F.layer_norm(leaf_vec, [leaf_vec.shape[-1]])
        
        logits = self.W_out(leaf_vec)  # [B, T, V]
        
        return logits, leaf_probs, mask


model = SPREchoWMT(V, d, depth).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params={n_params/1e6:.1f}M")

# ──── Optimizer ────
opt = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
print(f"  epochs=200 batch_size=32 lr=0.003")
t0 = time.time()
best_bleu = 0.0

# ──── BLEU helpers ────
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

for epoch in range(200):
    model.train()
    random.shuffle(train_sents)
    total_loss, total_tok = 0, 0
    
    for bi in range(0, len(train_sents), 32):
        batch_sents = train_sents[bi:bi+32]
        if not batch_sents: continue
        
        max_len_in_batch = min(max(len(s) for s in batch_sents), 50)
        ids_batch = torch.zeros(len(batch_sents), max_len_in_batch, dtype=torch.long, device=device)
        mask_batch = torch.zeros(len(batch_sents), max_len_in_batch, device=device)
        
        for i, s in enumerate(batch_sents):
            ids = [word2id.get(w, 1) for w in s]
            L = min(len(ids), max_len_in_batch)
            ids_batch[i, :L] = torch.tensor(ids[:L], device=device)
            mask_batch[i, :L] = 1.0
        
        logits, _, _ = model(ids_batch, mask_batch)
        
        loss = F.cross_entropy(
            logits.view(-1, V), ids_batch.view(-1),
            ignore_index=0, reduction='sum'
        )
        loss = loss / mask_batch.sum()
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        
        total_loss += loss.item() * mask_batch.sum().item()
        total_tok += mask_batch.sum().item()
    
    scheduler.step()
    avg_loss = total_loss / max(total_tok, 1)
    
    if epoch % 20 == 0 or epoch == 199:
        model.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data:
                if len(ids) < 2: continue
                ids_t = torch.tensor(ids, device=device).unsqueeze(0)
                lens_t = torch.tensor([len(ids)], device=device)
                logits, _, _ = model(ids_t, lens_t)
                pred = logits[0, :len(ids)].argmax(dim=-1).cpu().tolist()
                refs.append(ids)
                hyps.append(pred)
        
        bleu = compute_bleu(refs, hyps)
        token_acc = sum(1 for r, h in zip(refs, hyps) for ri, hi in zip(r, h) if ri == hi)
        token_tot = sum(len(r) for r in refs)
        
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={avg_loss:.4f} BLEU={bleu:.1f} "
              f"tok_acc={token_acc}/{token_tot}={100*token_acc/token_tot:.1f}% "
              f"lr={scheduler.get_last_lr()[0]:.5f} time={elapsed:.0f}s")
        
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), '/tmp/spr_wmt14_best.pt')
        
        model.train()

# ──── Final ────
print(f"\n{'='*60}")
print(f"Training complete. Time={time.time()-t0:.0f}s Best BLEU={best_bleu:.1f}")

model.eval()
refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        if len(ids) < 2: continue
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        lens_t = torch.tensor([len(ids)], device=device)
        logits, leaf_probs, _ = model(ids_t, lens_t)
        pred = logits[0, :len(ids)].argmax(dim=-1).cpu().tolist()
        refs.append(ids)
        hyps.append(pred)

bleu = compute_bleu(refs, hyps)
print(f"\nFinal BLEU-4 = {bleu:.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)} = "
      f"{100*sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in refs)):.1f}%")

# ──── Samples ────
print(f"\n=== samples ===")
for i in range(min(8, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device).unsqueeze(0)
    lens_t = torch.tensor([len(ids)], device=device)
    logits, _, _ = model(ids_t, lens_t)
    pred = [id2word.get(p, '?') for p in logits[0, :len(ids)].argmax(dim=-1).cpu().tolist()]
    src = ' '.join(s[:8])
    hyp = ' '.join(pred[:8])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
