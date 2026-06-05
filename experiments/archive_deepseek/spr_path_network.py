"""
SPR Discrete Spatio-Temporal Path Network — No root hash
Encode: syntactic distance tree → per-token path labels
Decode: path_emb[t] + GRU(prev_token) → token prediction
Sentence = group of interwoven tree trajectories, not a single hash
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
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]

# ──── Encode: tree → per-token path labels (NO root hash) ────
def syntactic_distances(embs):
    T = embs.shape[0]
    dists = torch.zeros(T-1, device=device)
    current = embs.clone()
    for dp in range(1, 4):
        rolled = torch.roll(current, shifts=dp, dims=-1) * SIGN_MASK
        scores = (embs * rolled).sum(dim=-1)
        dists += (scores[:-1] - scores[1:]).abs()
    return dists / 3.0

def build_paths_only(embs, dists):
    """Build tree, return ONLY per-token path labels (no hashes)"""
    T = len(embs)
    if T <= 1:
        return [''] if T >= 1 else []
    if len(dists) == 0:
        return ['']
    
    split_pos = dists.argmax().item(); L = split_pos + 1
    left_paths = build_paths_only(
        embs[:L], dists[:split_pos] if split_pos > 0 else dists[:0])
    right_paths = build_paths_only(
        embs[L:], dists[L:] if L < len(dists) else dists[:0])
    
    return ['L' + p for p in left_paths] + ['R' + p for p in right_paths]


def get_sentence_paths(ids):
    """Encode sentence → list of path strings (one per token)"""
    if len(ids) < 2:
        return [''] if len(ids) == 1 else []
    ids_t = torch.tensor(ids, device=device)
    embs = E(ids_t)
    dists = syntactic_distances(embs)
    return build_paths_only(embs, dists)


# ──── Decode: path + momentum → token ────
class PathNetworkDecoder(nn.Module):
    """Auto-regressive decoder: path_emb[t] + prev_token → GRU → logits[t]"""
    def __init__(self, d, max_path_len=32):
        super().__init__()
        self.d = d
        
        # Encode path: L/R sequence → path embedding
        self.path_enc_L = nn.Embedding(max_path_len, d)
        self.path_enc_R = nn.Embedding(max_path_len, d)
        
        # Transform path_vec to match GRU input size
        self.path_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        
        # GRU for autoregressive context
        self.gru = nn.GRUCell(d, d)
        
        # Output projection
        self.W_out = nn.Linear(d, V)
    
    def encode_path(self, path):
        """path string → d-dim vector"""
        v = torch.zeros(self.d, device=self.path_enc_L.weight.device)
        for depth, c in enumerate(path):
            if depth >= self.path_enc_L.num_embeddings: break
            idx = torch.tensor(depth, device=v.device)
            pe = self.path_enc_L(idx) if c == 'L' else self.path_enc_R(idx)
            v = v + pe
        return v / max(len(path), 1)
    
    def forward_teacher(self, path_strings, gold_token_ids, E_weight):
        """
        Teacher forcing: at each step, use gold_prev as GRU input
        path_strings: list of path strings (one per token position)
        gold_token_ids: [T] tensor of gold token IDs
        Returns: logits [T, V]
        """
        T = len(path_strings)
        h = torch.zeros(self.d, device=self.path_enc_L.weight.device)
        logits_list = []
        
        for t in range(T):
            path_vec = self.encode_path(path_strings[t])
            path_vec = self.path_proj(path_vec)
            
            if t == 0:
                inp = path_vec  # first token: only path info
            else:
                prev_emb = E_weight[gold_token_ids[t-1]]
                inp = path_vec + prev_emb
            
            h = self.gru(inp, h)
            logits_list.append(self.W_out(h))
        
        return torch.stack(logits_list, dim=0)  # [T, V]
    
    def forward_generate(self, path_strings, E_weight, max_len=50):
        """Auto-regressive generation (no teacher)"""
        T = len(path_strings)
        h = torch.zeros(self.d, device=self.path_enc_L.weight.device)
        prev_id = torch.tensor(1, device=h.device).long()  # <unk> as start
        preds = []
        
        for t in range(T):
            path_vec = self.encode_path(path_strings[t])
            path_vec = self.path_proj(path_vec)
            
            if t == 0:
                inp = path_vec
            else:
                prev_emb = E_weight[prev_id]
                inp = path_vec + prev_emb
            
            h = self.gru(inp, h)
            logits = self.W_out(h)
            prev_id = logits.argmax(dim=-1)
            preds.append(prev_id.item())
        
        return preds


decoder = PathNetworkDecoder(d).to(device)
opt = torch.optim.Adam(list(decoder.parameters()) + list(E.parameters()), lr=0.003)
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

print(f"\n{'='*60}")
print("Training Path Network (no root hash, path + GRU momentum)")
print(f"  epochs=100 batch=16 lr=0.003")
t0 = time.time()

for epoch in range(100):
    decoder.train(); E.train()
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
            paths = get_sentence_paths(ids)
            
            if len(paths) != len(ids):
                continue  # path generation failed
            
            logits = decoder.forward_teacher(paths, ids_t, E.weight)
            loss = F.cross_entropy(logits, ids_t)
            batch_loss = batch_loss + loss
            n_sents += 1
        
        if n_sents == 0: continue
        batch_loss = batch_loss / n_sents
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(E.parameters()), 2.0)
        opt.step()
        
        ti += 1; tl += batch_loss.item(); tt += 1
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 99:
        decoder.eval(); E.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                if len(ids) < 3: continue
                ids_t = torch.tensor(ids, device=device)
                paths = get_sentence_paths(ids)
                if len(paths) != len(ids): continue
                logits = decoder.forward_teacher(paths, ids_t, E.weight)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids); hyps.append(pred)
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        tok_tot = sum(len(r) for r in refs)
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} "
              f"tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% "
              f"time={time.time()-t0:.0f}s")
        decoder.train(); E.train()

# ──── Final ────
decoder.eval(); E.eval(); refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        if len(ids) < 3: continue
        ids_t = torch.tensor(ids, device=device)
        paths = get_sentence_paths(ids)
        if len(paths) != len(ids): continue
        logits = decoder.forward_teacher(paths, ids_t, E.weight)
        pred = logits.argmax(dim=-1).cpu().tolist()
        refs.append(ids); hyps.append(pred)
print(f"\nFinal BLEU-4 = {compute_bleu(refs, hyps):.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples (auto-regressive gen) ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    paths = get_sentence_paths(ids)
    if len(paths) != len(ids): continue
    pred = decoder.forward_generate(paths, E.weight)
    pred_words = [id2word.get(p, '?') for p in pred]
    print(f"  src: {' '.join(s[:8])}")
    print(f"  hyp: {' '.join(pred_words[:8])}")
    print(f"  path: {' '.join(p[:3] for p in paths[:5])}")
    print()
