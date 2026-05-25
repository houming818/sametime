"""
SPR Syntactic Distance Tree — Autoencoder with Internal Node Supervision
Encoder: pure geometry (0 params), stores ALL node hashes
Decoder: trained W_split, supervised by MSE on intermediate nodes + CE on leaves
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda'
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

# ──── Encoder: Syntactic Distance Tree (stores ALL node hashes) ────
def syntactic_distances(embs):
    T = embs.shape[0]
    dists = torch.zeros(T-1, device=device)
    current = embs.clone()
    for dp in range(1, 4):
        rolled = torch.roll(current, shifts=dp, dims=-1) * SIGN_MASK
        scores = (embs * rolled).sum(dim=-1)
        dists += (scores[:-1] - scores[1:]).abs()
    return dists / 3.0

def build_tree_with_hashes(embs, dists, tree_depth=1):
    """
    Build tree AND store all node hashes.
    Returns: (root_hash, node_hashes, tree_struct, leaf_hashes)
    node_hashes: list of (depth, node_hash) for all internal nodes
    """
    T = len(embs)
    if T <= 1:
        leaf = embs[0] if T >= 1 else torch.zeros(d, device=device)
        return leaf, [], [('leaf',)], [leaf]
    if len(dists) == 0:
        return embs[0], [], [('leaf',)], [embs[0]]
    
    split_pos = dists.argmax().item()
    L = split_pos + 1
    
    left_embs = embs[:L]
    left_dists = dists[:split_pos] if split_pos > 0 else dists[:0]
    right_embs = embs[L:]
    right_dists = dists[L:] if L < len(dists) else dists[:0]
    
    left_hash, left_nodes, left_struct, left_leaves = \
        build_tree_with_hashes(left_embs, left_dists, tree_depth + 1)
    right_hash, right_nodes, right_struct, right_leaves = \
        build_tree_with_hashes(right_embs, right_dists, tree_depth + 1)
    
    # Merge with depth-dependent shift
    shifts = tree_depth
    merged = left_hash + SIGN_MASK * torch.roll(right_hash, shifts=shifts)
    merged = merged / (merged.norm() + 1e-8)
    
    # Store this node's hash (for supervision)
    node_hashes = [(tree_depth, merged.detach(), left_hash.detach(), right_hash.detach())]
    node_hashes += left_nodes + right_nodes
    
    tree_struct = [('node', tree_depth, left_struct, right_struct)] + left_struct + right_struct
    all_leaves = left_leaves + right_leaves
    
    return merged, node_hashes, tree_struct, all_leaves


# ──── Decoder: Trainable W_split ────
class WSplitDecoder(nn.Module):
    def __init__(self, d, n_modules=8):
        super().__init__()
        self.d = d
        self.W_split = nn.ModuleList([
            nn.Linear(d, d * 2) for _ in range(n_modules)
        ])
        for w in self.W_split:
            nn.init.normal_(w.weight, 0, 0.02)
            nn.init.zeros_(w.bias)
    
    def forward(self, node_hash, depth, tree_struct):
        """
        Recursively decode. Returns (node_hashes, leaf_hashes)
        node_hashes: for MSE supervision, leaf_hashes: for CE supervision
        """
        return self._decode(node_hash, depth, tree_struct)
    
    def _decode(self, node_hash, depth, struct):
        if not struct:
            return [], [node_hash]
        
        head = struct[0]
        if head[0] == 'leaf':
            return [], [node_hash]
        
        _, node_depth, left_struct, right_struct = head
        
        idx = depth % len(self.W_split)
        split_out = self.W_split[idx](node_hash)
        left_pred = split_out[:self.d]
        right_pred = split_out[self.d:]
        
        # Store decoded node hash for supervision (depth, pred, structure)
        decoded_nodes = [(node_depth, node_hash, left_pred, right_pred)]
        
        left_decoded, left_leaves = self._decode(left_pred, depth + 1, left_struct)
        right_decoded, right_leaves = self._decode(right_pred, depth + 1, right_struct)
        
        return decoded_nodes + left_decoded + right_decoded, left_leaves + right_leaves


# ──── Train + Eval ────
decoder = WSplitDecoder(d, n_modules=6).to(device)
opt = torch.optim.Adam(list(decoder.parameters()), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

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
print(f"Training W_split decoder with internal node supervision...")
print(f"  n_modules=6 epochs=300 batch=32 lr=0.003")
t0 = time.time()

for epoch in range(300):
    decoder.train()
    random.shuffle(train_sents)
    ti, tl_node, tl_ce, tt = 0, 0, 0, 0
    
    for bi in range(0, 3000, 32):  # 3000 sentences per epoch for speed
        batch_sents = train_sents[bi:bi+32]
        if not batch_sents: continue
        
        for s in batch_sents:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            
            ids_t = torch.tensor(ids, device=device)
            embs = E(ids_t)
            
            # ── Encode (no grad, save node hashes for supervision) ──
            with torch.no_grad():
                dists = syntactic_distances(embs)
                root_hash, true_nodes, tree_struct, true_leaves = \
                    build_tree_with_hashes(embs, dists)
            decoded_nodes, decoded_leaves = decoder(root_hash, 1, tree_struct)
            
            # ── Losses ──
            # MSE on internal nodes: each (depth, parent) → (left, right) vs true
            loss_node = torch.tensor(0.0, device=device)
            node_count = 0
            for (depth, parent, true_left, true_right) in true_nodes:
                # Find matching decoded node
                for (d_depth, d_parent, d_left, d_right) in decoded_nodes:
                    if d_depth == depth:
                        # Supervise: d_left ≈ true_left, d_right ≈ true_right
                        loss_node += F.mse_loss(d_left, true_left) + F.mse_loss(d_right, true_right)
                        node_count += 1
                        break
            
            if node_count > 0:
                loss_node = loss_node / node_count
            
            # CE on leaf tokens
            loss_ce = torch.tensor(0.0, device=device)
            leaf_count = 0
            for lh in decoded_leaves:
                logits = lh @ E.weight.T  # [V]
                if leaf_count < len(ids):
                    loss_ce += F.cross_entropy(logits.unsqueeze(0), ids_t[leaf_count].unsqueeze(0))
                leaf_count += 1
            
            if leaf_count > 0:
                loss_ce = loss_ce / leaf_count
            
            loss = loss_node + loss_ce
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)
            opt.step()
            
            ti += 1; tl_node += loss_node.item(); tl_ce += loss_ce.item(); tt += len(ids)
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 30 == 0 or epoch == 299:
        decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                ids_t = torch.tensor(ids, device=device)
                embs = E(ids_t)
                dists = syntactic_distances(embs)
                root_hash, _, tree_struct, _ = build_tree_with_hashes(embs, dists)
                _, d_leaves = decoder(root_hash, 1, tree_struct)
                
                pred = []
                for lh in d_leaves[:len(ids)]:
                    logits = lh @ E.weight.T
                    pred.append(int(logits.argmax().item()))
                
                refs.append(ids)
                hyps.append(pred[:len(ids)])
        
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r, h in zip(refs, hyps) for ri, hi in zip(r, h) if ri == hi)
        tok_tot = sum(len(r) for r in refs)
        
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} node_loss={tl_node/ti:.4f} ce_loss={tl_ce/ti:.4f} "
              f"BLEU={bleu:.1f} tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% "
              f"time={elapsed:.0f}s")
        decoder.train()

# ──── Final ────
print(f"\n{'='*60}")
decoder.eval()
refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        ids_t = torch.tensor(ids, device=device)
        embs = E(ids_t)
        dists = syntactic_distances(embs)
        root_hash, _, tree_struct, _ = build_tree_with_hashes(embs, dists)
        _, d_leaves = decoder(root_hash, 1, tree_struct)
        
        pred = []
        for lh in d_leaves[:len(ids)]:
            pred.append(int((lh @ E.weight.T).argmax().item()))
        
        refs.append(ids)
        hyps.append(pred[:len(ids)])

bleu = compute_bleu(refs, hyps)
print(f"Final BLEU-4 = {bleu:.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device)
    embs = E(ids_t)
    dists = syntactic_distances(embs)
    root_hash, _, tree_struct, _ = build_tree_with_hashes(embs, dists)
    _, d_leaves = decoder(root_hash, 1, tree_struct)
    pred = [id2word.get((lh @ E.weight.T).argmax().item(), '?') for lh in d_leaves[:len(ids)]]
    src = ' '.join(s[:8])
    hyp = ' '.join(pred[:8])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
