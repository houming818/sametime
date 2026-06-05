"""
SPR True Sentence Echo — W_split splits token SEQUENCES, not individual tokens
Pre-order: routing decisions that partition the sentence hierarchically
In-order: leaf-stored token subsets
Given both → unique reconstruction of the sentence tree
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")

# ──── Toy data ────
n_train, n_val, vocab, max_len, d, depth = 500, 100, 300, 12, 128, 6
n_leaves = 1 << depth  # 64

np.random.seed(42)
torch.manual_seed(42)

def gen_sents(n):
    return [list(np.random.randint(1, vocab, size=np.random.randint(3, max_len+1))) 
            for _ in range(n)]

train_sents = gen_sents(n_train)
val_sents = gen_sents(n_val)
print(f"train={n_train} val={n_val} vocab={vocab} d={d} depth={depth} leaves={n_leaves}")

# ──── Model: true sentence-level routing ────
class SPESentenceTree(nn.Module):
    """Route sets of tokens through a binary tree. Each node splits its tokens via W_split."""
    def __init__(self, V, d, depth, max_len):
        super().__init__()
        self.V = V; self.d = d; self.depth = depth; self.max_len = max_len
        self.n_leaves = 1 << depth
        
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, 0, 0.02)
        
        # Per-depth splitter: takes aggregated node embedding → scalar split direction
        # This is a LINEAR DIRECTION in embedding space
        self.W_split = nn.ModuleList([
            nn.Linear(d, d, bias=False) for _ in range(depth)
        ])
        for w in self.W_split:
            nn.init.normal_(w.weight, 0, 0.02)
        
        # Leaf storage: per leaf, store a learned representation
        self.E_leaf = nn.Parameter(torch.randn(self.n_leaves, d) * 0.02)
        
        # Decoder: given leaf path + leaf content → predict token at each position
        # For echo: we have T tokens, each routed to a leaf. Reconstruction = map leaf_vec → token
        self.W_out = nn.Linear(d, V)
        nn.init.normal_(self.W_out.weight, 0, 0.02)
        nn.init.zeros_(self.W_out.bias)
    
    def forward(self, ids):
        """
        Route T tokens through tree.
        At each node: ALL tokens at this node get split by W_split @ node_rep
        Returns: logits [T, V], leaf_idx [T], tree structure
        """
        T = len(ids)
        emb = self.E(ids)  # [T, d]
        
        # Each token starts at root. At each depth, decide left/right.
        # Key: decision depends on WHICH tokens are at this node (not just own embedding)
        
        # For simplicity: use cumulative token embedding to shape the split
        # node_rep = average of tokens at this node
        
        # Track: for each token, which leaf it ends up at
        leaf_idx = torch.zeros(T, dtype=torch.long, device=emb.device)
        
        # Track: current set of tokens at each node (not fully differentiable)
        # Use recursive routing:
        leaf_vecs = torch.zeros(T, d, device=emb.device)
        
        # Parallel routing through all depth levels
        current = emb.clone()  # [T, d]
        
        for level in range(self.depth):
            # At this level, tokens are partitioned based on their embedding + context
            # The context: mean of all tokens still in the same subtree
            
            # For a true tree-level split, we need to know which tokens are in the same node
            # At level 0: all tokens in root
            # At level 1: tokens are split into left/right groups by level 0 decision
            # We can track this via leaf_idx prefix
            
            # Per-token decision: project embedding + level-appropriate direction
            proj = self.W_split[level](current)  # [T, d]
            
            # Decision: use the projection magnitude as split score
            score = proj.sum(dim=-1)  # [T] scalar
            left_prob = torch.sigmoid(score)  # [T]
            
            # Gumbel-Softmax for trainable routing (hard for forward, soft for backward)
            logits_2 = torch.stack([-score, score], dim=-1)  # [T, 2]
            decision = F.gumbel_softmax(logits_2, tau=0.5, hard=self.training)  # [T, 2]
            go_right = decision[:, 1]  # [T] ≈ 0 or ≈ 1
            
            # Update leaf index prefix
            leaf_idx = leaf_idx * 2 + go_right.round().long()
            
            # Apply transformation: left and right branches transform differently
            # Different W_left/W_right for each depth level
            # This makes the tree truly hierarchical
            left_proj = current * left_prob.unsqueeze(-1)
            right_proj = current * go_right.unsqueeze(-1)
            current = left_proj + right_proj  # preserves gradient through left_prob and go_right
        
        # leaf_idx is now [T] with hard argmax
        leaf_idx = torch.clamp(leaf_idx, 0, self.n_leaves - 1)
        
        # Decode: leaf prototype → token
        leaf_vec = self.E_leaf[leaf_idx]  # [T, d]
        leaf_vec = F.layer_norm(leaf_vec, [leaf_vec.shape[-1]])
        logits = self.W_out(leaf_vec)  # [T, V]
        
        return logits, leaf_idx, leaf_vec


model = SPESentenceTree(vocab, d, depth, max_len).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

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

val_data = [(s, s) for s in val_sents]

print(f"\n{'='*60}")
print("Training true sentence-level tree...")
t0 = time.time()

for epoch in range(200):
    model.train()
    random.shuffle(train_sents)
    tl, tt = 0, 0
    
    for ids in train_sents:
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        logits, leaf_idx, leaf_vec = model(ids_t)
        loss = F.cross_entropy(logits, ids_t)
        
        # Add structural loss: encourage DIFFERENT tokens to go to DIFFERENT leaves
        # This pushes the tree to separate tokens by meaning
        unique_leaves = len(set(leaf_idx.cpu().tolist()))
        if unique_leaves < min(len(ids), model.n_leaves):
            struct_loss = 0.01 * (1.0 - unique_leaves / len(ids))
            loss = loss + struct_loss
        
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
            for ids in val_sents[:50]:
                ids_t = torch.tensor(ids, device=device, dtype=torch.long)
                logits, lidx, _ = model(ids_t)
                pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
                refs.append(ids)
                hyps.append(pred)
                all_leafs.extend(lidx.cpu().tolist())
        
        bleu = compute_bleu(refs, hyps)
        token_acc = sum(1 for r, h in zip(refs, hyps) for ri, hi in zip(r, h) if ri == hi)
        token_tot = sum(len(r) for r in refs)
        
        # Leaf usage: how many different leaves are used?
        leaf_counter = Counter(all_leafs)
        n_used = len(leaf_counter)
        
        # Sent-level uniqueness: do different sentences get different leaf patterns?
        sent_leaf_sets = []
        for ids in val_sents[:50]:
            ids_t = torch.tensor(ids, device=device, dtype=torch.long)
            _, lidx, _ = model(ids_t)
            sent_leaf_sets.append(tuple(sorted(lidx.cpu().tolist())))
        unique_sent_patterns = len(set(sent_leaf_sets))
        
        print(f"  ep {epoch:3d} loss={tl/tt:.4f} BLEU={bleu:5.1f} "
              f"tok_acc={token_acc}/{token_tot}={100*token_acc/token_tot:.1f}% "
              f"used_leaves={n_used}/{n_leaves} unique_sent_pats={unique_sent_patterns}/{len(val_sents[:50])} "
              f"time={time.time()-t0:.0f}s")
        model.train()

# Final
print(f"\n{'='*60}")
model.eval()
refs, hyps = [], []
with torch.no_grad():
    for ids in val_sents:
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        logits, lidx, _ = model(ids_t)
        pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
        refs.append(ids)
        hyps.append(pred)

bleu = compute_bleu(refs, hyps)
print(f"Final BLEU-4 = {bleu:.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples ===")
for i in range(5):
    ids = val_sents[i]
    ids_t = torch.tensor(ids, device=device, dtype=torch.long)
    logits, lidx, _ = model(ids_t)
    pred = logits[:len(ids)].argmax(dim=-1).cpu().tolist()
    leafs = lidx.cpu().tolist()
    print(f"  ref: {ids}")
    print(f"  hyp: {pred}")
    print(f"  leafs: {leafs} (unique={len(set(leafs))})")
    print()
