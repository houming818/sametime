"""
SPR v3 MSE — Pure geometry encoder + W_split decoder
Encoder: fixed 4 templates + roll+SIGN_MASK → root_hash (0 params)
Template: Gumbel-Softmax selection (forward hard, backward soft, gradients flow to E)
Decoder: W_split[depth] recursively splits root → leaves → dot E → tokens
Supervision: MSE on internal node hashes + leaf hashes (no CE, no classifier)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR v3 MSE — Encoder(0 param) + W_split Decoder")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ════════════════════════════════════════
# DATA
# ════════════════════════════════════════
def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t", 1)[1].strip().lower().split())
    return sents

print("loading...")
train_sents = load_sents(train_file, 10000)
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

V, d, MAX_LEN = len(word2id), 128, 32
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d} max_len={MAX_LEN}")

# ════════════════════════════════════════
# FIXED TEMPLATES
# ════════════════════════════════════════
torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d // 2 + 1), device=device)[:d]
TEMPLATES = {
    'Left_Heavy':  lambda n, d: 1,
    'Right_Heavy': lambda n, d: max(1, n - 1),
    'Balanced':    lambda n, d: max(1, n // 2),
    'Spec_Head':   lambda n, d: 3 if n >= 6 else max(1, n // 2),
}
TEMPLATE_NAMES = list(TEMPLATES.keys())

def gen_paths(T, fn):
    def _gen(embs, _, depth=1):
        n = len(embs)
        if n <= 1: return [('', 0)] if n == 1 else []
        s = fn(n, depth); L = max(1, min(s, n - 1))
        l = _gen(embs[:L], None, depth + 1)
        r = _gen(embs[L:], None, depth + 1)
        return [('L' + p, i) for p, i in l] + [('R' + p, i + L) for p, i in r]
    results = _gen(list(range(T)), None)
    paths = [''] * T
    for p, i in results: paths[i] = p
    return paths

MAX_TREE_LEN = 50
template_paths = {t: {T: gen_paths(T, fn) for T in range(2, MAX_TREE_LEN + 1)} for t, fn in TEMPLATES.items()}

# ════════════════════════════════════════
# ENCODER (0 params, pure geometry)
# ════════════════════════════════════════
def encode_tree(E, ids, tau=0.5):
    """
    Gumbel-Softmax template selection.
    Returns: (root, template_name, paths, node_table)
    node_table[depth] = [(parent_hash, left_hash, right_hash), ...]
    """
    T = len(ids)
    if T < 2:
        emb = E(torch.tensor([ids[0] if T >= 1 else 0], device=device))
        return emb, 'Balanced', [''], {}

    k = min(T, MAX_TREE_LEN)

    # Compute root for each template (with grad)
    root_norms = []
    root_candidates = []
    for tn in TEMPLATE_NAMES:
        paths = template_paths[tn].get(k, gen_paths(k, TEMPLATES[tn]))
        if len(paths) != T: paths = gen_paths(T, TEMPLATES[tn])
        if len(paths) != T: paths = [''] * T

        # _build_tree returns root + stores intermediate hashes
        root, node_table = _build_tree_with_table(E, ids, paths)
        root_candidates.append(root)
        root_norms.append(root.norm().detach())  # detach norm for selection

    # Gumbel-Softmax: forward hard (one template), backward soft (all templates)
    logits = torch.stack(root_norms) / tau
    weights = F.gumbel_softmax(logits, tau=tau, hard=True)  # [4], one-hot in forward
    best_idx = weights.argmax().item()
    best_tpl = TEMPLATE_NAMES[best_idx]
    best_paths = template_paths[best_tpl].get(k, gen_paths(k, TEMPLATES[best_tpl]))
    if len(best_paths) != T: best_paths = gen_paths(T, TEMPLATES[best_tpl])

    # Recompute root for best template (with full grad)
    root, node_table = _build_tree_with_table(E, ids, best_paths)

    # Also blend: weighted root from all templates for gradient flow
    blended_root = sum(w * r for w, r in zip(F.softmax(logits / tau, dim=0), root_candidates))

    return blended_root, best_tpl, best_paths, node_table


def _build_tree_with_table(E, ids, paths):
    """Bottom-up merge. Store all internal node (parent, left, right) hashes."""
    ids_t = torch.tensor(ids, device=device)
    embs = E(ids_t)
    T = len(embs)

    cur = {}
    for t in range(T):
        cur[('leaf', paths[t])] = embs[t]

    node_table = {}  # depth → [(parent, left, right)]

    md = max(len(p) for p in paths) if paths else 0
    for depth in range(md, 0, -1):
        node_table[depth] = []
        for pfx in set(p[:depth - 1] if depth > 1 else ''
                       for p in paths if len(p) >= max(depth - 1, 0)):
            lk = ('leaf', pfx + 'L') if depth > 1 else ('leaf', 'L')
            rk = ('leaf', pfx + 'R') if depth > 1 else ('leaf', 'R')
            if lk in cur and rk in cur:
                lft = cur.pop(lk)
                rgt = cur.pop(rk)
                merged = lft + SIGN_MASK * torch.roll(rgt, shifts=depth)
                merged = merged / (merged.norm() + 1e-8)
                cur[('node', pfx)] = merged
                node_table[depth].append((merged, lft, rgt))

    root = next(iter(cur.values())).squeeze() if cur else torch.zeros(d, device=embs.device)
    return root, node_table


# ════════════════════════════════════════
# DECODER (trainable W_split)
# ════════════════════════════════════════
class WSplitDecoder(nn.Module):
    def __init__(self, d, n_splits=3):
        super().__init__()
        self.d = d
        self.splits = nn.ModuleList([nn.Linear(d, d * 2) for _ in range(n_splits)])

    def forward(self, root, paths, node_table):
        T = len(paths)
        pred_leaves = [None] * T
        layer_losses = []  # Fix 2: per-layer average, not global average

        def split_recurse(node, idxs, depth):
            if len(idxs) == 0: return
            if all(len(paths[i]) <= depth for i in idxs):
                for i in idxs: pred_leaves[i] = node
                return

            left_idx, right_idx = [], []
            for i in idxs:
                if len(paths[i]) > depth:
                    (left_idx if paths[i][depth] == 'L' else right_idx).append(i)
                else:
                    left_idx.append(i)

            level = depth % len(self.splits)
            split_out = self.splits[level](node)
            left_child = split_out[:self.d]
            right_child = split_out[self.d:]

            # Fix 3: sphere normalization — lock output to unit sphere (match encoder)
            left_child = left_child / (left_child.norm() + 1e-8)
            right_child = right_child / (right_child.norm() + 1e-8)

            # Fix 2: accumulate per-layer, then average across layers
            if depth + 1 in node_table:
                step_loss = torch.tensor(0.0, device=root.device)
                step_count = 0
                for ph, lh, rh in node_table[depth + 1]:
                    step_loss += F.mse_loss(left_child, lh) + F.mse_loss(right_child, rh)
                    step_count += 1
                if step_count > 0:
                    layer_losses.append(step_loss / step_count)

            split_recurse(left_child, left_idx, depth + 1)
            split_recurse(right_child, right_idx, depth + 1)

        split_recurse(root, list(range(T)), 0)

        loss_internal = sum(layer_losses) / len(layer_losses) if layer_losses else torch.tensor(0.0, device=root.device)

        for i in range(T):
            if pred_leaves[i] is None:
                pred_leaves[i] = torch.zeros(self.d, device=root.device)

        return torch.stack(pred_leaves, dim=0), loss_internal


def get_pos_emb(T, dim=128):
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = 10000 ** (torch.arange(0, dim // 2, device=device).float() * 2 / dim)
    return torch.cat([torch.sin(pos / div), torch.cos(pos / div)], dim=-1)

# ════════════════════════════════════════
# INIT
# ════════════════════════════════════════
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
decoder = WSplitDecoder(d, n_splits=3).to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
nP = sum(p.numel() for m in [E, decoder] for p in m.parameters())
print(f"params={nP / 1e6:.1f}M")

# ════════════════════════════════════════
# BLEU
# ════════════════════════════════════════
def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values()); mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bpv = [1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    bp = min(2.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents[:100] if len(s) >= 2]

# ════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"Training v3 MSE: Encoder(0 param) → node_table → W_split Decoder → leaves")
print(f"  epochs=30 batch=16 lr=0.003")
t0 = time.time()
tau = 1.0  # Gumbel-Softmax temperature

for epoch in range(30):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl_node, tl_leaf, tt = 0, 0, 0, 0
    tau = max(0.1, 1.0 - epoch / 20.0)  # anneal τ: 1.0 → 0.1

    for bi in range(0, 2000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl_node, bl_leaf, n = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0

        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            T = min(len(ids), MAX_LEN)
            ids = ids[:T]
            ids_t = torch.tensor(ids, device=device)

            # Encode: root + node_table (encoder is 0-param, pure geometry)
            # Gumbel-Softmax gives blended root with gradient flow
            root, tpl, paths, node_table = encode_tree(E, ids, tau)

            # Decode: follow paths, produce leaves + internal node MSE
            pred_leaves, loss_node = decoder(root, paths, node_table)

            # Leaf MSE: pred_leaves → close to gold token embeddings + position info
            pos_emb = get_pos_emb(T)
            gold_leaf = E(ids_t) + 0.3 * pos_emb  # position-differentiated targets
            gold_leaf = gold_leaf / (gold_leaf.norm(dim=-1, keepdim=True) + 1e-8)
            loss_leaf = F.mse_loss(pred_leaves[:T], gold_leaf)

            loss = loss_node + loss_leaf
            bl_node += loss_node.item(); bl_leaf += loss_leaf.item()
            batch_loss = loss_node + loss_leaf
            batch_loss.backward(retain_graph=False)  # accumulate grad
            n += 1

        if n == 0: continue
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
        opt.step()
        ti += 1; tl_node += bl_node / n; tl_leaf += bl_leaf / n; tt += 1

    if ti == 0: continue
    avg_node = tl_node / ti; avg_leaf = tl_leaf / ti

    if epoch % 5 == 0 or epoch == 29:
        E.eval(); decoder.eval()
        rf, hp = [], []
        cos_all = []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = min(len(ids), MAX_LEN)
                ids = ids[:T]
                root, tpl, paths, node_table = encode_tree(E, ids, tau=0.1)
                pred_leaves, _ = decoder(root, paths, node_table)

                # Leaf cosine diagnostic
                cos_mat = F.cosine_similarity(
                    pred_leaves[:T].unsqueeze(0), pred_leaves[:T].unsqueeze(1), dim=-1
                )
                cos_all.append(cos_mat)

                # Token prediction: dot product with E.weight
                logits = pred_leaves[:T] @ E.weight.T  # [T, V]
                pred = logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T])
                hp.append(pred)

        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))

        # Mean off-diagonal cosine (lower = more differentiated)
        if cos_all:
            off_diag_vals = []
            for c in cos_all:
                n = c.shape[0]
                mask = ~torch.eye(n, dtype=torch.bool, device=c.device)
                off_diag_vals.append(c[mask])
            mean_cos = torch.cat(off_diag_vals).mean().item() if off_diag_vals else 1.0
        else:
            mean_cos = 1.0

        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} node_mse={avg_node:.4f} leaf_mse={avg_leaf:.4f} "
              f"cos={mean_cos:.3f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% "
              f"τ={tau:.2f} {elapsed:.0f}s")
        E.train(); decoder.train()

# ════════════════════════════════════════
# FINAL
# ════════════════════════════════════════
E.eval(); decoder.eval()
rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        T = min(len(ids), MAX_LEN)
        ids = ids[:T]
        root, tpl, paths, node_table = encode_tree(E, ids, tau=0.1)
        pred_leaves, _ = decoder(root, paths, node_table)
        logits = pred_leaves[:T] @ E.weight.T
        pred = logits.argmax(dim=-1).cpu().tolist()
        rf.append(ids[:T])
        hp.append(pred)

bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s][:MAX_LEN]
    if len(ids) < 3: continue
    with torch.no_grad():
        root, tpl, paths, node_table = encode_tree(E, ids, tau=0.1)
        pred_leaves, _ = decoder(root, paths, node_table)
        logits = pred_leaves[:len(ids)] @ E.weight.T
        pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}  [{tpl}]")
    print()
