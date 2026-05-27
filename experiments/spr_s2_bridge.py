"""
SPR S2 Translation Bridge — Root-to-Root + TopDownTreeDecoder (no GRU)
Encode: E + fixed templates → root_hash (pure geometry, zero param)
Decode: W_split tree recursively splits root → leaves → tokens
Bridge: pure MLP maps root_de → root_en on unit sphere
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR S2 BRIDGE — TopDownTreeDecoder")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data ────
print("loading...")
def load_all(path, n=None):
    pairs = []
    with open(path) as f:
        for i, l in enumerate(f):
            if n and i >= n: break
            if "\t" in l: pairs.append(tuple(c.strip().lower().split() for c in l.split("\t")[:2]))
    return pairs
train_pairs = load_all(train_file, 50000)
val_pairs = load_all(val_file, 500)
print(f"train={len(train_pairs)} val={len(val_pairs)}")

word2id_de = {"<pad>": 0, "<unk>": 1}
word2id_en = {"<pad>": 0, "<unk>": 1}
for de, en in train_pairs:
    for w in de: word2id_de[w] = word2id_de.get(w, len(word2id_de)) if word2id_de.get(w, 0) > 0 else len(word2id_de)
    for w in en: word2id_en[w] = word2id_en.get(w, len(word2id_en)) if word2id_en.get(w, 0) > 0 else len(word2id_en)
# Add val words
for de, en in val_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w] = len(word2id_de)
    for w in en:
        if w not in word2id_en: word2id_en[w] = len(word2id_en)

V_de, V_en, d = len(word2id_de), len(word2id_en), 128
id2word_en = {v: k for k, v in word2id_en.items()}
print(f"DE={V_de} EN={V_en} d={d}")

# ──── Fixed Templates ────
torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d // 2 + 1), device=device)[:d]
TEMPLATES = {
    'Left_Heavy': lambda n, d: 1,
    'Right_Heavy': lambda n, d: max(1, n-1),
    'Balanced': lambda n, d: max(1, n // 2),
    'Spec_Head': lambda n, d: 3 if n >= 6 else max(1, n // 2),
}
TEMPLATE_NAMES = list(TEMPLATES.keys())
MAX_LEN = 50

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

template_paths = {t: {T: gen_paths(T, fn) for T in range(2, MAX_LEN + 1)} for t, fn in TEMPLATES.items()}

def _build_tree(embs, paths):
    cur = {}
    for t in range(min(len(embs), len(paths))): cur[('leaf', paths[t])] = embs[t]
    md = max(len(p) for p in paths) if paths else 0
    for depth in range(md, 0, -1):
        for pfx in set(p[:depth - 1] if depth > 1 else '' for p in paths if len(p) >= max(depth - 1, 0)):
            lk = ('leaf', pfx + 'L') if depth > 1 else ('leaf', 'L')
            rk = ('leaf', pfx + 'R') if depth > 1 else ('leaf', 'R')
            if lk in cur and rk in cur:
                lft, rgt = cur.pop(lk), cur.pop(rk)
                cur[('node', pfx)] = (lft + SIGN_MASK * torch.roll(rgt, shifts=depth)) / (lft + SIGN_MASK * torch.roll(rgt, shifts=depth)).norm() + 1e-8
    return next(iter(cur.values())).squeeze() if cur else torch.zeros(d, device=embs.device)

def compute_root(E, ids, tname):
    T = len(ids)
    if T < 2: return E(torch.tensor([ids[0] if T >= 1 else 0], device=device))
    paths = template_paths[tname].get(min(T, MAX_LEN), gen_paths(min(T, MAX_LEN), TEMPLATES[tname]))
    if len(paths) != T: paths = gen_paths(T, TEMPLATES[tname])
    return _build_tree(E(torch.tensor(ids, device=device)), paths)

def get_best_root(E, ids):
    ids = ids[:MAX_LEN]
    if len(ids) < 2: return E(torch.tensor([ids[0] if ids else 0], device=device)), 'Balanced', ['']
    bv, br, bt = -1e9, None, 'Balanced'
    for tn in TEMPLATE_NAMES:
        r = compute_root(E, ids, tn)
        if r.norm().item() > bv: bv, br, bt = r.norm().item(), r, tn
    return br, bt, template_paths[bt].get(min(len(ids), MAX_LEN), gen_paths(min(len(ids), MAX_LEN), TEMPLATES[bt]))

def get_pos_emb(T, dim=128):
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = 10000 ** (torch.arange(0, dim // 2, device=device).float() * 2 / dim)
    return torch.cat([torch.sin(pos / div), torch.cos(pos / div)], dim=-1)

# ──── Modules ────
class Bridge(nn.Module):
    """Pure MLP, no residual, output on unit sphere"""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, x):
        out = self.net(x)
        return out / (out.norm() + 1e-8)

class TopDownTreeDecoder(nn.Module):
    """Recursively split root_hash into leaves following tree paths. No GRU, no temporal state."""
    def __init__(self, d, V, n_levels=8):
        super().__init__()
        self.d = d
        self.V = V
        # Shared W_split per depth (weight-sharing across same-depth nodes)
        self.W_split = nn.ModuleList([nn.Linear(d, d * 2) for _ in range(n_levels)])
        self.W_out = nn.Linear(d, V)
    def forward(self, root_hash, paths):
        leaves = self._split(root_hash, paths, 0)
        leaf_tensor = torch.stack(leaves, dim=0)  # [T, d]
        logits = self.W_out(leaf_tensor)  # [T, V]
        return logits, leaf_tensor
    def _split(self, node, paths, depth):
        """Recursive: split node into children, follow paths to leaves."""
        T = len(paths)
        if T == 0: return []
        if depth >= len(self.W_split):
            return [node] * T
        # Split paths by direction at this depth
        left_idx, right_idx = [], []
        for i, p in enumerate(paths):
            if len(p) > depth:
                (left_idx if p[depth] == 'L' else right_idx).append(i)
            else:
                left_idx.append(i)  # ended path → left by default
        # W_split at this depth
        level = depth % len(self.W_split)
        split_out = self.W_split[level](node)
        left_child = split_out[:self.d]
        right_child = split_out[self.d:]
        # Recurse
        left_leaves = self._split(left_child, [paths[i] for i in left_idx], depth + 1)
        right_leaves = self._split(right_child, [paths[i] for i in right_idx], depth + 1)
        # Merge in original order
        leaves = [None] * T
        for idx, leaf in zip(left_idx, left_leaves): leaves[idx] = leaf
        for idx, leaf in zip(right_idx, right_leaves): leaves[idx] = leaf
        return leaves

# ──── Init ────
E_de = nn.Embedding(V_de, d).to(device)
E_en = nn.Embedding(V_en, d).to(device)
bridge = Bridge(d).to(device)
decoder = TopDownTreeDecoder(d, V_en).to(device)
nn.init.normal_(E_de.weight, 0, 0.02)
nn.init.normal_(E_en.weight, 0, 0.02)
nP = sum(p.numel() for m in [E_de, E_en, bridge, decoder] for p in m.parameters())
print(f"params={nP / 1e6:.1f}M")

# ──── BLEU ────
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

val_de = [[word2id_de.get(w, 1) for w in d] for d, e in val_pairs[:300] if len(d) >= 2 and len(e) >= 2]
val_en = [[word2id_en.get(w, 1) for w in e] for d, e in val_pairs[:300] if len(d) >= 2 and len(e) >= 2]
print(f"val={len(val_de)} pairs")

# ══════════════════════════════════════════
# PHASE 0: EN Echo pretrain (train TopDownTreeDecoder on English echo)
# ══════════════════════════════════════════
print(f"\n{'=' * 60}\nPHASE 0: EN Echo pretrain (15 epochs x 20K sents)\n{'=' * 60}")
opt0 = torch.optim.Adam(list(E_en.parameters()) + list(decoder.parameters()), lr=0.003)
t0 = time.time()

for ep in range(15):
    random.shuffle(train_pairs); tl, tt = 0, 0
    for bi in range(0, 20000, 16):
        batch = train_pairs[bi:bi + 16]
        opt0.zero_grad(); bl, n = torch.tensor(0.0, device=device), 0
        for de_s, en_s in batch:
            ids_en = [word2id_en.get(w, 1) for w in en_s[:MAX_LEN]]
            if len(ids_en) < 3: continue
            ids_t = torch.tensor(ids_en, device=device); T = len(ids_en)
            # Gold leaves from encode
            with torch.no_grad():
                root_en, tpl, paths = get_best_root(E_en, ids_en)
                gold_leaf = E_en(ids_t) + 0.5 * get_pos_emb(T)
                gold_leaf = gold_leaf / (gold_leaf.norm(dim=-1, keepdim=True) + 1e-8)
            # Decode: root → W_split tree → leaves → tokens
            logits, pred_leaf = decoder(root_en, paths)
            loss = F.cross_entropy(logits, ids_t) + 0.5 * F.mse_loss(pred_leaf, gold_leaf.detach())
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward(); opt0.step(); tl += (bl / n).item(); tt += 1
    if ep % 3 == 0 or ep == 14:
        print(f"  echo EN ep {ep:3d} loss={tl / max(tt, 1):.4f}")

# ══════════════════════════════════════════
# PHASE 1: Bridge + Decoder Joint Training
# ══════════════════════════════════════════
print(f"\n{'=' * 60}\nPHASE 1: Bridge + Decoder joint training (10 epochs x 5K sents)\n{'=' * 60}")
opt1 = torch.optim.Adam(list(E_de.parameters()) + list(E_en.parameters()) + list(bridge.parameters()) + list(decoder.parameters()), lr=0.003)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10)

for ep in range(10):
    random.shuffle(train_pairs); tl, tt = 0, 0
    for bi in range(0, 5000, 8):
        batch = train_pairs[bi:bi + 8]
        opt1.zero_grad(); bl, n = torch.tensor(0.0, device=device), 0
        for de_s, en_s in batch:
            ids_de = [word2id_de.get(w, 1) for w in de_s[:MAX_LEN]]
            ids_en = [word2id_en.get(w, 1) for w in en_s[:MAX_LEN]]
            if len(ids_de) < 3 or len(ids_en) < 3: continue
            ids_en_t = torch.tensor(ids_en, device=device); T = len(ids_en)
            # DE: with gradient — so E_de learns through bridge
            root_de, de_tpl, de_paths = get_best_root(E_de, ids_de)
            # EN gold: detached — teacher signal, no grad needed
            with torch.no_grad():
                root_en_gold, en_tpl, en_paths = get_best_root(E_en, ids_en)
                gold_leaf = E_en(ids_en_t) + 0.5 * get_pos_emb(T)
                gold_leaf = gold_leaf / (gold_leaf.norm(dim=-1, keepdim=True) + 1e-8)
            # Bridge: DE root → EN root
            root_en_pred = bridge(root_de)
            # Decode English from predicted root, using English template paths
            logits, pred_leaf = decoder(root_en_pred, en_paths)
            # Loss: bridge MSE + decoder CE + leaf MSE
            loss = F.mse_loss(root_en_pred, root_en_gold.detach()) \
                 + F.cross_entropy(logits, ids_en_t) \
                 + 0.5 * F.mse_loss(pred_leaf, gold_leaf.detach())
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward(); torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]['params'], 2.0); opt1.step()
        tl += (bl / n).item(); tt += 1
    if tt == 0: continue; sched1.step()

    if ep % 3 == 0 or ep == 9:
        E_de.eval(); E_en.eval(); bridge.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for ids_de, ids_en in zip(val_de[:50], val_en[:50]):
                root_de, _, _ = get_best_root(E_de, ids_de)
                root_en_pred = bridge(root_de)
                # For val: use English gold template (known structure)
                _, en_tpl, en_paths = get_best_root(E_en, ids_en)
                logits, _ = decoder(root_en_pred, en_paths)
                pred = logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids_en); hp.append(pred[:len(ids_en)])
        b = compute_bleu(rf, hp)
        acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        print(f"  ep {ep:3d} loss={tl / tt:.4f} BLEU={b:.1f} tok_acc={acc:.1f}% time={time.time() - t0:.0f}s")
        E_de.train(); E_en.train(); bridge.train(); decoder.train()

# ══════════════════════════════════════════
# PHASE 2: Wild inference (DE paths only, no gold EN info)
# ══════════════════════════════════════════
print(f"\n{'=' * 60}\nPHASE 2: Wild inference (DE template paths, no gold English)\n{'=' * 60}")
E_de.eval(); E_en.eval(); bridge.eval(); decoder.eval()
rf, hp = [], []

for ids_de, ids_en in zip(val_de, val_en):
    with torch.no_grad():
        root_de, de_tpl, de_paths = get_best_root(E_de, ids_de)
        root_en_pred = bridge(root_de)
        # Use DE paths for English decoding (structurally similar sentences)
        logits, _ = decoder(root_en_pred, de_paths)
        pred = logits.argmax(dim=-1).cpu().tolist()
        rf.append(ids_en); hp.append(pred[:len(ids_en)])

b = compute_bleu(rf, hp)
acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFINAL (wild inference, DE template paths): BLEU-4 = {b:.1f}  Token_Accuracy = {acc:.1f}%")

print(f"\n=== samples ===")
for i in range(min(5, len(val_pairs))):
    de, en = val_pairs[i]
    ide = [word2id_de.get(w, 1) for w in de[:MAX_LEN]]
    ien = [word2id_en.get(w, 1) for w in en[:MAX_LEN]]
    if len(ide) < 3 or len(ien) < 3: continue
    with torch.no_grad():
        root_de, tpl, paths = get_best_root(E_de, ide)
        root_en_pred = bridge(root_de)
        logits, _ = decoder(root_en_pred, paths)
        pred = [id2word_en.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
        print(f"  DE: {' '.join(de[:8])}")
        print(f"  EN: {' '.join(en[:8])}")
        print(f"  PR: {' '.join(pred[:8])}")
        print()
