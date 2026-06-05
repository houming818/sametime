"""
SPR v3 Path-Aware Decoder — Break leaf homogeneity by injecting path prefix
Each split sees WHERE it is in the tree: "LLR" ≠ "RRL" → unique leaf vectors
Encoder: 4 fixed templates, Gumbel-Softmax selection (0 param)
Decoder: PathAwareSplit(depth, path_prefix) → left, right
Loss: internal MSE + leaf MSE + leaf dot E → CE
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sys
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lang = sys.argv[1] if len(sys.argv) > 1 else 'en'
col = 0 if lang == 'de' else 1
print(f"device={device}  SPR v3 PATH-AWARE [{lang.upper()}] col={col}")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t")[col].strip().lower().split())
    return sents

print("loading...")
train_sents = load_sents(train_file, 450000)  # 10% of WMT14
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
print(f"vocab={V} d={d}")

# ──── Templates ────
torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d // 2 + 1), device=device)[:d]
TEMPLATES = {'Left_Heavy': lambda n, d: 1, 'Right_Heavy': lambda n, d: max(1, n - 1),
             'Balanced': lambda n, d: max(1, n // 2), 'Spec_Head': lambda n, d: 3 if n >= 6 else max(1, n // 2)}
TEMPLATE_NAMES = list(TEMPLATES.keys())
MAX_TREE_LEN = 50

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

template_paths = {t: {T: gen_paths(T, fn) for T in range(2, MAX_TREE_LEN + 1)} for t, fn in TEMPLATES.items()}

# ──── Encoder (0 param) ────
def encode(E, ids, tau=0.5):
    T = len(ids)
    if T < 2:
        return E(torch.tensor([ids[0] if T>=1 else 0], device=device)), 'Balanced', [''], {}
    k = min(T, MAX_TREE_LEN)
    root_norms = []
    all_roots = []
    for tn in TEMPLATE_NAMES:
        paths = template_paths[tn].get(k, gen_paths(k, TEMPLATES[tn]))
        if len(paths) != T: paths = gen_paths(T, TEMPLATES[tn])
        root, table = _build_tree(E, ids, paths)
        all_roots.append(root)
        root_norms.append(root.norm().detach())
    logits = torch.stack(root_norms) / tau
    weights = F.gumbel_softmax(logits, tau=tau, hard=True)
    best_idx = weights.argmax().item()
    best_tpl = TEMPLATE_NAMES[best_idx]
    best_paths = template_paths[best_tpl].get(k, gen_paths(k, TEMPLATES[best_tpl]))
    if len(best_paths) != T and k != T:
        best_paths = gen_paths(T, TEMPLATES[best_tpl])
    root, table = _build_tree(E, ids, best_paths)
    blended_root = sum(w * r for w, r in zip(F.softmax(logits / tau, dim=0), all_roots))
    return blended_root, best_tpl, best_paths, table

def _build_tree(E, ids, paths):
    ids_t = torch.tensor(ids, device=device)
    embs = E(ids_t)
    T = len(embs)
    cur = {('leaf', paths[t]): embs[t] for t in range(min(T, len(paths)))}
    table = {}
    md = max(len(p) for p in paths) if paths else 0
    for depth in range(md, 0, -1):
        table[depth] = []
        for pfx in set(p[:depth - 1] if depth > 1 else '' for p in paths if len(p) >= max(depth - 1, 0)):
            lk = ('leaf', pfx + 'L') if depth > 1 else ('leaf', 'L')
            rk = ('leaf', pfx + 'R') if depth > 1 else ('leaf', 'R')
            if lk in cur and rk in cur:
                lft, rgt = cur.pop(lk), cur.pop(rk)
                merged = lft + SIGN_MASK * torch.roll(rgt, shifts=depth)
                merged = merged / (merged.norm() + 1e-8)
                cur[('node', pfx)] = merged
                table[depth].append((merged, lft, rgt))
    root = next(iter(cur.values())).squeeze() if cur else torch.zeros(d, device=embs.device)
    return root, table


# ──── Path-Aware Decoder ────
class PathAwareSplit(nn.Module):
    def __init__(self, d, max_depth=60):
        super().__init__()
        self.depth_enc = nn.Embedding(max_depth, d)
        self.dir_L = nn.Embedding(max_depth, d)
        self.dir_R = nn.Embedding(max_depth, d)
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d * 2))
    def forward(self, node, depth, path_prefix):
        dp = torch.tensor(min(depth, self.depth_enc.weight.shape[0] - 1), device=node.device)
        d_emb = self.depth_enc(dp)
        path_emb = torch.zeros(node.shape[-1], device=node.device)
        max_p = self.dir_L.weight.shape[0]
        for pos, ch in enumerate(path_prefix):
            sidx = torch.tensor(min(pos, max_p - 1), device=node.device)
            path_emb += self.dir_L(sidx) if ch == 'L' else self.dir_R(sidx)
        node = node + d_emb + path_emb / max(len(path_prefix), 1)
        out = self.net(node)
        left = out[:node.shape[-1]] / (out[:node.shape[-1]].norm() + 1e-8)
        right = out[node.shape[-1]:] / (out[node.shape[-1]:].norm() + 1e-8)
        return left, right


class PathAwareDecoder(nn.Module):
    def __init__(self, d, V, n_splits=3, max_depth=60):
        super().__init__()
        self.d = d; self.max_depth = max_depth
        self.splits = nn.ModuleList([PathAwareSplit(d, max_depth) for _ in range(n_splits)])
        # No W_out — leaves directly dot E.weight for token recall
    def forward(self, root, E_weight, paths, node_table):
        T = len(paths)
        pred_leaves = [None] * T
        layer_losses = []
        def split_recurse(node, idxs, depth, path_prefix):
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
            split = self.splits[depth % len(self.splits)]
            left_child, right_child = split(node, depth, path_prefix)
            if depth + 1 in node_table:
                step_loss = torch.tensor(0.0, device=root.device); cnt = 0
                for ph, lh, rh in node_table[depth + 1]:
                    step_loss += F.mse_loss(left_child, lh) + F.mse_loss(right_child, rh)
                    cnt += 1
                if cnt > 0: layer_losses.append(step_loss / cnt)
            split_recurse(left_child, left_idx, depth + 1, path_prefix + 'L')
            split_recurse(right_child, right_idx, depth + 1, path_prefix + 'R')
        split_recurse(root, list(range(T)), 0, '')
        loss_internal = sum(layer_losses) / len(layer_losses) if layer_losses else torch.tensor(0.0, device=root.device)
        for i in range(T):
            if pred_leaves[i] is None: pred_leaves[i] = torch.zeros(d, device=root.device)
        leaves = torch.stack(pred_leaves, dim=0)
        logits = leaves @ E_weight.T  # dot product, no classifier
        return logits, leaves, loss_internal


# ──── Init ────
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
decoder = PathAwareDecoder(d, V).to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
print(f"params={sum(p.numel() for m in [E, decoder] for p in m.parameters())/1e6:.1f}M")

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

# ──── Train ────
print(f"\n{'='*60}")
print(f"Training Path-Aware + Contrastive Echo [{lang.upper()}]")
print(f"  epochs=30 batch=16 lr=0.003 train=50K")
t0 = time.time()
tau=1.0

for epoch in range(30):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl_ce, tl_mse, tl_ct, tt = 0, 0, 0, 0, 0
    tau = max(0.1, 1.0 - epoch / 20.0)

    for bi in range(0, 50000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl_ce, bl_mse, bl_ct, n = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            T = min(len(ids), MAX_LEN); ids = ids[:T]
            ids_t = torch.tensor(ids, device=device)

            root, tpl, paths, table = encode(E, ids, tau)
            logits, leaves, loss_node = decoder(root, E.weight, paths, table)
            loss_ce = F.cross_entropy(logits[:T], ids_t)

            # Contrastive: penalize leaf homogeneity (cos>0.5 between different positions)
            leaf_norm = leaves[:T] / (leaves[:T].norm(dim=-1, keepdim=True) + 1e-8)
            cos_mat = leaf_norm @ leaf_norm.T  # [T,T]
            off_mask = ~torch.eye(T, dtype=torch.bool, device=device)
            off_diag = cos_mat[off_mask]
            loss_contrast = F.relu(off_diag - 0.5).mean()

            loss = loss_ce + loss_node + 0.1 * loss_contrast
            bl_ce += loss_ce.item(); bl_mse += loss_node.item(); bl_ct += loss_contrast.item(); n += 1
            loss.backward(retain_graph=False)

        if n == 0: continue
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0); opt.step()
        ti += 1; tl_ce += bl_ce / n; tl_mse += bl_mse / n; tt += 1

    if ti == 0: continue
    avg_ce = tl_ce / ti; avg_mse = tl_mse / ti; avg_ct = tl_ct / ti

    if epoch % 5 == 0 or epoch == 29:
        E.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = min(len(ids), MAX_LEN); ids = ids[:T]
                root, tpl, paths, table = encode(E, ids, tau=0.1)
                logits, leaves, _ = decoder(root, E.weight, paths, table)
                pred = logits[:T].argmax(dim=-1).cpu().tolist()
                rf.append(ids); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))

        # Parameter diagnostics
        E_mu = E.weight.mean().item(); E_sigma = E.weight.std().item()
        W_mus = []
        for sp in decoder.splits:
            for p in sp.net.parameters():
                W_mus.append(p.mean().item())
        avg_W_mu = sum(W_mus) / len(W_mus) if W_mus else 0

        print(f"  ep {epoch:3d} ce={avg_ce:.4f} mse={avg_mse:.4f} ct={avg_ct:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% "
              f"E μ={E_mu:.3f} σ={E_sigma:.3f} W μ={avg_W_mu:.3f} "
              f"τ={tau:.2f} {time.time()-t0:.0f}s")
        E.train(); decoder.train()

# Final
E.eval(); decoder.eval(); rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        T = min(len(ids), MAX_LEN); ids = ids[:T]
        root, tpl, paths, table = encode(E, ids, tau=0.1)
        logits, leaves, _ = decoder(root, E.weight, paths, table)
        pred = logits[:T].argmax(dim=-1).cpu().tolist()
        rf.append(ids); hp.append(pred)
bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s][:MAX_LEN]
    if len(ids) < 3: continue
    with torch.no_grad():
        root, tpl, paths, table = encode(E, ids, tau=0.1)
        logits, leaves, _ = decoder(root, E.weight, paths, table)
        pred = [id2word.get(p, '?') for p in logits[:len(ids)].argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}  [{tpl}]")
    print()
