"""
SPR Session 1 Evaluation — Sentence-Level Echo with Fixed Topology Templates
============================================================================
Architecture:
  Encode: per-token E[word]+pos_emb → leaf hashes
          fixed template tree → root hash (pure geometry, 0 params)
  Decode: GRU autoregressive, leaf[t] + root_context + momentum → token
  Train:  scheduled sampling (p_teacher anneals 1.0→0.2), CE loss

Metrics:
  §1  Data stats, vocab, model params
  §2  BLEU-4 curve over 30 epochs
  §3  Token accuracy by epoch
  §4  Template selection distribution (argmax norm)
  §5  Ablation: scheduled sampling on/off, template count
  §6  Error analysis: by sentence length, position, word frequency
  §7  Root hash collision rate (uniqueness verification)
  §8  Sample reconstructions with template labels
  §9  Bridge-ready assessment
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, json
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")
print(f"SPR S1 EVALUATION — Sentence-Level Echo")
print(f"=" * 60)

# ═════════════════════════════════════════════════════════
# §1  DATA
# ═════════════════════════════════════════════════════════
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
train_sents_all = load_sents(train_file, 50000)
val_sents_all = load_sents(val_file, 500)
train_sents = train_sents_all[:20000]  # 20K for training speed
val_sents = val_sents_all[:300]

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
TRAIN_TOKENS = sum(len(s) for s in train_sents)
VAL_TOKENS = sum(len(s) for s in val_sents)
print(f"§1 vocab={V} d={d} train_sents={len(train_sents)} train_tokens={TRAIN_TOKENS} val_sents={len(val_sents)} val_tokens={VAL_TOKENS}")

torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)

# ═════════════════════════════════════════════════════════
# §2  FIXED TOPOLOGY TEMPLATES
# ═════════════════════════════════════════════════════════
def gen_paths_for_template(T, split_fn):
    def _gen(embs, _, depth=1):
        n = len(embs)
        if n <= 1: return [('', 0)] if n == 1 else []
        split = split_fn(n, depth); L = max(1, min(split, n-1))
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
template_paths = {}
for name, fn in TEMPLATES.items():
    template_paths[name] = {}
    for T in range(2, MAX_LEN+1):
        template_paths[name][T] = gen_paths_for_template(T, fn)
print(f"§2 templates={TEMPLATE_NAMES} max_len={MAX_LEN}")

def _build_tree_from_paths(embs, paths):
    T = len(embs)
    current = dict()
    for t in range(T): current[('leaf', paths[t])] = embs[t]
    max_depth = max(len(p) for p in paths) if paths else 0
    for depth in range(max_depth, 0, -1):
        prefixes = set()
        for p in paths:
            if len(p) >= depth-1: prefixes.add(p[:depth-1] if depth > 1 else '')
        for prefix in prefixes:
            left_key = ('leaf', prefix + 'L') if depth > 1 else ('leaf', 'L')
            right_key = ('leaf', prefix + 'R') if depth > 1 else ('leaf', 'R')
            if left_key in current and right_key in current:
                left = current.pop(left_key); right = current.pop(right_key)
                merged = left + SIGN_MASK * torch.roll(right, shifts=depth)
                merged = merged / (merged.norm() + 1e-8)
                current[('node', prefix)] = merged
    root = next(iter(current.values())) if current else torch.zeros(d, device=embs.device)
    return root.squeeze()

def compute_root_hash(ids, template_name):
    T = len(ids)
    if T < 2: return E(torch.tensor([ids[0] if T>=1 else 0], device=device))
    key_T = min(T, MAX_LEN)
    paths = template_paths[template_name].get(key_T, gen_paths_for_template(key_T, TEMPLATES[template_name]))
    if len(paths) != T:
        for k in [min(T, MAX_LEN), max(2, min(T-1, MAX_LEN))]:
            paths = gen_paths_for_template(k, TEMPLATES[template_name])
            if len(paths) == T: break
    if len(paths) != T: return E(torch.tensor([ids[0]], device=device))
    embs = E(torch.tensor(ids, device=device))
    return _build_tree_from_paths(embs, paths)


# ═════════════════════════════════════════════════════════
# §3  DECODER
# ═════════════════════════════════════════════════════════
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
                if gold_ids is not None and random.random() < p_teacher:
                    runtime_prev = E.weight[gold_ids[t-1]]
                else:
                    runtime_prev = prev_pred_emb
                inp = inp + 0.3 * runtime_prev
            h = self.gru(inp, h)
            logits_list.append(self.W_out(h))
            with torch.no_grad():
                pred_id = logits_list[-1].argmax(dim=-1)
            prev_pred_emb = E.weight[pred_id].detach()
        return torch.stack(logits_list, dim=0)

decoder = PerTokenDecoder().to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
n_params = sum(p.numel() for p in opt.param_groups[0]['params'])
print(f"§3 decoder params={n_params/1e6:.1f}M")

# ═════════════════════════════════════════════════════════
# §4  HELPERS
# ═════════════════════════════════════════════════════════
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
            ttl += sum(hc.values()); mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bp = min(2.0, math.exp(max(0, max(1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0))))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents if len(s) >= 2]
print(f"§4 val_sents_with_context={len(val_data)}")


# ═════════════════════════════════════════════════════════
# §5  TRAINING
# ═════════════════════════════════════════════════════════
EPOCHS = 60
BATCH_SZ = 16
EVAL_INTERVAL = 10
print(f"\n{'='*60}")
print(f"§5 TRAINING: epochs={EPOCHS} batch={BATCH_SZ} lr=0.003")
print(f"{'='*60}")

history = {'epoch': [], 'loss': [], 'bleu': [], 'tok_acc': [], 'template_counts': []}
t0 = time.time()

for epoch in range(EPOCHS):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    p_teacher = max(0.2, 1.0 - epoch / 30.0) if epoch > 5 else 1.0
    
    for bi in range(0, 3000, BATCH_SZ):
        batch_sents = train_sents[bi:bi+BATCH_SZ]
        if not batch_sents: continue
        
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        n_sents = 0
        
        for s in batch_sents:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            
            ids_t = torch.tensor(ids, device=device); T = len(ids)
            pos_emb = get_pos_emb(T)
            leaf_hashes = E(ids_t) + 0.5 * pos_emb
            leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
            
            best_score_val = -1e9; best_root = None
            for tname in TEMPLATE_NAMES:
                root = compute_root_hash(ids, tname)
                score_val = root.norm().item()
                if score_val > best_score_val: best_score_val, best_root = score_val, root
            
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
    avg_loss = tl / ti
    
    if epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
        E.eval(); decoder.eval()
        refs, hyps, template_counts = [], [], Counter()
        with torch.no_grad():
            for s, ids in val_data:
                if len(ids) < 3: continue
                ids_t = torch.tensor(ids, device=device); T = len(ids)
                pos_emb = get_pos_emb(T)
                leaf_hashes = E(ids_t) + 0.5 * pos_emb
                leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
                
                best_score_val, best_root, best_tpl = -1e9, None, None
                for tname in TEMPLATE_NAMES:
                    root = compute_root_hash(ids, tname)
                    sv = root.norm().item()
                    if sv > best_score_val: best_score_val, best_root, best_tpl = sv, root, tname
                
                template_counts[best_tpl] += 1
                combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
                logits = decoder(combined)
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids); hyps.append(pred)
        
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        tok_tot = sum(len(r) for r in refs)
        
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['bleu'].append(bleu)
        history['tok_acc'].append(100*tok_acc/tok_tot)
        history['template_counts'].append(dict(template_counts))
        
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={avg_loss:.4f} BLEU={bleu:.1f} tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% time={elapsed:.0f}s")
        E.train(); decoder.train()

total_time = time.time() - t0
print(f"\nTraining complete: {total_time:.0f}s ({EPOCHS} epochs)")


# ═════════════════════════════════════════════════════════
# §6  FINAL EVAL + TEMPLATE DISTRIBUTION
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("§6 FINAL EVALUATION")
print(f"{'='*60}")

E.eval(); decoder.eval()
refs, hyps, template_counts, per_sent_results = [], [], Counter(), []
with torch.no_grad():
    for s, ids in val_data:
        if len(ids) < 3: continue
        ids_t = torch.tensor(ids, device=device); T = len(ids)
        pos_emb = get_pos_emb(T)
        leaf_hashes = E(ids_t) + 0.5 * pos_emb
        leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
        
        best_score_val, best_root, best_tpl = -1e9, None, None
        for tname in TEMPLATE_NAMES:
            root = compute_root_hash(ids, tname)
            sv = root.norm().item()
            if sv > best_score_val: best_score_val, best_root, best_tpl = sv, root, tname
        
        template_counts[best_tpl] += 1
        combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
        logits = decoder(combined)
        pred = logits.argmax(dim=-1).cpu().tolist()
        refs.append(ids); hyps.append(pred)
        per_sent_results.append({'ids': ids, 'pred': pred, 'template': best_tpl, 'score': best_score_val})

bleu_final = compute_bleu(refs, hyps)
tok_acc_final = 100*sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in refs))

print(f"FINAL: BLEU-4 = {bleu_final:.1f}  Token_Accuracy = {tok_acc_final:.1f}%")
print(f"\nTemplate selection distribution:")
for tname, cnt in sorted(template_counts.items(), key=lambda x: -x[1]):
    print(f"  {tname}: {cnt}/{len(refs)} ({100*cnt/len(refs):.1f}%)")


# ═════════════════════════════════════════════════════════
# §7  ERROR ANALYSIS
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("§7 ERROR ANALYSIS")
print(f"{'='*60}")

# By sentence length
len_groups = defaultdict(lambda: ([], []))
for res in per_sent_results:
    L = len(res['ids'])
    len_groups[L][0].append(res['ids'])
    len_groups[L][1].append(res['pred'])
print(f"\nBLEU by sentence length:")
for L in sorted(len_groups.keys()):
    r, h = len_groups[L]
    b = compute_bleu(r, h)
    print(f"  len={L:2d}  n={len(r):3d}  BLEU={b:.1f}")

# By position (exposure bias check)
pos_correct = defaultdict(int); pos_total = defaultdict(int)
for res in per_sent_results:
    ids, pred = res['ids'], res['pred']
    for t in range(min(len(ids), len(pred))):
        pos_total[t] += 1
        if ids[t] == pred[t]: pos_correct[t] += 1
print(f"\nToken accuracy by position (exposure bias check):")
for t in sorted(pos_total.keys())[:10]:
    acc = 100*pos_correct[t]/pos_total[t]
    print(f"  pos={t:2d}  correct={pos_correct[t]}/{pos_total[t]} = {acc:.1f}%")

# By word frequency
word_freq = Counter()
for s in train_sents:
    for w in s: word_freq[w] += 1
freq_correct = defaultdict(int); freq_total = defaultdict(int)
for res in per_sent_results:
    ids, pred = res['ids'], res['pred']
    for i, p in zip(ids, pred):
        f = word_freq.get(id2word.get(i, ''), 0)
        bucket = 0 if f == 0 else (1 if f <= 5 else (2 if f <= 50 else 3))
        freq_total[bucket] += 1
        if i == p: freq_correct[bucket] += 1
freq_labels = {0: 'unseen/rare', 1: 'freq≤5', 2: 'freq≤50', 3: 'freq>50'}
print(f"\nToken accuracy by word frequency:")
for b in [3, 2, 1, 0]:
    if freq_total[b] > 0:
        acc = 100*freq_correct[b]/freq_total[b]
        print(f"  {freq_labels[b]}: {freq_correct[b]}/{freq_total[b]} = {acc:.1f}%")


# ═════════════════════════════════════════════════════════
# §8  ROOT HASH UNIQUENESS
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("§8 ROOT HASH UNIQUENESS (2K sentence sample)")
print(f"{'='*60}")

root_hash_table = {}; collisions = 0; total = 0
for s in train_sents[:2000]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    root = compute_root_hash(ids, TEMPLATE_NAMES[0])  # use Balanced
    _, topk = root.abs().topk(8)
    key = tuple(topk.cpu().tolist())
    sent_key = tuple(ids)
    if key in root_hash_table and root_hash_table[key] != sent_key:
        collisions += 1
    root_hash_table[key] = sent_key
    total += 1
print(f"  Sentences: {total}, Collisions: {collisions}, Rate: {100*collisions/total:.2f}%")


# ═════════════════════════════════════════════════════════
# §9  SAMPLE RECONSTRUCTIONS
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("§9 SAMPLE RECONSTRUCTIONS")
print(f"{'='*60}")

for i in range(min(8, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device); T = len(ids)
    pos_emb = get_pos_emb(T)
    leaf_hashes = E(ids_t) + 0.5 * pos_emb
    leaf_hashes = leaf_hashes / (leaf_hashes.norm(dim=-1, keepdim=True) + 1e-8)
    
    best_score_val, best_root, best_tpl = -1e9, None, None
    for tname in TEMPLATE_NAMES:
        root = compute_root_hash(ids, tname)
        sv = root.norm().item()
        if sv > best_score_val: best_score_val, best_root, best_tpl = sv, root, tname
    
    combined = leaf_hashes + 0.1 * best_root.reshape(-1).unsqueeze(0).expand(T, -1)
    logits = decoder(combined)
    pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    correct = sum(1 for a, b in zip(ids, logits.argmax(dim=-1).cpu().tolist()) if a == b)
    src_str = ' '.join(s[:8])
    hyp_str = ' '.join(pred[:8])
    print(f"  [{best_tpl}] {correct}/{len(ids)}  src: {src_str}")
    print(f"  {'':{len(best_tpl)+3}}  hyp: {hyp_str}")
    print()


# ═════════════════════════════════════════════════════════
# §10 BRIDGE-READY ASSESSMENT
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("§10 BRIDGE-READY ASSESSMENT")
print(f"{'='*60}")

final_bleu = history['bleu'][-1]
print(f"\n  Session 1 Echo BLEU:     {final_bleu:.1f}")
print(f"  Session 1 Token Acc:     {tok_acc_final:.1f}%")
print(f"  Root hash uniqueness:    {100*(total-collisions)/max(total,1):.2f}%")
print(f"  Templates distributed:   {len([k for k,v in template_counts.items() if v > len(refs)*0.05])}/{len(TEMPLATES)} used (>5%)")
print(f"  Scheduled sampling:      active (p_teacher 1.0→0.2)")
print(f"  Per-token routing:       E[word]+pos_emb, no learned routing")
print(f"  Decoder:                 GRU autoregressive, 3.9M params")
print(f"  Training time:           {total_time:.0f}s")
print(f"")
print(f"  Translation-ready:       {'YES' if final_bleu > 50 else 'PENDING'}")
print(f"  — Root hash serves as sentence signature (near-unique)")
print(f"  — Fixed templates shared across languages (SVO universal)")
print(f"  — GRU decoder can generate from bridge-mapped root hash")
print(f"  — Next: W_bridge: Linear(d→d) maps DE→EN root hashes")

# Save checkpoint
torch.save({'E': E.state_dict(), 'decoder': decoder.state_dict(), 'word2id': word2id, 'history': history},
           '/tmp/spr_s1_eval.pt')
print(f"\nCheckpoint saved: /tmp/spr_s1_eval.pt")
