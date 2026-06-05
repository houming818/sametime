"""L1 as spatial pointer — L1 context defines WHERE in world space, L0 finds WHAT."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, re, math, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])

L0 = nn.Embedding(V, d).to(device)
L1 = BiGRU(d).to(device)

# Load checkpoints
ckpt_tree = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)
ckpt_ce = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt_tree['L0']); L0.eval()
L1.load_state_dict(ckpt_ce['L1']); L1.eval()

# Build tree
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device)
nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    return w

def heap_world(tok_ids):
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    w = F.normalize(tw_vec(tok_ids), dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

def pad_to_heap(ids, T):
    k = 1; k = 1
    while (1 << (k - 1)) < T: k += 1
    nl = 1 << (k - 1)
    return torch.tensor(ids + [0] * (nl - T), device=device), nl

# Load anchors
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))
valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei, zi))
print(f"Anchor pairs: {len(valid)}")

# L1 spatial projection: EN context → ZH world space target
L1_proj = nn.Linear(d, d).to(device)
nn.init.eye_(L1_proj.weight); nn.init.zeros_(L1_proj.bias)

# Train L1_proj on anchor pairs in sentence context
# Find sentences containing anchor EN words
parallel = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3: parallel.append((zh_t, en_t))

# Build per-anchor EN sentence index
print("Building anchor sentence index...")
anchor_sents = {i: [] for i in range(len(valid))}
for si, (zh_sent, en_sent) in enumerate(parallel[-50000:]):
    if si % 10000 == 0: print(f"  {si}/50000")
    for ai, (ei_list, zi_list) in enumerate(valid):
        if len(anchor_sents[ai]) >= 3: continue
        # Check if EN anchor word (all BPE tokens) appears
        match = False
        for t in range(len(en_sent) - len(ei_list) + 1):
            if en_sent[t:t+len(ei_list)] == ei_list:
                match = True; pos = t; break
        if match and pos < 50:  # cap position to MAX_LEN
            anchor_sents[ai].append((en_sent[:50], pos))

# Train L1_proj
print("Training L1 spatial projection...")
opt = torch.optim.Adam(L1_proj.parameters(), lr=0.01)
for ep in range(100):
    tl = 0.0; n_batch = 0
    for ai in range(len(valid)):
        if len(anchor_sents[ai]) == 0: continue
        en_sent, pos = random.choice(anchor_sents[ai])
        en_ids = torch.tensor(en_sent[:50], device=device)
        zh_ids = torch.tensor(valid[ai][1], device=device)
        T = len(en_ids)
        ids_pad, _ = pad_to_heap(en_ids.tolist(), T)
        with torch.no_grad():
            emb = L0(ids_pad).unsqueeze(0)
            h_ctx = L1.fe(emb).squeeze(0)  # [T, 128]
            zh_world = F.normalize(heap_world(zh_ids[:1]).mean(dim=0), dim=-1) if len(zh_ids) > 1 else F.normalize(heap_world(zh_ids), dim=-1)

        l1_target = L1_proj(h_ctx[pos:pos+1]).squeeze(0)
        zh_w = zh_world.squeeze(0) if zh_world.dim()>1 else zh_world; loss = F.mse_loss(l1_target, zh_w)
        loss.backward(); opt.step(); opt.zero_grad()
        tl += loss.item(); n_batch += 1
    if ep % 25 == 0:
        print(f"  ep {ep:3d} loss={tl/max(n_batch,1):.6f}")

# Test translation
print("\nEvaluating...")
all_zh_ids = torch.arange(V, device=device)
all_zh_w = F.normalize(heap_world(all_zh_ids), dim=-1)
# ZH-only filter
is_zh = lambda p: any('\u4E00' <= c <= '\u9FFF' for c in p)
zh_mask = torch.tensor([is_zh(sp.id_to_piece(i)) for i in range(V)], device=device)
zh_only = all_zh_ids[zh_mask]
zh_only_w = all_zh_w[zh_mask]

MAX_LEN = 50
def bleu(refs, hyps):
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
def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]

test_pairs = parallel[-2000:]
refs, hyps, ct, tt = [], [], 0, 0
for alphas in [[0.0], [0.1, 0.3, 0.5, 1.0]]:
    if isinstance(alphas, list):
        for alpha in alphas:
            refs, hyps, ct, tt = [], [], 0, 0
            for zh_sent, en_sent in random.sample(test_pairs, 50):
                T = min(len(en_sent), len(zh_sent), MAX_LEN)
                en_ids = en_sent[:T]; zh_ids = zh_sent[:T]
                ids_pad, _ = pad_to_heap(en_ids, T)
                with torch.no_grad():
                    emb = L0(ids_pad).unsqueeze(0)
                    h_ctx = L1.fe(emb).squeeze(0)[:T]
                    l0_w = F.normalize(heap_world(ids_pad[:T]), dim=-1)
                    l1_target = F.normalize(L1_proj(h_ctx), dim=-1)
                    search = F.normalize(l0_w + alpha * l1_target, dim=-1)
                    cos = search @ zh_only_w.T
                    pred = zh_only[cos.argmax(-1)].tolist()
                refs.append(zh_ids); hyps.append(pred)
                ct += sum(p == z for p, z in zip(pred, zh_ids)); tt += T
            print(f"  L1_spatial α={alpha:.1f}: BLEU={bleu(refs, hyps):.1f} tok_acc={100*ct/tt:.1f}%")

# Show samples
print(f"\n=== Samples (α=0.3) ===")
for zh_sent, en_sent in random.sample(test_pairs, 3):
    T = min(len(en_sent), len(zh_sent), 30)
    en_ids = en_sent[:T]; zh_ids = zh_sent[:T]
    ids_pad, _ = pad_to_heap(en_ids, T)
    with torch.no_grad():
        emb = L0(ids_pad).unsqueeze(0)
        h_ctx = L1.fe(emb).squeeze(0)[:T]
        l0_w = F.normalize(heap_world(ids_pad[:T]), dim=-1)
        l1_target = F.normalize(L1_proj(h_ctx), dim=-1)
        search = F.normalize(l0_w + 0.3 * l1_target, dim=-1)
        cos = search @ zh_only_w.T
        pred = zh_only[cos.argmax(-1)].tolist()
    print(f"  EN: {sp.decode_ids(en_ids)[:80]}")
    print(f"  ZH: {sp.decode_ids(zh_ids)[:80]}")
    print(f"  PR: {sp.decode_ids(pred)[:80]}")
    print()
