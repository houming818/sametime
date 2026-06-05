"""L0 translation with ZH-only candidate pool — fix EN cluster self-match."""
import torch, torch.nn.functional as F, sentencepiece as spm, re, math, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()

def is_zh_piece(p): return any('\u4E00' <= c <= '\u9FFF' for c in p)

class BiGRU(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = torch.nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = torch.nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])

L0 = torch.nn.Embedding(V, d).to(device)
ckpt_tree = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)
L0.load_state_dict(ckpt_tree['L0']); L0.eval()

td = 5
t_nodes = torch.nn.ModuleList([torch.nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): torch.nn.init.zeros_(t_nodes[i].weight)
t_merge = torch.nn.Linear(d, d).to(device)
torch.nn.init.eye_(t_merge.weight); torch.nn.init.zeros_(t_merge.bias)

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
    k = 1
    while (1 << (k - 1)) < T: k += 1
    nl = 1 << (k - 1)
    return torch.tensor(ids + [0] * (nl - T), device=device), nl

# Build ZH-only candidate pool
all_zh_ids = torch.arange(V, device=device)
all_zh_w = F.normalize(heap_world(all_zh_ids), dim=-1)
zh_mask = torch.tensor([is_zh_piece(sp.id_to_piece(i)) for i in range(V)], device=device)
zh_only_ids = all_zh_ids[zh_mask]
zh_only_w = all_zh_w[zh_mask]
print(f"ZH-only candidates: {len(zh_only_ids)} / {V}")

MAX_LEN = 50
def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values()); mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bp = min(2.0, math.exp(max(1 - sum(len(r)/max(len(h),1) for r,h in zip(refs,hyps) if len(h)>0)/max(len(refs),1), 0)))
    return math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

# Translate with ZH-only candidates
test_pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3: test_pairs.append((zh_t, en_t))
test_pairs = test_pairs[-2000:]

refs, hyps, ct, tt = [], [], 0, 0
for zh_ids, en_ids in random.sample(test_pairs, 100):
    T = min(len(en_ids), len(zh_ids), MAX_LEN)
    en_ids, zh_ids = en_ids[:T], zh_ids[:T]
    with torch.no_grad():
        en_w = F.normalize(heap_world(torch.tensor(en_ids, device=device)), dim=-1)
        cos = en_w @ zh_only_w.T  # [T, N_zh]
        pred_local = cos.argmax(-1)
        pred = zh_only_ids[pred_local].tolist()
    refs.append(zh_ids); hyps.append(pred)
    ct += sum(p == z for p, z in zip(pred, zh_ids)); tt += T

b = bleu(refs, hyps)
print(f"\n=== ZH-only candidates ===")
print(f"  BLEU: {b:.1f}  Token acc: {100*ct/tt:.1f}%")

print(f"\n=== Samples ===")
for zh_ids, en_ids in random.sample(test_pairs, 5):
    T = min(len(en_ids), len(zh_ids), 30)
    en_ids, zh_ids = en_ids[:T], zh_ids[:T]
    with torch.no_grad():
        en_w = F.normalize(heap_world(torch.tensor(en_ids, device=device)), dim=-1)
        cos = en_w @ zh_only_w.T
        pred = zh_only_ids[cos.argmax(-1)].tolist()
    print(f"  EN: {sp.decode_ids(en_ids)[:70]}")
    print(f"  ZH: {sp.decode_ids(zh_ids)[:70]}")
    print(f"  PR: {sp.decode_ids(pred)[:70]}")
    print()
