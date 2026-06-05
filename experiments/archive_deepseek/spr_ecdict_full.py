"""
全量 ECDICT 训练 → 扩展锚点 → 词级准确率
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, re, time
import sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)
t0 = time.time()

# ── L0 ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
for p in L0.parameters(): p.requires_grad = True
for p in t_merge.parameters(): p.requires_grad = True
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = True

def heap_world(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    w = F.normalize(w, dim=-1)
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    tL, tR = t[..., :d//2], t[..., d//2:]
    wL, wR = w[..., :d//2], w[..., d//2:]
    return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

# ── ECDICT 全量锚点 ──
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)

pairs = set()
for en, senses in dict_data.items():
    for zh, pos in senses:
        if pos in ('n','v','a','adj','adv','vi','vt') and 1 <= len(zh) <= 6:
            ei = sp.encode_as_ids(en); zi = sp.encode_as_ids(zh)
            if ok(ei) and ok(zi) and 2 <= len(en) <= 15:
                pairs.add((en, zh))

pairs = list(pairs)
print(f"ECDICT valid pairs: {len(pairs)}")

# Build anchor ts
anchor_ts = []
en_to_zh_map = {}
zh_candidates = []
for en, zh in pairs:
    ei = sp.encode_as_ids(en); zi = sp.encode_as_ids(zh)
    anchor_ts.append((en, zh, torch.tensor(ei, device=device), torch.tensor(zi, device=device)))
    en_to_zh_map[en] = zh
    zh_candidates.append(zh)

N = len(anchor_ts)
print(f"Anchors: {N}")

# ZH world matrix
zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world_mat[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# ── 训练 ──
trainable = list(L0.parameters()) + list(t_merge.parameters()) + [p for tn in t_nodes for p in tn.parameters()]
opt = torch.optim.Adam(trainable, lr=0.003)

print(f"\nTraining {N} anchors...")
B = 128
for ep in range(100):
    indices = list(range(N))
    random.shuffle(indices)
    for bi in range(0, N, B):
        batch_idx = indices[bi:bi+B]; opt.zero_grad()
        losses = []
        for ai in batch_idx:
            ei = anchor_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_mat.T) / tau
            losses.append(F.cross_entropy(logits, torch.tensor([ai], device=device)))
        if losses: (sum(losses)/len(losses)).backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    if ep % 20 == 0 or ep == 99:
        print(f"  ep {ep:3d} {time.time()-t0:.0f}s")

# Rebuild ZH matrix after training
print("Rebuilding ZH world matrix...")
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world_mat[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# ── 词级评测 ──
def model_prediction(en_word):
    ei = sp.encode_as_ids(en_word)
    if not ok(ei): return None
    hw = F.normalize(heap_world(torch.tensor([ei[0]], device=device)).mean(dim=0), dim=-1)
    sims = (hw.unsqueeze(0) @ zh_world_mat.T).squeeze(0)
    return zh_candidates[sims.argmax().item()]

test_sents = [
    ("the sun is hot today",         "太阳很热今天"),
    ("i love to eat bread",          "我爱吃面包"),
    ("he drinks milk every day",     "他每天喝牛奶"),
    ("the river is long and deep",   "河又长又深"),
    ("she gave me a red flower",     "她给了我一朵红花"),
    ("we walk to the big mountain",  "我们走到大山"),
    ("they live in a new house",     "他们住在新房子里"),
    ("my mother bought rice and fish","我母亲买了米和鱼"),
    ("the old man died last night",  "老人昨晚死了"),
    ("i want to see the moon and stars","我想看月亮和星星"),
    ("fire is hot and water is cold","火是热的水是冷的"),
    ("the bird flies in the sky",    "鸟在天上飞"),
    ("he cut the bread with a knife","他用刀切面包"),
    ("she has a strong heart",       "她有一颗坚强的心"),
    ("we drink tea every morning",   "我们每天早上喝茶"),
    ("it is a good day",             "今天是个好天"),
    ("they kill the big fish",       "他们杀大鱼"),
    ("come and see the gold star",   "来看金星"),
    ("the new year is here",         "新年到了"),
    ("love is strong and true",      "爱是坚强而真实的"),
]

total = correct = 0
print(f"\n{'EN':12s} → {'pred':10s} crt | true")
print("-"*40)
for en_s, zh_ref in test_sents:
    for word in en_s.split():
        w = word.strip('.,!?;:()[]"')
        pred = model_prediction(w)
        if pred is None: continue
        total += 1
        hit = pred in zh_ref
        if hit: correct += 1
        true_zh = en_to_zh_map.get(w, '?')
        print(f"{w:12s} → {pred:10s} {'✓' if hit else '✗'} | {true_zh}")

print(f"\n=== ECDICT全量训练后词级准确率 ===")
print(f"  Correct: {correct}/{total} = {100*correct/max(total,1):.1f}%")
print(f"  Total time: {time.time()-t0:.0f}s")
