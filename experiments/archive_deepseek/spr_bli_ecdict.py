"""
BLI benchmark: ECDICT 5000 train / 1500 test, P@1 eval
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time
import sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)
t0 = time.time()

# ── ECDICT 数据 ──
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)

pairs_all = set()
for en, senses in dict_data.items():
    for zh, pos in senses:
        if pos in ('n','v','a','adj','adv','vi','vt') and 1 <= len(zh) <= 8:
            ei = sp.encode_as_ids(en); zi = sp.encode_as_ids(zh)
            if ok(ei) and ok(zi) and 2 <= len(en) <= 15:
                pairs_all.add((en, zh))
pairs_all = list(pairs_all)
print(f"ECDICT valid pairs: {len(pairs_all)}")

# Sort by frequency (low BPE ID = more common)
pairs_all.sort(key=lambda x: sp.encode_as_ids(x[0])[0])
pairs_all = pairs_all[:8000]  # take top 8K

random.seed(42)
random.shuffle(pairs_all)
N_TRAIN = 5000
N_TEST = min(1500, len(pairs_all) - N_TRAIN)
train_pairs = pairs_all[:N_TRAIN]
test_pairs = pairs_all[N_TRAIN:N_TRAIN + N_TEST]

print(f"Train: {len(train_pairs)}  Test: {len(test_pairs)}")

# ── 模型 ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)

# Init from pretrained ckpt
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

# ── 构建 anchor ──
train_ts = [(en, zh, torch.tensor(sp.encode_as_ids(en), device=device), 
             torch.tensor(sp.encode_as_ids(zh), device=device)) for en, zh in train_pairs]
N = len(train_ts)

zh_world_train = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(train_ts):
        zh_world_train[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# Evaluate baseline (pretrained, before ECDICT training)
en_test_ids = [torch.tensor(sp.encode_as_ids(en), device=device) for en, _ in test_pairs]
zh_test_ids = [torch.tensor(sp.encode_as_ids(zh), device=device) for _, zh in test_pairs]

def eval_p1(en_list, zh_list, zh_mat, Nt):
    with torch.no_grad():
        correct = 0
        for i in range(min(Nt, len(en_list))):
            hw = F.normalize(heap_world(en_list[i]).mean(dim=0), dim=-1)
            sims = (hw.unsqueeze(0) @ zh_mat.T / tau).squeeze(0)
            # Among all ZH anchors, find the EN test pair's expected ZH
            # We need the index of the correct ZH in zh_mat
            # But zh_mat only has train anchors, not test anchors
            # So we compute pairwise cosine between EN test and ZH train
            pass  # This approach is wrong - need to include test ZH in matrix
    return -1

# Better approach: compute pairwise matrix between ALL test EN and ALL train ZH
# P@1 = % of test EN words whose closest ZH (in train) is the correct one
def eval_bli():
    with torch.no_grad():
        correct = 0
        for i, (en_ids, zh_ids) in enumerate(test_pairs[:N_TEST]):
            ei = torch.tensor(sp.encode_as_ids(en_ids), device=device)
            zi = torch.tensor(sp.encode_as_ids(zh_ids), device=device)
            
            hw_en = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            hw_zh = F.normalize(heap_world(zi).mean(dim=0), dim=-1)
            
            # Compute sim between test EN and ALL train ZH
            sims = (hw_en.unsqueeze(0) @ zh_world_train.T / tau).squeeze(0)
            best_train_ai = sims.argmax().item()
            
            # Check if best_train_zh matches the test ZH (by word string)
            if train_pairs[best_train_ai][1] == zh_ids:
                correct += 1
        
        return 100 * correct / N_TEST

p1_before = eval_bli()
print(f"\nP@1 before ECDICT training: {p1_before:.1f}% ({N_TEST} test pairs)")

# ── 训练 ──
trainable = list(L0.parameters()) + list(t_merge.parameters()) + [p for tn in t_nodes for p in tn.parameters()]
opt = torch.optim.Adam(trainable, lr=0.003)

B = 128
for ep in range(50):
    indices = list(range(N))
    random.shuffle(indices)
    for bi in range(0, N, B):
        batch = indices[bi:bi+B]; opt.zero_grad()
        losses = []
        for ai in batch:
            ei = train_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_train.T) / tau
            losses.append(F.cross_entropy(logits, torch.tensor([ai], device=device)))
        if losses:
            (sum(losses)/len(losses)).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    
    if ep % 10 == 0 or ep == 49:
        # Rebuild ZH world matrix
        with torch.no_grad():
            for ai, (_, _, _, zi) in enumerate(train_ts):
                zh_world_train[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)
        p1 = eval_bli()
        print(f"  ep {ep:3d} P@1={p1:.1f}% {time.time()-t0:.0f}s")

print(f"\nP@1: {p1_before:.1f}% → {eval_bli():.1f}%")
print(f"Done in {time.time()-t0:.0f}s")
