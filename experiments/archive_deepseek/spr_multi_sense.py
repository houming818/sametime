"""
Multi-sense anchor training — 同一 EN 词走不同树路径到不同 ZH 词
Test words: light(光/轻), spring(春/泉), right(正确/权利)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, random, time
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; MAX_LEN = 50
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

# ── Model ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
t_merge = nn.Linear(d, d).to(device)

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
L1 = BiGRU(d).to(device)

# Load checkpoints
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
ckpt_ce = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device, weights_only=True)
enc_sd = {k: v for k, v in ckpt_ce['L1'].items() if k.startswith('enc') or k.startswith('ep')}
L1.load_state_dict(enc_sd, strict=False)
print(f"Loaded L0+tree+L1 ckpts")

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
    return t_merge(torch.cat([tL * wL - tR * wR, tL * wR + tR * wL], -1))

# ── Context routing heads ──
ctx_heads = nn.ModuleList([nn.Linear(d, 2 ** i).to(device) for i in range(td)])
for h in ctx_heads: nn.init.normal_(h.weight, 0, 0.01); nn.init.zeros_(h.bias)

def ctx_path_vec(h_ctx):
    w = torch.zeros(d, device=device)
    for l in range(td):
        attn = F.softmax(ctx_heads[l](h_ctx), dim=-1)
        w = w + attn @ t_nodes[l](torch.arange(2 ** l, device=device))
    return F.normalize(w, dim=-1)

def cmul(a, b):
    aL, aR = a[..., :d // 2], a[..., d // 2:]
    bL, bR = b[..., :d // 2], b[..., d // 2:]
    return torch.cat([aL * bL - aR * bR, aL * bR + aR * bL], -1)

def context_world(tok_id, h_ctx):
    base = F.normalize(L0.weight[tok_id], dim=-1)
    cp = ctx_path_vec(h_ctx)
    return cmul(base, cp)

# ── Build multi-sense anchor data ──
MULTI = {
    'light': [('光', 'noun', ['turn', 'shine', 'bulb', 'sun', 'bright', 'dark', 'blue', 'red', 'green', 'flashing']),
               ('轻', 'adj', ['weight', 'heavy', 'breeze', 'touch', 'soft', 'thin', 'small'])],
    'spring': [('春', 'season', ['summer', 'season', 'bloom', 'flower', 'rain', 'march', 'april', 'may', 'festival']),
               ('泉水', 'water', ['water', 'river', 'stream', 'drink', 'fountain', 'hot', 'mineral'])],
    'right': [('正确', 'correct', ['correct', 'answer', 'wrong', 'true', 'direction', 'decision', 'way', 'side']),
              ('权利', 'rights', ['freedom', 'human', 'law', 'legal', 'constitution', 'vote', 'equal'])],
}

print("\n=== Multi-sense anchors ===")
MULTI_FLAT = []  # [('light','光','noun'), ...]
multi_anchors = []
for en_word, senses in MULTI.items():
    en_ids = sp.encode_as_ids(en_word)
    if not ok(en_ids):
        continue
    for zh_word, label, keywords in senses:
        zh_ids = sp.encode_as_ids(zh_word)
        if not ok(zh_ids):
            continue
        multi_anchors.append((en_word, zh_word, en_ids, zh_ids, label, keywords))
        MULTI_FLAT.append((en_word, zh_word, label, keywords))
        print(f"  {en_word:10s} → {zh_word:6s} [{label}]")

# ── Load single-sense anchors (LaBSE + manual) ──
with open('/workspace/labse_anchors.py') as f:
    labse_pairs = eval(f.read().split('=', 1)[1].strip())

MANUAL_WORDS = [
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),
    ('we','我们'),('they','他们'),('me','我'),('him','他'),
    ('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),
    ('hand','手'),('eye','眼'),('head','头'),('heart','心'),
    ('water','水'),('fire','火'),('sun','太阳'),('moon','月'),
    ('mountain','山'),('river','河'),('sky','天'),('earth','地'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('wood','木'),('stone','石'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),
    ('fish','鱼'),('bird','鸟'),('horse','马'),('cow','牛'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),
    ('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),
    ('milk','牛奶'),('egg','蛋'),('fruit','水果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),
    ('spring','春'),('summer','夏'),('autumn','秋'),('winter','冬'),
    ('go','去'),('come','来'),('eat','吃'),('drink','喝'),
    ('see','看'),('hear','听'),('say','说'),('read','读'),
    ('write','写'),('walk','走'),('sit','坐'),('stand','站'),
    ('give','给'),('take','拿'),('make','做'),('know','知道'),
    ('think','想'),('want','要'),('love','爱'),('like','喜欢'),
    ('work','工作'),('play','玩'),('buy','买'),('sell','卖'),
    ('live','生活'),('die','死'),('kill','杀'),('speak','说'),
    ('talk','谈'),('find','找'),('learn','学'),('teach','教'),
    ('help','帮助'),('open','开'),('close','关'),('start','开始'),
    ('stop','停'),('run','跑'),('fly','飞'),
    ('remember','记住'),('forget','忘记'),('believe','相信'),
    ('understand','理解'),('explain','解释'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),
    ('push','推'),('pull','拉'),('carry','带'),('throw','扔'),
    ('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('need','需要'),('keep','保持'),
    ('agree','同意'),('return','返回'),('leave','离开'),
    ('arrive','到达'),('stay','停留'),('move','移动'),
    ('pass','通过'),('rise','上升'),('fall','下降'),('drop','掉'),
    ('lift','举起'),('burn','烧'),('choose','选择'),('decide','决定'),
    ('prepare','准备'),('finish','完成'),('enjoy','享受'),
    ('suffer','受苦'),('worry','担心'),('protect','保护'),
    ('attack','攻击'),('defend','防御'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),
    ('manage','管理'),('organize','组织'),('save','保存'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),
    ('happen','发生'),('appear','出现'),('continue','继续'),('reduce','减少'),
    ('increase','增加'),('raise','提高'),('avoid','避免'),('expect','期望'),
    ('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),
    ('fast','快'),('slow','慢'),('beautiful','美'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),
    ('sad','悲伤'),('young','年轻'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),
    ('free','自由'),('fair','公平'),('bright','明亮'),('dark','黑暗'),
    ('dry','干'),('full','满'),('empty','空'),
    ('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
]
manual_anchors = []
for e, z in MANUAL_WORDS:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): manual_anchors.append((e, z, ei, zi))

# Build anchor list: single-sense + multi-sense
anchor_list = []  # all entries in the InfoNCE matrix
multi_en = {m[0] for m in multi_anchors}

# First add LaBSE anchors (str, str)
for e, z in labse_pairs:
    if e in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi):
        anchor_list.append((e, z, torch.tensor(ei, device=device), torch.tensor(zi, device=device)))
# Then add manual anchors (str, str, list, list)
for e, z, ei, zi in manual_anchors:
    if e in multi_en: continue
    anchor_list.append((e, z, torch.tensor(ei, device=device), torch.tensor(zi, device=device)))
# Then add multi-sense anchors
for en_word, zh_word, en_ids, zh_ids, sense_label, keywords in multi_anchors:
    anchor_list.append((en_word, zh_word, torch.tensor(en_ids, device=device), torch.tensor(zh_ids, device=device)))

N = len(anchor_list)
print(f"\nAnchor list: {N} entries (multi-sense EN words excluded from single-sense)")

# Precompute ZH world positions
zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (en_w, zh_w, ei, zi) in enumerate(anchor_list):
        ei_t = ei.to(device) if ei.device != device else ei
        zi_t = zi.to(device) if zi.device != device else zi
        zh_world_mat[ai] = F.normalize(heap_world(zi_t).mean(dim=0), dim=-1)

# Build EN_name → anchor_indices
en_idx_map = defaultdict(list)
for ai, (en_w, zh_w, ei, zi) in enumerate(anchor_list):
    en_idx_map[en_w].append(ai)

print(f"Multi-sense EN words in anchor list:")
for en in multi_en:
    idxs = en_idx_map[en]
    zh_words = [anchor_list[i][1] for i in idxs]
    print(f"  {en}: indices={idxs} → {zh_words}")

# ── Build sentence context data ──
print("\n=== Building sentence contexts ===")
pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' not in l: continue
        zh, en = l.strip().split('\t', 1)
        zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
        if len(zh_t) >= 3 and len(en_t) >= 3:
            pairs.append((zh_t, en_t))
print(f"  parallel pairs: {len(pairs)}")

# Look for sentences containing our multi-sense EN words — sense determined by context keywords
def pad_sent(ids):
    T = min(len(ids), MAX_LEN)
    return torch.tensor(ids[:T], device=device), T

def detect_sense(en_sent, pos, en_ids_bpe, keywords):
    """Check if any keyword appears within ±5 tokens of the EN word position"""
    sent_str = sp.decode(en_sent[max(0, pos-5):pos+len(en_ids_bpe)+5])
    sent_str_lower = sent_str.lower()
    for kw in keywords:
        if kw in sent_str_lower:
            return True
    return False

train_examples = []

for en_word, senses in MULTI.items():
    idxs = en_idx_map[en_word]
    en_ids_bpe = sp.encode_as_ids(en_word)
    found = {ai: 0 for ai in idxs}
    sense_keywords = {ai: kw for ai, (_, _, _, _, _, kw) in zip(idxs, multi_anchors) if multi_anchors[idxs.index(ai)][0] == en_word}
    # Rebuild: ai → keywords mapping
    sense_kw = {}
    for ai in idxs:
        for m in multi_anchors:
            if m[0] == en_word and anchor_list[ai][1] == m[1]:
                sense_kw[ai] = m[5]
    
    for zh_s, en_s in pairs[-50000:]:
        all_found = all(c >= 5 for c in found.values())
        if all_found: break
        for t, tok in enumerate(en_s[:MAX_LEN]):
            if tok != en_ids_bpe[0]: continue
            if t + len(en_ids_bpe) <= len(en_s) and en_s[t:t + len(en_ids_bpe)] == en_ids_bpe:
                # Determine sense by keyword match
                matched_ai = None
                best_score = -1
                for ai in idxs:
                    if found[ai] >= 5: continue
                    kws = sense_kw.get(ai, [])
                    score = sum(1 for kw in kws if kw in sp.decode(en_s[max(0,t-5):t+len(en_ids_bpe)+5]).lower())
                    if score > best_score and score > 0:
                        best_score = score
                        matched_ai = ai
                if matched_ai is not None:
                    found[matched_ai] += 1
                    train_examples.append((en_s[:MAX_LEN], t, matched_ai))
                break

print(f"  training examples: {len(train_examples)}")
for en_word in multi_en:
    idxs = en_idx_map[en_word]
    for ai in idxs:
        c = sum(1 for _, _, tgt in train_examples if tgt == ai)
        print(f"    {anchor_list[ai][0]}→{anchor_list[ai][1]}: {c} examples")

# ── Freeze L0 + merge ──
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False

# We train: ctx_heads only (t_nodes frozen for now)
trainable = list(ctx_heads.parameters())
opt = torch.optim.Adam(trainable, lr=0.003)
tau = 0.07

# ── Baseline evaluation ──
print("\n=== Baseline ===")
with torch.no_grad():
    L1.eval()
    for h in ctx_heads: h.eval()
    for en_word in multi_en:
        idxs = en_idx_map[en_word]
        ei_t = torch.tensor(sp.encode_as_ids(en_word), device=device)
        hw_en = F.normalize(heap_world(ei_t).mean(dim=0), dim=-1)
        logits = (hw_en.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
        print(f"  {en_word}:")
        for ai in idxs:
            zh_word = anchor_list[ai][1]
            rank = (logits > logits[ai]).sum().item() + 1
            print(f"    → {zh_word:8s} rank={rank}/{N}  cos_with_target={logits[ai].item():.4f}")

# ── Multi-sense InfoNCE training ──
print(f"\n=== Training ({len(train_examples)} examples) ===")
EPOCHS = 200
for ep in range(EPOCHS):
    L1.train()
    for h in ctx_heads: h.train()
    random.shuffle(train_examples)
    tl, ti = 0.0, 0
    for bi in range(0, len(train_examples), 8):
        batch = train_examples[bi:bi + 8]
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        n = 0
        for en_sent, pos, target_ai in batch:
            ids_t, T = pad_sent(en_sent)
            L = min(pos, T - 1)
            with torch.no_grad():
                emb = L0(ids_t).unsqueeze(0)
                h_ctx = L1.fe(emb).squeeze(0)
            h_p = h_ctx[L]
            cw = context_world(ids_t[L], h_p)
            cw_n = F.normalize(cw, dim=-1)
            logits = (cw_n.unsqueeze(0) @ zh_world_mat.T) / tau
            batch_loss += F.cross_entropy(logits, torch.tensor([target_ai], device=device))
            n += 1
        if n > 0:
            (batch_loss / n).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tl += (batch_loss / n).item(); ti += 1

    if ep % 20 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            L1.eval()
            for h in ctx_heads: h.eval()
            # Compute multi-sense disambiguation accuracy
            correct_by_sense = defaultdict(lambda: [0, 0])
            for en_sent, pos, target_ai in train_examples[:200]:
                en_word = anchor_list[target_ai][0]
                ids_t, T = pad_sent(en_sent)
                L = min(pos, T - 1)
                emb = L0(ids_t).unsqueeze(0)
                h_ctx = L1.fe(emb).squeeze(0)
                h_p = h_ctx[L]
                cw = F.normalize(context_world(ids_t[L], h_p), dim=-1)
                logits = (cw.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
                pred = logits.argmax().item()
                correct_by_sense[en_word][0] += 1
                correct_by_sense[en_word][1] += (1 if pred == target_ai else 0)
        parts = [f"loss={tl/max(ti,1):.4f}"]
        for en_word, (total, correct) in correct_by_sense.items():
            parts.append(f"{en_word}={100*correct/total:.0f}%")
        print(f"  ep {ep:3d} {' '.join(parts)}")

# ── Final disambiguation eval ──
print(f"\n=== Final disambiguation ===")
with torch.no_grad():
    L1.eval()
    for h in ctx_heads: h.eval()
    for en_word in multi_en:
        idxs = en_idx_map[en_word]
        print(f"  {en_word}:")
        hw_en = F.normalize(heap_world(torch.tensor(sp.encode_as_ids(en_word), device=device)).mean(dim=0), dim=-1)
        logits_l0 = (hw_en.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
        for ai in idxs:
            zh_word = anchor_list[ai][1]
            rank_l0 = (logits_l0 > logits_l0[ai]).sum().item() + 1
            print(f"    → {zh_word:8s}  L0_rank={rank_l0}/{N}")

print("\nDone.")
