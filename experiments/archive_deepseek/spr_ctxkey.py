"""Step 1: CMul orthogonality. Step 2: Node path orthogonality.
Step 3: ContextKey routing. Step 4: InfoNCE train. Step 5: Disambiguation."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; MAX_LEN = 50; td = 5
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size(); print(f"vocab={V} device={device}")

def ok(ids): return all(x != 0 for x in ids)

# ── Model ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
t_merge = nn.Linear(d, d).to(device)

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d//2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
L1 = BiGRU(d).to(device)

# Load best L0
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
# Also load L1 from CE pretrain (encoder only)
ckpt_ce = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
enc_sd = {k: v for k, v in ckpt_ce['L1'].items() if k.startswith('enc') or k.startswith('ep')}
missing, unexpected = L1.load_state_dict(enc_sd, strict=False)
print(f"Loaded L0+tree (gold~30%) + L1 enc ({len(enc_sd)} params, missing {len(missing)} keys)")

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    return w

def heap_world(tok_ids):
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    w = F.normalize(tw_vec(tok_ids), dim=-1)
    tL, tR = t[..., :d//2], t[..., d//2:]
    wL, wR = w[..., :d//2], w[..., d//2:]
    return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

def cmul(a, b):
    aL, aR = a[..., :d//2], a[..., d//2:]
    bL, bR = b[..., :d//2], b[..., d//2:]
    return torch.cat([aL*bL - aR*bR, aL*bR + aR*bL], -1)

# ── Step 1: CMul orthogonality ──
print("\n=== Step 1: CMul orthogonality ===")
N_test = 1000
bases  = F.normalize(torch.randn(N_test, d, device=device), dim=-1)
p1s    = F.normalize(torch.randn(N_test, d, device=device), dim=-1)
p2s    = F.normalize(torch.randn(N_test, d, device=device), dim=-1)
cos_raw = (p1s * p2s).sum(-1)
out1 = F.normalize(cmul(bases, p1s), dim=-1)
out2 = F.normalize(cmul(bases, p2s), dim=-1)
cos_cmul = (out1 * out2).sum(-1)
cos_err = (cos_cmul - cos_raw).abs().mean()
print(f"  N={N_test}  mean|cos(cmul)-cos(raw)| = {cos_err:.6f}")
print(f"  cos_raw mean={cos_raw.mean():.6f}  cos_cmul mean={cos_cmul.mean():.6f}")

# Test with real L0 embeddings
test_toks = torch.randint(3, 1000, (200,), device=device)
real_bases = F.normalize(L0(test_toks), dim=-1)
real_p1 = F.normalize(torch.randn(200, d, device=device), dim=-1)
real_p2 = F.normalize(torch.randn(200, d, device=device), dim=-1)
cos_raw_r = (real_p1 * real_p2).sum(-1)
out1_r = F.normalize(cmul(real_bases, real_p1), dim=-1)
out2_r = F.normalize(cmul(real_bases, real_p2), dim=-1)
cos_cmul_r = (out1_r * out2_r).sum(-1)
cos_err_r = (cos_cmul_r - cos_raw_r).abs().mean()
print(f"  Real L0 base: mean|Δcos|={cos_err_r:.6f} 1%ile={torch.quantile((cos_cmul_r-cos_raw_r).abs(), 0.99):.6f}")

# ── Step 2: Node path orthogonality ──
print("\n=== Step 2: Node path orthogonality ===")
def tw_path_vecs(paths):
    w = torch.zeros(len(paths), d, device=device)
    for l in range(td):
        w = w + t_nodes[l](paths[:, l])
    return w

n_sample = 1000
paths = torch.randint(0, 2**(td-1), (n_sample, td), device=device)
for l in range(td):
    paths[:, l] = paths[:, l] % (2**l)
paths[:, 0] = 0
path_vecs = F.normalize(tw_path_vecs(paths), dim=-1)
cos_mat = path_vecs @ path_vecs.T
mask = ~torch.eye(n_sample, dtype=torch.bool, device=device)
off_diag = cos_mat[mask]
print(f"  {n_sample} random paths: mean_off_cos={off_diag.mean():.6f} max={off_diag.max():.6f} min={off_diag.min():.6f}")

# adversarial: differ by exactly one node at one level
single_bit = []
for _ in range(1000):
    base = torch.randint(0, 2**(td-1), (td,), device=device)
    for l in range(td): base[l] = base[l] % (2**l)
    base[0] = 0
    alt = base.clone()
    L = random.randint(1, td-1)
    alt[L] = (alt[L] + 1) % (2**L)
    v1 = F.normalize(tw_path_vecs(base.unsqueeze(0)), dim=-1)
    v2 = F.normalize(tw_path_vecs(alt.unsqueeze(0)), dim=-1)
    single_bit.append((v1 * v2).sum().item())
print(f"  1000 single-bit-diff: mean_cos={np.mean(single_bit):.6f} max={np.max(single_bit):.6f}")

# same prefix, different L4
same_pref = []
for _ in range(1000):
    base = torch.randint(0, 2**(td-1), (td-1,), device=device)
    for l in range(td-1): base[l] = base[l] % (2**l)
    base[0] = 0
    p1 = torch.cat([base, torch.tensor([random.randint(0,15)], device=device)])
    p2 = torch.cat([base, torch.tensor([random.randint(0,15)], device=device)])
    v1 = F.normalize(tw_path_vecs(p1.unsqueeze(0)), dim=-1)
    v2 = F.normalize(tw_path_vecs(p2.unsqueeze(0)), dim=-1)
    same_pref.append((v1 * v2).sum().item())
print(f"  1000 same-prefix-L4-diff: mean_cos={np.mean(same_pref):.6f} max={np.max(same_pref):.6f}")

# ── Step 3: ContextKey routing ──
print("\n=== Step 3: ContextKey routing ===")
ctx_heads = nn.ModuleList([nn.Linear(d, 2**i).to(device) for i in range(td)])
for h in ctx_heads: nn.init.normal_(h.weight, 0, 0.01); nn.init.zeros_(h.bias)
ctx_alpha = nn.Parameter(torch.tensor(0.0, device=device))

def context_path(h_ctx):
    path = []
    for l in range(td):
        attn = F.softmax(ctx_heads[l](h_ctx), dim=-1)
        path.append(attn @ t_nodes[l](torch.arange(2**l, device=device)))
    return torch.stack(path)  # [td, d]

def context_world(tok_id, h_ctx):
    base = F.normalize(L0.weight[tok_id], dim=-1)
    path = context_path(h_ctx)
    delta = F.normalize(path.sum(dim=0), dim=-1) * torch.sigmoid(ctx_alpha)
    return cmul(base, delta)

# Quick orthogonality test
test_h1 = torch.randn(d, device=device)
test_h2 = torch.randn(d, device=device)
cid = sp.encode_as_ids("cat"); tid = cid[0] if ok(cid) else 3
w1 = context_world(torch.tensor(tid, device=device), test_h1)
w2 = context_world(torch.tensor(tid, device=device), test_h2)
cos_ctx = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
print(f"  cos(ctx_world(cat|h1), ctx_world(cat|h2)) = {cos_ctx.item():.6f}")
print(f"  ContextKey params: {sum(p.numel() for p in ctx_heads.parameters())+1}")

# ── Step 4: Build sentence context index ──
print("\n=== Step 4: Sentence context index ===")

# Load LaBSE anchors for training
with open('/workspace/labse_anchors.py') as f:
    labse_pairs = eval(f.read().split('=',1)[1].strip())
train_anchors = []
for e, z in labse_pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): train_anchors.append((torch.tensor(ei, device=device), torch.tensor(zi, device=device)))
N_train = len(train_anchors)
print(f"  train anchors: {N_train}")

# Load manual anchors for gold eval
MANUAL = [
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('we','我们'),('they','他们'),('me','我'),('him','他'),
    ('my','我'),('your','你'),('his','他'),('her','她'),('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),
    ('six','六'),('seven','七'),('eight','八'),('nine','九'),('ten','十'),('hundred','百'),('thousand','千'),
    ('first','第一'),('second','第二'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),('blood','血'),
    ('body','身体'),('mouth','口'),('ear','耳'),('foot','脚'),('arm','手臂'),('leg','腿'),('hair','头发'),
    ('brain','脑'),('tongue','舌'),('neck','颈'),('water','水'),('fire','火'),('sun','太阳'),('moon','月'),
    ('mountain','山'),('river','河'),('sky','天'),('earth','地'),('wind','风'),('rain','雨'),('snow','雪'),
    ('sea','海'),('tree','树'),('flower','花'),('wood','木'),('stone','石'),('gold','金'),('iron','铁'),
    ('star','星'),('cloud','云'),('lake','湖'),('ice','冰'),('sand','沙'),('fish','鱼'),('bird','鸟'),
    ('horse','马'),('cow','牛'),('chicken','鸡'),('dragon','龙'),('bear','熊'),('elephant','象'),('wolf','狼'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),('yellow','黄'),('mother','母亲'),('father','父亲'),
    ('son','儿子'),('daughter','女儿'),('husband','丈夫'),('wife','妻子'),('family','家'),('child','孩子'),
    ('brother','兄弟'),('parent','父母'),('baby','婴儿'),('food','食物'),('rice','米'),('meat','肉'),
    ('bread','面包'),('milk','牛奶'),('egg','蛋'),('fruit','水果'),('sugar','糖'),('salt','盐'),
    ('oil','油'),('wine','酒'),('tea','茶'),('coffee','咖啡'),('apple','苹果'),('day','天'),
    ('night','夜'),('year','年'),('month','月'),('week','周'),('hour','小时'),('time','时间'),
    ('today','今天'),('tomorrow','明天'),('morning','早上'),('evening','晚上'),('spring','春'),
    ('summer','夏'),('autumn','秋'),('winter','冬'),('go','去'),('come','来'),('eat','吃'),
    ('drink','喝'),('see','看'),('hear','听'),('say','说'),('read','读'),('write','写'),
    ('walk','走'),('sit','坐'),('stand','站'),('give','给'),('take','拿'),('make','做'),
    ('know','知道'),('think','想'),('want','要'),('love','爱'),('like','喜欢'),('work','工作'),
    ('play','玩'),('buy','买'),('sell','卖'),('live','生活'),('die','死'),('kill','杀'),
    ('speak','说'),('talk','谈'),('find','找'),('learn','学'),('teach','教'),('help','帮助'),
    ('open','开'),('close','关'),('start','开始'),('stop','停'),('run','跑'),('fly','飞'),
    ('sing','唱'),('fight','战斗'),('win','赢'),('send','送'),('receive','收'),('remember','记住'),
    ('forget','忘记'),('believe','相信'),('understand','理解'),('explain','解释'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),('push','推'),('pull','拉'),
    ('carry','带'),('throw','扔'),('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('hope','希望'),('try','尝试'),('allow','允许'),('need','需要'),('keep','保持'),
    ('agree','同意'),('promise','承诺'),('refuse','拒绝'),('accept','接受'),('return','返回'),
    ('leave','离开'),('enter','进入'),('arrive','到达'),('stay','停留'),('move','移动'),
    ('pass','通过'),('rise','上升'),('fall','下降'),('drop','掉'),('lift','举起'),('burn','烧'),
    ('melt','融化'),('choose','选择'),('decide','决定'),('prepare','准备'),('finish','完成'),
    ('join','加入'),('share','分享'),('divide','分'),('separate','分开'),('enjoy','享受'),
    ('suffer','受苦'),('worry','担心'),('fear','恐惧'),('protect','保护'),('attack','攻击'),
    ('defend','防御'),('destroy','破坏'),('create','创造'),('produce','生产'),('develop','发展'),
    ('improve','改善'),('manage','管理'),('organize','组织'),('save','保存'),('remain','保持'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),('happen','发生'),
    ('appear','出现'),('continue','继续'),('reduce','减少'),('increase','增加'),('raise','提高'),
    ('avoid','避免'),('expect','期望'),('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),('long','长'),('short','短'),
    ('high','高'),('low','低'),('fast','快'),('slow','慢'),('beautiful','美'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('sad','悲伤'),('young','年轻'),
    ('heavy','重'),('light','轻'),('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),('free','自由'),
    ('fair','公平'),('bright','明亮'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),
    ('nice','好'),('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
]
gold_anchors = []
for e, z in MANUAL:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): gold_anchors.append((torch.tensor(ei, device=device), torch.tensor(zi, device=device)))
Ng = len(gold_anchors)
print(f"  gold anchors: {Ng}")

# Precompute ZH world positions
zh_world_train = torch.zeros(N_train, d, device=device)
with torch.no_grad():
    for ai, (ei, zi) in enumerate(train_anchors):
        zh_world_train[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

zh_gold_mat = torch.zeros(Ng, d, device=device)
with torch.no_grad():
    for gi, (ei, zi) in enumerate(gold_anchors):
        zh_gold_mat[gi] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# Scan parallel sentences for anchor contexts
pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' not in l: continue
        zh, en = l.strip().split('\t', 1)
        zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
        if len(zh_t) >= 3 and len(en_t) >= 3: pairs.append((zh_t, en_t))

ank_hash = defaultdict(list)
for ai, (ei, zi) in enumerate(train_anchors):
    ank_hash[ei[0].item()].append(ai)

anchor_ctxs = [[] for _ in range(N_train)]
print("  scanning sentences...")
for si, (zh_s, en_s) in enumerate(pairs[-50000:]):
    if si % 10000 == 0:
        filled = sum(1 for a in anchor_ctxs if len(a) >= 3)
        print(f"    {si}/50000  filled={filled}/{N_train}")
        if filled >= int(N_train * 0.7): break
    for t, tok in enumerate(en_s[:MAX_LEN]):
        if tok not in ank_hash: continue
        for ai in ank_hash[tok]:
            if len(anchor_ctxs[ai]) >= 3: continue
            et = train_anchors[ai][0].tolist()
            if t + len(et) <= len(en_s) and en_s[t:t+len(et)] == et:
                anchor_ctxs[ai].append((en_s[:MAX_LEN], t))

train_data = []
for ai, ctxs in enumerate(anchor_ctxs):
    for en_s, pos in ctxs[:3]:
        train_data.append((ai, en_s, pos))
print(f"  training examples: {len(train_data)}")

def pad_sent(ids):
    T = min(len(ids), MAX_LEN)
    return torch.tensor(ids[:T], device=device), T

# ── Step 5: InfoNCE training ──
print("\n=== Step 5: InfoNCE training ===")
for p in L0.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for p in L1.enc.parameters(): p.requires_grad = False
for p in L1.ep.parameters(): p.requires_grad = False

ctx_params = list(ctx_heads.parameters()) + [ctx_alpha]
opt = torch.optim.Adam(ctx_params, lr=0.003)
tau = 0.07

def eval_gold():
    with torch.no_grad():
        c = 0
        for gi, (ei, zi) in enumerate(gold_anchors):
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_gold_mat.T / tau).squeeze(0)
            if logits.argmax() == gi: c += 1
        return 100 * c / Ng

def eval_nce():
    with torch.no_grad():
        had = [ai for ai in range(N_train) if len(anchor_ctxs[ai]) >= 1]
        c = 0
        for ai in had:
            hw = F.normalize(heap_world(train_anchors[ai][0]).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_train.T / tau).squeeze(0)
            if logits.argmax() == ai: c += 1
        return 100 * c / max(len(had), 1)

gold0 = eval_gold(); nce0 = eval_nce()
print(f"  L0 baseline: gold={gold0:.1f}% nce={nce0:.1f}%")

EPOCHS = 100
for ep in range(EPOCHS):
    L1.train()
    for h in ctx_heads: h.train()
    random.shuffle(train_data)
    tl, ti = 0.0, 0
    for bi in range(0, len(train_data), 16):
        batch = train_data[bi:bi+16]
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device); n = 0
        for ai, en_s, pos in batch:
            ids_t, T = pad_sent(en_s)
            L = min(pos, T-1)
            with torch.no_grad():
                emb = L0(ids_t).unsqueeze(0)
                h_ctx = L1.fe(emb).squeeze(0)
            h_p = h_ctx[L]
            cw = context_world(ids_t[L], h_p)
            cw_n = cw / (cw.norm() + 1e-8)
            logits = (cw_n.unsqueeze(0) @ zh_world_train.T) / tau
            batch_loss += F.cross_entropy(logits, torch.tensor([ai], device=device))
            n += 1
        if n > 0:
            (batch_loss / n).backward()
            torch.nn.utils.clip_grad_norm_(ctx_params, 1.0)
            opt.step()
            tl += (batch_loss / n).item(); ti += 1
    
    if ep % 10 == 0 or ep == EPOCHS - 1:
        gold = eval_gold(); nce = eval_nce()
        alpha = torch.sigmoid(ctx_alpha).item()
        print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} α={alpha:.3f} gold={gold:.1f}% nce={nce:.1f}%")

print(f"\n  gold change: {gold0:.1f}% → {eval_gold():.1f}%")

print("\nDone.")
