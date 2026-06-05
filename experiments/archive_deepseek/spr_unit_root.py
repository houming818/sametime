"""
最小验证：单位根 tree + CMul，同一 base 两个 path 能否被 InfoNCE 推分离
d=128, depth=5, V=16000, 两个 learnable path, 单义锚点抱 gold
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, math
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07

# ── 模型 ──
L0 = nn.Embedding(V, d).to(device)
# t_nodes: 单位根初始化 — 纯旋转
t_nodes = nn.ModuleList()
for l in range(td):
    n = 2 ** l
    # 每层 n 个节点，每个节点是 64 对单位复数 = 纯旋转角
    w = torch.zeros(n, d)  # [n, 128]
    for i in range(n):
        angle = 2 * math.pi * i / n  # 第 i 个节点的旋转角
        for p in range(d // 2):
            w[i, 2*p]   = math.cos(angle)   # real
            w[i, 2*p+1] = math.sin(angle)   # imag
    t_nodes.append(nn.Embedding(n, d, _weight=w).to(device))
t_merge = nn.Linear(d, d).to(device)
nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

# 加载 L0 (从最佳 ckpt)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0'])

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
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    aL, aR = a_n[..., :d//2], a_n[..., d//2:]
    bL, bR = b_n[..., :d//2], b_n[..., d//2:]
    return torch.cat([aL*bL - aR*bR, aL*bR + aR*bL], -1)

# ── 两条 learnable path 向量 ──
path_B = nn.Parameter(torch.randn(d, device=device) * 0.01)  # light → 光
path_C = nn.Parameter(torch.randn(d, device=device) * 0.01)  # light → 轻

# ── 构建锚点 ──
import sentencepiece as spm
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

# 手工锚点 (来自现有实验)
MANUAL = [
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('one','一'),('two','二'),('three','三'),('four','四'),
    ('five','五'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),('water','水'),('fire','火'),('sun','太阳'),
    ('moon','月'),('mountain','山'),('river','河'),('sky','天'),('earth','地'),('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('gold','金'),('iron','铁'),('star','星'),('cloud','云'),('fish','鱼'),('bird','鸟'),
    ('horse','马'),('red','红'),('green','绿'),('white','白'),('black','黑'),('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),('milk','牛奶'),('egg','蛋'),('fruit','水果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),('spring','春'),('summer','夏'),('autumn','秋'),('winter','冬'),
    ('go','去'),('come','来'),('eat','吃'),('drink','喝'),('see','看'),('hear','听'),('say','说'),('read','读'),
    ('write','写'),('walk','走'),('sit','坐'),('stand','站'),('give','给'),('take','拿'),('make','做'),('know','知道'),
    ('think','想'),('want','要'),('love','爱'),('like','喜欢'),('work','工作'),('play','玩'),('buy','买'),('sell','卖'),
    ('live','生活'),('die','死'),('kill','杀'),('speak','说'),('talk','谈'),('find','找'),('learn','学'),('teach','教'),
    ('help','帮助'),('open','开'),('close','关'),('start','开始'),('stop','停'),('run','跑'),('fly','飞'),
    ('remember','记住'),('forget','忘记'),('believe','相信'),('understand','理解'),('explain','解释'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),('push','推'),('pull','拉'),('carry','带'),('throw','扔'),
    ('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),('wait','等'),('need','需要'),('keep','保持'),
    ('agree','同意'),('return','返回'),('leave','离开'),('arrive','到达'),('stay','停留'),('move','移动'),
    ('pass','通过'),('rise','上升'),('fall','下降'),('lift','举起'),('burn','烧'),('choose','选择'),('decide','决定'),
    ('prepare','准备'),('finish','完成'),('enjoy','享受'),('protect','保护'),('attack','攻击'),('defend','防御'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),('manage','管理'),('organize','组织'),('save','保存'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),('happen','发生'),('appear','出现'),
    ('continue','继续'),('reduce','减少'),('increase','增加'),('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),('long','长'),('short','短'),('high','高'),('low','低'),
    ('fast','快'),('slow','慢'),('rich','富'),('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('sad','悲伤'),
    ('heavy','重'),('light','轻'),('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),('free','自由'),('fair','公平'),
    ('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),('kind','善良'),('brave','勇敢'),('angry','愤怒'),
]

anchor_list = []
for e, z in MANUAL:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi):
        # 排除 light/spring 等已经在多义测试里的
        if e in ('light', 'spring', 'right', 'fair', 'run', 'hard'):
            anchor_list.append((e, z, ei, zi))
        else:
            anchor_list.append((e, z, ei, zi))

# 单义锚点 indices (非多义)
single_ai = [ai for ai, (en, _, _, _) in enumerate(anchor_list) if en not in ('light',)]

# 找 light→光 和 light→轻 的 anchor index
en_to_ai = defaultdict(list)
for ai, (en, zh, _, _) in enumerate(anchor_list):
    if en == 'light':
        en_to_ai[en].append((ai, zh))

# 加入 '光' 作为 light 的第二个义项
light_guang_ids = sp.encode_as_ids('光')
if ok(light_guang_ids):
    anchor_list.append(('light', '光', sp.encode_as_ids('light'), light_guang_ids))
    ai_guang = len(anchor_list) - 1
else:
    raise ValueError("光 not encodable")

ai_qing = [ai for ai, (en, zh, _, _) in enumerate(anchor_list) if en == 'light' and zh == '轻'][0]
ai_guang_used = ai_guang

N = len(anchor_list)
anchor_ts = [(e, z, torch.tensor(ei, device=device), torch.tensor(zi, device=device)) for e, z, ei, zi in anchor_list]
zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world_mat[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

print(f"Anchors: {N}  light→轻={ai_qing}  light→光={ai_guang_used}")

# ── Gold eval ──
GOLD = [('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('one','一'),('two','二'),('three','三'),('four','四'),
    ('five','五'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),('water','水'),('fire','火'),('sun','太阳'),
    ('moon','月'),('mountain','山'),('river','河'),('sky','天'),('earth','地'),('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('gold','金'),('iron','铁'),('star','星'),('cloud','云'),('fish','鱼'),('bird','鸟'),
    ('horse','马'),('red','红'),('green','绿'),('white','白'),('black','黑'),('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),('milk','牛奶'),('egg','蛋'),('fruit','水果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),('go','去'),('come','来'),('eat','吃'),('drink','喝'),
    ('see','看'),('hear','听'),('write','写'),('walk','走'),('sit','坐'),('stand','站'),('give','给'),('take','拿'),
    ('make','做'),('know','知道'),('think','想'),('want','要'),('love','爱'),('like','喜欢'),('work','工作'),('play','玩'),
    ('buy','买'),('sell','卖'),('live','生活'),('die','死'),('kill','杀'),('speak','说'),('talk','谈'),('find','找'),
    ('learn','学'),('teach','教'),('help','帮助'),('open','开'),('close','关'),('start','开始'),('stop','停'),('run','跑'),
    ('fly','飞'),('remember','记住'),('forget','忘记'),('believe','相信'),('understand','理解'),('explain','解释'),
    ('change','改变'),('grow','生长'),('build','建设'),('cut','切'),('break','打破'),('push','推'),('pull','拉'),
    ('carry','带'),('throw','扔'),('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('need','需要'),('keep','保持'),('agree','同意'),('return','返回'),('leave','离开'),
    ('arrive','到达'),('stay','停留'),('move','移动'),('pass','通过'),('rise','上升'),('fall','下降'),
    ('lift','举起'),('burn','烧'),('choose','选择'),('decide','决定'),('prepare','准备'),('finish','完成'),
    ('enjoy','享受'),('protect','保护'),('attack','攻击'),('defend','防御'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),('manage','管理'),('organize','组织'),
    ('save','保存'),('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),('happen','发生'),
    ('appear','出现'),('continue','继续'),('reduce','减少'),('increase','增加'),
    ('big','大'),('small','小'),('good','好'),('bad','坏'),('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),('fast','快'),('slow','慢'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('sad','悲伤'),('heavy','重'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),('clean','干净'),('safe','安全'),
    ('easy','容易'),('important','重要'),('free','自由'),('fair','公平'),('bright','明亮'),('dark','黑暗'),
    ('dry','干'),('full','满'),('empty','空'),('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒')]
ge = [torch.tensor(sp.encode_as_ids(e), device=device) for e,_ in GOLD if ok(sp.encode_as_ids(e))]
gz = [torch.tensor(sp.encode_as_ids(z), device=device) for _,z in GOLD if ok(sp.encode_as_ids(z))]
gw = torch.zeros(len(gz), d, device=device)
with torch.no_grad():
    for i, zi in enumerate(gz): gw[i] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

def eval_gold():
    with torch.no_grad():
        c = 0
        for i, ei in enumerate(ge):
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            if (hw.unsqueeze(0) @ gw.T / tau).squeeze(0).argmax() == i: c += 1
        return 100 * c / len(ge)

# ── Pre-train t_nodes 的 InfoNCE (单义锚点, 让单位根学会对齐) ──
print("Pre-training t_nodes on single-sense anchors...")
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
trainable_t = [p for tn in t_nodes for p in tn.parameters()]
opt_t = torch.optim.Adam(trainable_t, lr=0.003)

gold0 = eval_gold()
print(f"  gold before: {gold0:.1f}%")

for ep in range(200):
    random.shuffle(single_ai)
    for bi in range(0, len(single_ai), 32):
        batch = single_ai[bi:bi+32]; opt_t.zero_grad()
        l = torch.tensor(0.0, device=device); n = 0
        for ai in batch:
            ei = anchor_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_mat.T) / tau
            l += F.cross_entropy(logits, torch.tensor([ai], device=device)); n += 1
        if n > 0: (l/n).backward(); torch.nn.utils.clip_grad_norm_(trainable_t, 1.0); opt_t.step()
    if ep % 50 == 0 or ep == 199:
        gold = eval_gold()
        print(f"  ep {ep:3d} gold={gold:.1f}%")

gold1 = eval_gold()
print(f"  gold after: {gold1:.1f}%")

# ── 冻结 t_nodes, L0, t_merge ──
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False

# ── 核心实验：训 path_B 和 path_C ──
print(f"\n=== 核心实验: path_B (light→光) vs path_C (light→轻) ===")
light_id = sp.encode_as_ids('light')[0]
with torch.no_grad():
    L0_light = F.normalize(L0.weight[light_id], dim=-1)

# 初始值
path_B_init = path_B.detach().clone()
path_C_init = path_C.detach().clone()
cos_path_init = F.cosine_similarity(F.normalize(path_B, dim=-1).unsqueeze(0),
                                     F.normalize(path_C, dim=-1).unsqueeze(0)).item()
out_B_init = cmul(L0_light, path_B)
out_C_init = cmul(L0_light, path_C)
cos_cmul_init = F.cosine_similarity(out_B_init.unsqueeze(0), out_C_init.unsqueeze(0)).item()
print(f"  initial: cos(path_B,path_C)={cos_path_init:.4f}  cos(cmul_B,cmul_C)={cos_cmul_init:.4f}")

opt = torch.optim.Adam([path_B, path_C], lr=0.01)

for ep in range(1000):
    random.shuffle(single_ai)
    for bi in range(0, len(single_ai), 64):
        batch_s = single_ai[bi:bi+64]
        opt.zero_grad()
        l = torch.tensor(0.0, device=device); n = 0
        # 单义锚点: heap_world (保 gold)
        for ai in batch_s:
            ei = anchor_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_mat.T) / tau
            l += F.cross_entropy(logits, torch.tensor([ai], device=device)); n += 1
        # 多义锚点: CMul with learnable paths
        # path_B → 光
        out_B = cmul(L0_light.unsqueeze(0), path_B.unsqueeze(0)).squeeze(0)
        logits_B = (out_B.unsqueeze(0) @ zh_world_mat.T) / tau
        l += F.cross_entropy(logits_B, torch.tensor([ai_guang_used], device=device)); n += 1
        # path_C → 轻
        out_C = cmul(L0_light.unsqueeze(0), path_C.unsqueeze(0)).squeeze(0)
        logits_C = (out_C.unsqueeze(0) @ zh_world_mat.T) / tau
        l += F.cross_entropy(logits_C, torch.tensor([ai_qing], device=device)); n += 1
        if n > 0: (l/n).backward(); torch.nn.utils.clip_grad_norm_([path_B, path_C], 1.0); opt.step()

    if ep % 100 == 0 or ep == 999:
        with torch.no_grad():
            cos_path = F.cosine_similarity(F.normalize(path_B, dim=-1).unsqueeze(0),
                                           F.normalize(path_C, dim=-1).unsqueeze(0)).item()
            out_B = cmul(L0_light.unsqueeze(0), path_B.unsqueeze(0)).squeeze(0)
            out_C = cmul(L0_light.unsqueeze(0), path_C.unsqueeze(0)).squeeze(0)
            cos_cmul = F.cosine_similarity(out_B.unsqueeze(0), out_C.unsqueeze(0)).item()
            logits_B = (out_B.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
            logits_C = (out_C.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
            rank_B = (logits_B > logits_B[ai_guang_used]).sum().item() + 1
            rank_C = (logits_C > logits_C[ai_qing]).sum().item() + 1
            gold = eval_gold()
            print(f"  ep {ep:4d} gold={gold:.1f}% cos_path={cos_path:.4f} cos_cmul={cos_cmul:.4f} "
                  f"rank_光={rank_B}/{N} rank_轻={rank_C}/{N}")

print(f"\n=== 总结 ===")
print(f"cos(path_B,path_C): {cos_path_init:.4f} → {cos_path:.4f}")
print(f"cos(cmul_B,cmul_C): {cos_cmul_init:.4f} → {cos_cmul:.4f}")
print(f"  Δpaths = {cos_path_init - cos_path:.4f}")
print(f"  Δcmul  = {cos_cmul_init - cos_cmul:.4f}")
print("Done.")
