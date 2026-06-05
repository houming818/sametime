"""
最简验证：冻结全部 (L0+t_nodes+t_merge), 两条 learnable 128D path
同一 base (light) × path_B → InfoNCE(光)  vs  × path_C → InfoNCE(轻)
gold 用原 heap_world 保持
"""
import torch, torch.nn as nn, torch.nn.functional as F, random
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
import sentencepiece as spm
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)

ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])

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
    a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
    aL, aR = a_n[..., :d//2], a_n[..., d//2:]
    bL, bR = b_n[..., :d//2], b_n[..., d//2:]
    return torch.cat([aL*bL - aR*bR, aL*bR + aR*bL], -1)

# 两条 learnable path，很小 init
path_B = nn.Parameter(torch.randn(d, device=device) * 0.001)  # light → 光
path_C = nn.Parameter(torch.randn(d, device=device) * 0.001)  # light → 轻

# ── 锚点 ──
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
    ('bright','明亮'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
]

anchor_list = []
for e, z in MANUAL:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchor_list.append((e, z, torch.tensor(ei), torch.tensor(zi)))

# 加 light→光
light_guang = sp.encode_as_ids('光')
if ok(light_guang):
    anchor_list.append(('light', '光', torch.tensor(sp.encode_as_ids('light')), torch.tensor(light_guang)))

anchor_ts = [(e, z, ei.to(device), zi.to(device)) for e, z, ei, zi in anchor_list]
N = len(anchor_ts)

zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world_mat[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# 单义锚点 indices + 多义索引
single_ai = []
ai_guang = ai_qing = None
for ai, (en, zh, _, _) in enumerate(anchor_ts):
    if en == 'light':
        if zh == '光': ai_guang = ai
        if zh == '轻': ai_qing = ai
    single_ai.append(ai)
print(f"Anchors: {N}  light→光={ai_guang}  light→轻={ai_qing}")

# ── Gold ──
GOLD = [('i','我'),('one','一'),('two','二'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),
    ('water','水'),('fire','火'),('sun','太阳'),('moon','月'),('mountain','山'),('river','河'),('sky','天'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),('tree','树'),('flower','花'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),('fish','鱼'),('bird','鸟'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),('mother','母亲'),('father','父亲'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),('milk','牛奶'),('egg','蛋'),('fruit','水果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),('go','去'),('come','来'),('eat','吃'),('drink','喝'),
    ('see','看'),('hear','听'),('write','写'),('walk','走'),('sit','坐'),('stand','站'),('give','给'),('take','拿'),
    ('make','做'),('know','知道'),('think','想'),('want','要'),('love','爱'),('like','喜欢'),
    ('work','工作'),('buy','买'),('live','生活'),('die','死'),('kill','杀'),('help','帮助'),
    ('open','开'),('close','关'),('start','开始'),('stop','停'),('run','跑'),
    ('remember','记住'),('forget','忘记'),('understand','理解'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),('push','推'),('pull','拉'),
    ('carry','带'),('throw','扔'),('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('need','需要'),('keep','保持'),('agree','同意'),('return','返回'),('leave','离开'),
    ('arrive','到达'),('move','移动'),('pass','通过'),('choose','选择'),('decide','决定'),
    ('prepare','准备'),('finish','完成'),('protect','保护'),('create','创造'),('produce','生产'),
    ('develop','发展'),('manage','管理'),('save','保存'),('disappear','消失'),('survive','幸存'),
    ('own','拥有'),('exist','存在'),('happen','发生'),('appear','出现'),('continue','继续'),
    ('reduce','减少'),('increase','增加'),('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),('long','长'),('high','高'),('low','低'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('heavy','重'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),('free','自由'),
    ('dark','黑暗'),('empty','空'),('kind','善良'),('brave','勇敢')]
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

# ── 冻结全部 ──
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False

light_id = sp.encode_as_ids('light')[0]
L0_light = F.normalize(L0.weight[light_id], dim=-1).detach()

gold0 = eval_gold()
cos_path_0 = F.cosine_similarity(F.normalize(path_B, dim=-1).unsqueeze(0), F.normalize(path_C, dim=-1).unsqueeze(0)).item()
out_B0 = cmul(L0_light.unsqueeze(0), path_B.unsqueeze(0)).squeeze(0)
out_C0 = cmul(L0_light.unsqueeze(0), path_C.unsqueeze(0)).squeeze(0)
cos_cmul_0 = F.cosine_similarity(out_B0.unsqueeze(0), out_C0.unsqueeze(0)).item()
print(f"\nINIT: gold={gold0:.1f}% cos_path={cos_path_0:.4f} cos_cmul={cos_cmul_0:.4f}")

# ── 训 path_B + path_C ──
opt = torch.optim.Adam([path_B, path_C], lr=0.01)
for ep in range(2000):
    random.shuffle(single_ai)
    for bi in range(0, len(single_ai), 32):
        batch = single_ai[bi:bi+32]; opt.zero_grad()
        l = torch.tensor(0.0, device=device); n = 0
        for ai in batch:
            ei = anchor_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world_mat.T) / tau
            l += F.cross_entropy(logits, torch.tensor([ai], device=device)); n += 1
        # multi-sense
        out_B = cmul(L0_light.unsqueeze(0), path_B.unsqueeze(0)).squeeze(0)
        logits_B = (out_B.unsqueeze(0) @ zh_world_mat.T) / tau
        l += F.cross_entropy(logits_B, torch.tensor([ai_guang], device=device)); n += 1
        out_C = cmul(L0_light.unsqueeze(0), path_C.unsqueeze(0)).squeeze(0)
        logits_C = (out_C.unsqueeze(0) @ zh_world_mat.T) / tau
        l += F.cross_entropy(logits_C, torch.tensor([ai_qing], device=device)); n += 1
        if n > 0: (l/n).backward(); torch.nn.utils.clip_grad_norm_([path_B, path_C], 1.0); opt.step()

    if ep % 200 == 0 or ep == 1999:
        with torch.no_grad():
            cos_p = F.cosine_similarity(F.normalize(path_B, dim=-1).unsqueeze(0), F.normalize(path_C, dim=-1).unsqueeze(0)).item()
            out_B = cmul(L0_light.unsqueeze(0), path_B.unsqueeze(0)).squeeze(0)
            out_C = cmul(L0_light.unsqueeze(0), path_C.unsqueeze(0)).squeeze(0)
            cos_c = F.cosine_similarity(out_B.unsqueeze(0), out_C.unsqueeze(0)).item()
            gold = eval_gold()
            logits_B = (out_B.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
            logits_C = (out_C.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
            rank_B = (logits_B > logits_B[ai_guang]).sum().item() + 1
            rank_C = (logits_C > logits_C[ai_qing]).sum().item() + 1
            print(f"  ep {ep:4d} gold={gold:.1f}% cos_p={cos_p:.4f} cos_cmul={cos_c:.4f} r_光={rank_B}/{N} r_轻={rank_C}/{N}")

print(f"\n=== 结果 ===")
print(f"cos(path):  {cos_path_0:.4f} → {cos_p:.4f}  Δ={cos_path_0-cos_p:.4f}")
print(f"cos(cmul):  {cos_cmul_0:.4f} → {cos_c:.4f}  Δ={cos_cmul_0-cos_c:.4f}")
print(f"gold: {gold0:.1f}% → {gold:.1f}%")
print("Done.")
