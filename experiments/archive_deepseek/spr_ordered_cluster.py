"""
有序训练: 按 ZH target 的 L4 路径分 16 簇, 每步只训一簇
节点只收该簇的梯度 → 不同簇互不干扰 → 有序隔离
独立 path (已验证) 作对照
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time
import sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)
t0 = time.time()

# ── 模型 ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
t_merge = nn.Linear(d, d).to(device)

ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False

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
def zh_L4_path(zh_ids):
    """返回 ZH 词的 L4 路径索引 (0-15)"""
    return (zh_ids[0] // (V // 16)) % 16

# ── 数据 ──
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)
with open('/workspace/labse_anchors.py') as f: labse_pairs = eval(f.read().split('=',1)[1].strip())
MANUAL = [('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('one','一'),('two','二'),('three','三'),('four','四'),
    ('five','五'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),('water','水'),('fire','火'),('sun','太阳'),
    ('moon','月'),('mountain','山'),('river','河'),('sky','天'),('earth','地'),('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('gold','金'),('iron','铁'),('star','星'),('cloud','云'),('fish','鱼'),('bird','鸟'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
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
    ('bright','明亮'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),]
ALL = labse_pairs + MANUAL

en_known = {}
for e, z in ALL: en_known.setdefault(e.lower(), set()).add(z)
multis = {}
for en, known in en_known.items():
    if en not in dict_data: continue
    existing = next(iter(known))
    extra = []
    for zh, pos in dict_data[en]:
        if zh not in known and pos in ('n','v','a','adj','adv','vi','vt') and 1 <= len(zh) <= 6:
            zh_ids = sp.encode_as_ids(zh)
            if ok(zh_ids): extra.append(zh)
    if extra: multis[en] = [existing] + extra[:2]
print(f"Multi-sense: {len(multis)}")

# ── 锚点 + 路径分簇 ──
categories = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # 16 clusters by L4 path
multi_en = set(multis.keys())
anchor_raw = []
for e, z in ALL:
    if e.lower() in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi):
        idx = len(anchor_raw)
        cat = zh_L4_path(zi)
        categories[cat].append(idx)
        anchor_raw.append((e, z, torch.tensor(ei), torch.tensor(zi)))
for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for zh in zhs[:3]:
        zh_ids = sp.encode_as_ids(zh)
        if ok(en_ids) and ok(zh_ids):
            idx = len(anchor_raw)
            cat = zh_L4_path(zh_ids)
            categories[cat].append(idx)
            anchor_raw.append((en, zh, torch.tensor(en_ids), torch.tensor(zh_ids)))

anchor_ts = [(e, z, ei.to(device), zi.to(device)) for e, z, ei, zi in anchor_raw]
N = len(anchor_ts)
en_idx = defaultdict(list)
for ai, (en, _, _, _) in enumerate(anchor_ts): en_idx[en].append(ai)

zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world_mat[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# Independent paths
paths = nn.ParameterDict()
path_meta = {}
for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for pi, zh in enumerate(zhs[:3]):
        for ai in en_idx[en]:
            if anchor_ts[ai][1] == zh:
                pname = f"{en}_{pi}"
                paths[pname] = nn.Parameter(torch.randn(d, device=device) * 0.001)
                path_meta[pname] = (en, zh, ai)
                break
pnames = list(paths.keys())

# Pre-compute L0 vecs for multi-sense words
l0_vecs = {}
for en in multi_en:
    en_ids = sp.encode_as_ids(en)
    if ok(en_ids): l0_vecs[en] = F.normalize(L0.weight[en_ids[0]], dim=-1).detach()

# Gold eval
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

gold0 = eval_gold()
print(f"Gold init: {gold0:.1f}%")

# Print cluster sizes
for c in range(16):
    single_c = [ai for ai in categories[c] if anchor_ts[ai][0] not in multi_en]
    multi_c = [ai for ai in categories[c] if anchor_ts[ai][0] in multi_en]
    print(f"  L4_cluster[{c:2d}]: {len(categories[c])} anchors ({len(single_c)} single + {len(multi_c)} multi)")

print(f"\nPaths: {len(paths)}  In clusters: {[sum(pnames.count(p) for p in paths) for _ in []][:1]}")
print(f"Training: {N} anchors in 16 clusters, {len(paths)} independent paths")

# ── 训练: 每步只训一个 cluster ──
opt = torch.optim.Adam(list(paths.parameters()), lr=0.01)

for ep in range(400):
    # Shuffle cluster order
    cluster_order = list(range(16))
    random.shuffle(cluster_order)
    for cat in cluster_order:
        cat_anchors = categories[cat]
        if len(cat_anchors) < 2: continue
        random.shuffle(cat_anchors)
        for bi in range(0, max(len(cat_anchors), 1), 16):
            opt.zero_grad()
            losses = []; n = 0
            for ai in cat_anchors[bi:bi+16]:
                ei = anchor_ts[ai][2]
                hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
                logits = (hw.unsqueeze(0) @ zh_world_mat.T) / tau
                lo = F.cross_entropy(logits, torch.tensor([ai], device=device))
                losses.append(lo); n += 1
            # Also train multi-sense paths whose ZH target is in this cluster
            for pname in random.sample(pnames, min(2, len(pnames))):
                en, zh, ai = path_meta.get(pname, ('','',''))
                if ai not in cat_anchors: continue
                if en not in l0_vecs: continue
                p = F.normalize(paths[pname], dim=-1)
                out = cmul(l0_vecs[en].unsqueeze(0), p.unsqueeze(0)).squeeze(0)
                logits = (out.unsqueeze(0) @ zh_world_mat.T) / tau
                lo = F.cross_entropy(logits, torch.tensor([ai], device=device))
                losses.append(lo); n += 1
            if n > 0:
                l = sum(losses) / n
                l.backward()
                torch.nn.utils.clip_grad_norm_(list(paths.parameters()), 1.0)
                opt.step()
    
    if ep % 100 == 0 or ep == 399:
        with torch.no_grad():
            gold = eval_gold()
            correct = 0; total = 0; ranks = []
            for pname in pnames:
                en, zh, ai = path_meta.get(pname, ('','',''))
                if en not in l0_vecs: continue
                p = F.normalize(paths[pname], dim=-1)
                out = cmul(l0_vecs[en].unsqueeze(0), p.unsqueeze(0)).squeeze(0)
                logits = (out.unsqueeze(0) @ zh_world_mat.T / tau).squeeze(0)
                rank = (logits > logits[ai]).sum().item() + 1
                ranks.append(rank)
                if rank <= 1: correct += 1
                total += 1
            ms = 100 * correct / max(total, 1)
            avg_rank = sum(ranks) / max(len(ranks), 1)
            cos_pairs = []
            for en in multi_en:
                e_names = [n for n in pnames if path_meta.get(n, ('','',''))[0] == en]
                if len(e_names) >= 2:
                    p0 = F.normalize(paths[e_names[0]], dim=-1)
                    p1 = F.normalize(paths[e_names[1]], dim=-1)
                    cos_pairs.append((p0 * p1).sum().item())
            avg_cos = sum(cos_pairs) / max(len(cos_pairs), 1)
            # Cluster-separated cos: only within same L4 cluster
            within_cos = []
            for c in range(16):
                c_ens = [en for en in multi_en if any(anchor_ts[ai][0]==en for ai in categories[c])]
                c_names = [n for n in pnames if path_meta.get(n,('','',''))[0] in c_ens]
                for i in range(len(c_names)):
                    for j in range(i+1, len(c_names)):
                        if path_meta[c_names[i]][0] == path_meta[c_names[j]][0]:
                            p0 = F.normalize(paths[c_names[i]], dim=-1)
                            p1 = F.normalize(paths[c_names[j]], dim=-1)
                            within_cos.append((p0 * p1).sum().item())
            avg_within_cos = sum(within_cos) / max(len(within_cos), 1)
            print(f"  ep {ep:4d} gold={gold:.1f}% ms={ms:.0f}% rank={avg_rank:.1f} cos={avg_cos:.4f} cos_within={avg_within_cos:.4f} {time.time()-t0:.0f}s")

print(f"\nGold: {gold0:.1f}% → {eval_gold():.1f}%")
print(f"avg_cos: {avg_cos:.4f}  within_cluster_cos: {avg_within_cos:.4f}")
print("Done.")
