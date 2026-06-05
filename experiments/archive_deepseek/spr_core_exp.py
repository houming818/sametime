"""
全量实验: unit root × multiplicative tree nodes
tree + paths → InfoNCE → 验证分离 + gold 无损
"""
import sys
sys.path.insert(0, '/workspace')
from core import TreeNodes, HeapWorld, Paths, AnchorIndex, Trainer, cmul
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time
import sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)
t0 = time.time()

# ── 数据 ──
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)
with open('/workspace/labse_anchors.py') as f: labse_pairs = eval(f.read().split('=',1)[1].strip())
MANUAL = [('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('one','一'),('two','二'),('three','三'),('four','四'),
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
    ('bright','明亮'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),]
ALL = labse_pairs + MANUAL

# 构建多义词表: ECDICT 附加义项
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
            if ok(zh_ids) and len(zh_ids) >= 1: extra.append(zh)
    if extra: multis[en] = [existing] + extra[:2]
print(f"Multi-sense words: {len(multis)}")

# ── 树: unit root + multiplicative ──
tree = TreeNodes(dim=d, depth=td, init='unit', combine='mul', vocab=V).to(device)
print(f"TreeNodes: {tree}")

# 加载 L0
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0_pretrained = nn.Embedding(V, d).to(device); L0_pretrained.load_state_dict(ckpt['L0'])
# freeze L0
for p in L0_pretrained.parameters(): p.requires_grad = False

# heap_world with new tree + pretrained L0
hw = HeapWorld(vocab=V, dim=d, tree=tree).to(device)
hw.embedding.load_state_dict(ckpt['L0'])
# Freeze L0 (inside hw.embedding)
for p in hw.embedding.parameters(): p.requires_grad = False
# Train tree + merge
for p in tree.parameters(): p.requires_grad = True
for p in hw.merge.parameters(): p.requires_grad = True

# ── 锚点 ──
anchors = AnchorIndex()
multi_en = set(multis.keys())
for e, z in ALL:
    if e.lower() in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchors.add(e, z, ei, zi)

# Paths
paths = Paths(dim=d)
for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for pi, zh in enumerate(zhs[:3]):
        zh_ids = sp.encode_as_ids(zh)
        if ok(zh_ids):
            # Add to anchor index (for Zh world matrix)
            anchors.add(en, zh, en_ids, zh_ids)
            # Add learnable path
            paths.add(en, zh, anchors.find(en)[-1], pi)

# Build ZH world matrix
anchors.build_zh_world(lambda tok: hw.forward(tok.unsqueeze(0)).squeeze(0))

single_ai = [ai for ai, (en, _, _, _) in enumerate(anchors.entries) 
             if en not in multi_en or en not in multis]

print(f"Anchors: {len(anchors)} (single={len(single_ai)})  Paths: {paths.num_paths}")
N = len(anchors)

# ── Gold eval ──
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
ge = [sp.encode_as_ids(e) for e,_ in GOLD if ok(sp.encode_as_ids(e))]
gz = [sp.encode_as_ids(z) for _,z in GOLD if ok(sp.encode_as_ids(z))]
gw = torch.zeros(len(gz), d, device=device)
with torch.no_grad():
    for i, zi in enumerate(gz):
        gw[i] = F.normalize(hw.forward(torch.tensor(zi, device=device).unsqueeze(0)).squeeze(0).mean(dim=0), dim=-1)

def eval_gold():
    with torch.no_grad():
        c = 0
        for i, ei in enumerate(ge):
            hw_vec = F.normalize(hw.forward(torch.tensor(ei, device=device).unsqueeze(0)).squeeze(0).mean(dim=0), dim=-1)
            if (hw_vec.unsqueeze(0) @ gw.T / tau).squeeze(0).argmax() == i: c += 1
        return 100 * c / len(ge)

gold0 = eval_gold()
print(f"Gold init: {gold0:.1f}%")
cos_paths_0 = sum(paths.pairwise_cos()) / max(len(paths.pairwise_cos()), 1)
print(f"Avg path cos (init): {cos_paths_0:.4f}")

# ── 训练 ──
# Single-sense pre-training (tree nodes need to learn alignment first)
# Actually, unit roots should already produce diverse paths — just need to push them to targets
trainable = (list(tree.parameters()) + list(hw.merge.parameters()) + list(paths.parameters()))
opt = torch.optim.Adam(trainable, lr=0.003)

def infonce_loss(query, target_ai):
    logits = (query @ anchors.zh_world.T) / tau
    return F.cross_entropy(logits.unsqueeze(0) if query.dim() == 1 else logits, 
                          torch.tensor([target_ai], device=device))

print(f"\nTraining: {len(single_ai)} single-sense + {paths.num_paths} multi-sense paths")
EPOCHS = 600
for ep in range(EPOCHS):
    random.shuffle(single_ai)
    path_names = paths.names()
    random.shuffle(path_names)
    
    for bi in range(0, max(len(single_ai), 1), 16):
        opt.zero_grad()
        l = torch.tensor(0.0, device=device); n = 0
        # Single-sense
        s_batch = single_ai[bi:bi+16]
        for ai in s_batch:
            en_word, _, en_ids, _ = anchors[ai]
            en_t = torch.tensor(en_ids, device=device)
            hw_vec = F.normalize(hw.forward(en_t.unsqueeze(0)).squeeze(0).mean(dim=0), dim=-1)
            l += infonce_loss(hw_vec, ai); n += 1
        # Multi-sense
        m_start = (bi * 2) % max(1, len(path_names) - 2)
        m_batch = path_names[m_start:m_start+2]
        for pname in m_batch:
            en_word, _, ai = paths.meta[pname]
            en_id = anchors[ai][2][0]
            cw = paths.world_pos(pname, hw.embedding, en_id)
            l += infonce_loss(F.normalize(cw, dim=-1), ai); n += 1
        if n > 0: (l/n).backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    
    if ep % 100 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            gold = eval_gold()
            # Multi-sense rank
            correct = 0; total = 0
            ranks = []
            for pname in paths.names():
                en_word, _, ai = paths.meta[pname]
                en_id = anchors[ai][2][0]
                cw = F.normalize(paths.world_pos(pname, hw.embedding, en_id), dim=-1)
                logits = (cw.unsqueeze(0) @ anchors.zh_world.T / tau).squeeze(0)
                rank = (logits > logits[ai]).sum().item() + 1
                ranks.append(rank)
                if rank <= 1: correct += 1
                total += 1
            ms_acc = 100 * correct / max(total, 1)
            avg_rank = sum(ranks) / max(len(ranks), 1)
            cos_pairs = paths.pairwise_cos()
            avg_cos = sum(cos_pairs) / max(len(cos_pairs), 1)
            elapsed = time.time() - t0
            print(f"  ep {ep:4d} gold={gold:.1f}% ms={ms_acc:.0f}% rank={avg_rank:.1f} cos={avg_cos:.4f} {elapsed:.0f}s")

print(f"\nDone in {time.time()-t0:.0f}s")
