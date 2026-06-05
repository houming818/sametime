"""
AttnPaths 实验: 3191 条 sense → 31D 注意力权重 → soft_path via t_nodes
验证: 概念共享, gold 无损, 路径不坍缩
"""
import sys; sys.path.insert(0, '.')
from core import TreeNodes, HeapWorld, AttnPaths, AnchorIndex, cmul
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

# 多义词表
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

# ── 加载 L0 + 冻结 t_nodes (旧树, 加法) ──
tree = TreeNodes(dim=d, depth=td, init='zero', combine='add', vocab=V).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
for i, tn in enumerate(tree.embeddings): tn.load_state_dict(ckpt['t_nodes'][i])
print(f"Tree (loaded): {tree}")

hw = HeapWorld(vocab=V, dim=d, tree=tree).to(device)
hw.embedding.load_state_dict(ckpt['L0'])
hw.merge.load_state_dict(ckpt['t_merge'])
# 冻结 L0 + tree + merge
for p in hw.embedding.parameters(): p.requires_grad = False
for p in tree.parameters(): p.requires_grad = False
for p in hw.merge.parameters(): p.requires_grad = False

# ── 锚点 ──
anchors = AnchorIndex()
multi_en = set(multis.keys())
for e, z in ALL:
    if e.lower() in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchors.add(e, z, ei, zi)

# AttnPaths: 31D 注意力权重
attn_paths = AttnPaths(tree).to(device)
for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for pi, zh in enumerate(zhs[:3]):
        zh_ids = sp.encode_as_ids(zh)
        if ok(zh_ids):
            anchors.add(en, zh, en_ids, zh_ids)
            attn_paths.add(en, zh, anchors.find(en)[-1], pi)

# Build ZH world matrix
def _hw_fn(tok_ids):
    t = torch.tensor(tok_ids, device=device)
    return hw.forward(t.unsqueeze(0)).squeeze(0)
anchors.build_zh_world(_hw_fn)

single_ai = [ai for ai, (en, _, _, _) in enumerate(anchors.entries) 
             if en not in multi_en or en not in multis]
print(f"Anchors: {len(anchors)} (single={len(single_ai)})  {attn_paths}")
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
cos0 = sum(attn_paths.pairwise_cos()) / max(len(attn_paths.pairwise_cos()), 1)
print(f"Gold init: {gold0:.1f}%  Avg soft_path cos (init): {cos0:.4f}")

# ── 训练 ──
trainable = list(attn_paths.parameters())
opt = torch.optim.Adam(trainable, lr=0.01)

def infonce_loss(query, target_ai):
    logits = (query @ anchors.zh_world.T) / tau
    return F.cross_entropy(logits.unsqueeze(0) if query.dim()==1 else logits, 
                          torch.tensor([target_ai], device=device))

print(f"\nTraining: {len(single_ai)} single + {attn_paths.num_paths} attn_paths")
EPOCHS = 600
for ep in range(EPOCHS):
    random.shuffle(single_ai)
    pnames = attn_paths.names()
    random.shuffle(pnames)
    
    for bi in range(0, max(len(single_ai), 1), 16):
        opt.zero_grad()
        l = torch.tensor(0.0, device=device); n = 0
        # Single-sense
        for ai in single_ai[bi:bi+16]:
            en_ids = torch.tensor(anchors[ai][2], device=device)
            hw_vec = F.normalize(hw.forward(en_ids.unsqueeze(0)).squeeze(0).mean(dim=0), dim=-1)
            l += infonce_loss(hw_vec, ai); n += 1
        # Multi-sense: 2 paths per batch
        m_start = (bi * 2) % max(1, len(pnames)-2)
        for pname in pnames[m_start:m_start+2]:
            _, _, ai = attn_paths.meta[pname]
            en_id = anchors[ai][2][0]
            cw = attn_paths.world_pos(pname, hw.embedding, en_id)
            l += infonce_loss(F.normalize(cw, dim=-1), ai); n += 1
        if n > 0: (l/n).backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    
    if ep % 100 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            gold = eval_gold()
            correct = 0; total = 0; ranks = []
            for pname in attn_paths.names():
                _, _, ai = attn_paths.meta[pname]
                en_id = anchors[ai][2][0]
                cw = F.normalize(attn_paths.world_pos(pname, hw.embedding, en_id), dim=-1)
                logits = (cw.unsqueeze(0) @ anchors.zh_world.T / tau).squeeze(0)
                rank = (logits > logits[ai]).sum().item() + 1
                ranks.append(rank)
                if rank <= 1: correct += 1
                total += 1
            ms_acc = 100 * correct / max(total, 1)
            avg_rank = sum(ranks) / max(len(ranks), 1)
            cos_pairs = attn_paths.pairwise_cos()
            avg_cos = sum(cos_pairs) / max(len(cos_pairs), 1)
            elapsed = time.time() - t0
            print(f"  ep {ep:4d} gold={gold:.1f}% ms={ms_acc:.0f}% rank={avg_rank:.1f} cos={avg_cos:.4f} {elapsed:.0f}s")

gold1 = eval_gold()
print(f"\nGold: {gold0:.1f}% → {gold1:.1f}%")
print(f"soft_path cos: {cos0:.4f} → {avg_cos:.4f}")

# ── 概念重叠分析 ──
print(f"\n=== 概念重叠: 同语义簇词对的节点激活 cos ===")
test_pairs = [
    ('light', 'sun'), ('light', 'bright'),   # 光/亮集群
    ('light', 'weight'), ('light', 'heavy'),  # 轻/重集群
    ('river', 'sea'), ('river', 'water'),     # 水集群
    ('river', 'mountain'),                    # 不同集群(应该低 cos)
    ('tree', 'flower'), ('tree', 'wood'),     # 植物集群
    ('day', 'night'), ('day', 'morning'),     # 时间集群
    ('big', 'small'), ('big', 'large'),       # 大小集群
]
# Only keep pairs where both words are in the training set
valid_pairs = []
for a, b in test_pairs:
    if a in multis and b in multis:
        valid_pairs.append((a, b))
    elif a in multis:
        valid_pairs.append((a, '<single>'))
    elif b in multis:
        valid_pairs.append(('<single>', b))

# Actually, compute overlap for multi-sense word pairs
overlap_words = [w for w in multis if any(p in ['light','sun','river','sea','tree','flower','day','night','big','small'] for p in multis[w])]
overlap_words = sorted(set(overlap_words))[:12]
try:
    overlaps = attn_paths.concept_overlap(overlap_words)
    for (a, b), cos in sorted(overlaps.items(), key=lambda x: -x[1])[:15]:
        print(f"  {a:15s} ↔ {b:15s}  attn_cos={cos:.4f}")
except Exception as e:
    print(f"  skipped: {e}")

print(f"\nDone in {time.time()-t0:.0f}s")
