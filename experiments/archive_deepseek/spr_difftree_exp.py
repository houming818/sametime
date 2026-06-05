"""
DiffTree 实验: 树上差分梯度 → 防止共享节点坍缩
unit_root×mul tree + AttnPaths + DiffTree + InfoNCE
"""
import sys; sys.path.insert(0, '.')
from core import TreeNodes, HeapWorld, AttnPaths, AnchorIndex, DiffTree, cmul
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

# ── Models ──
tree = TreeNodes(dim=d, depth=td, init='unit', combine='mul', vocab=V).to(device)
print(f"Tree (unit_root×mul): {tree}")
hw = HeapWorld(vocab=V, dim=d, tree=tree).to(device)
nn.init.normal_(hw.embedding.weight, 0, 0.02)
for p in hw.embedding.parameters(): p.requires_grad = True
for p in tree.parameters(): p.requires_grad = True
for p in hw.merge.parameters(): p.requires_grad = True

diff = DiffTree(tree)

# ── Anchors + Paths ──
anchors = AnchorIndex()
multi_en = set(multis.keys())
for e, z in ALL:
    if e.lower() in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchors.add(e, z, ei, zi)

attn_paths = AttnPaths(tree).to(device)
zh_hard_paths = {}  # ai → [int,...] for diff tree

for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for pi, zh in enumerate(zhs[:3]):
        zh_ids = sp.encode_as_ids(zh)
        if ok(zh_ids):
            anchors.add(en, zh, en_ids, zh_ids)
            ai = anchors.find(en)[-1]
            attn_paths.add(en, zh, ai, pi)
            zh_hard_paths[ai] = diff.get_path_for_token(zh_ids[0])

def _hw_fn(tok_ids):
    return hw.forward(torch.tensor(tok_ids, device=device).unsqueeze(0)).squeeze(0)
anchors.build_zh_world(_hw_fn)

single_ai = [ai for ai, (en, _, _, _) in enumerate(anchors.entries) 
             if en not in multi_en or en not in multis]
print(f"Anchors: {len(anchors)} single={len(single_ai)} paths={attn_paths.num_paths}")
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

# L4 node cos
l4 = tree.embeddings[4].weight
l4_cos = F.cosine_similarity(l4[:,None,:], l4[None,:,:], dim=-1)
mask_l4 = ~torch.eye(16, dtype=torch.bool, device=device)
l4_cos0 = l4_cos[mask_l4].mean().item()

print(f"Gold init: {gold0:.1f}%  cos init: {cos0:.4f}  L4cos init: {l4_cos0:.4f}")
print(f"\nTraining: {len(single_ai)} single + {attn_paths.num_paths} paths, DiffTree scale=0.0")

# ── Train ──
trainable = (list(tree.parameters()) + list(hw.merge.parameters()) + 
             list(hw.embedding.parameters()) + list(attn_paths.parameters()))
opt = torch.optim.Adam(trainable, lr=0.003)

def infonce_loss(query, target_ai):
    logits = (query @ anchors.zh_world.T) / tau
    return F.cross_entropy(logits.unsqueeze(0) if query.dim()==1 else logits, 
                          torch.tensor([target_ai], device=device))

SCALES = [0.0, 0.3, 1.0]  # diff scale: 0=full cancel, 1=no diff
results = {}

for scale in SCALES:
    print(f"\n{'='*60}")
    print(f"DiffTree scale={scale}")
    print(f"{'='*60}")
    
    opt = torch.optim.Adam(trainable, lr=0.003)
    EPOCHS = 100 if scale == 0.0 else 50  # shorter for controls
    
    for ep in range(EPOCHS):
        random.shuffle(single_ai)
        pnames = attn_paths.names()
        random.shuffle(pnames)
        
        for bi in range(0, max(len(single_ai), 1), 16):
            opt.zero_grad()
            losses = []
            path_pairs = []  # for diff tree
            
            for ai in single_ai[bi:bi+16]:
                en_ids = torch.tensor(anchors[ai][2], device=device)
                hw_vec = F.normalize(hw.forward(en_ids.unsqueeze(0)).squeeze(0).mean(dim=0), dim=-1)
                lo = infonce_loss(hw_vec, ai)
                losses.append(lo)
                # single-sense anchors also contribute to diff: use their ZH path
                if ai in zh_hard_paths:
                    path_pairs.append((zh_hard_paths[ai], zh_hard_paths[ai]))
            
            m_start = (bi * 4) % max(1, len(pnames)-4)
            m_batch = []
            for pname in pnames[m_start:m_start+4]:
                _, _, ai = attn_paths.meta[pname]
                en_id = anchors[ai][2][0]
                cw = attn_paths.world_pos(pname, hw.embedding, en_id)
                lo = infonce_loss(F.normalize(cw, dim=-1), ai)
                losses.append(lo)
                m_batch.append(ai)
            
            # Build diff path pairs: same EN word, different senses
            en_pairs = defaultdict(list)
            for ai in m_batch:
                en = anchors[ai][0]
                en_pairs[en].append(ai)
            for en, ai_list in en_pairs.items():
                for i in range(len(ai_list)):
                    for j in range(i+1, len(ai_list)):
                        if ai_list[i] in zh_hard_paths and ai_list[j] in zh_hard_paths:
                            path_pairs.append((zh_hard_paths[ai_list[i]], 
                                             zh_hard_paths[ai_list[j]]))
            
            if len(losses) > 0:
                l = sum(losses) / len(losses)
                l.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scale < 1.0:
                    diff.apply(path_pairs, scale=scale)
                opt.step()
        
        if ep % 25 == 0 or ep == EPOCHS - 1:
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
                ms = 100 * correct / max(total, 1)
                avg_rank = sum(ranks) / max(len(ranks), 1)
                cos_p = attn_paths.pairwise_cos()
                avg_cos = sum(cos_p) / max(len(cos_p), 1)
                l4c = F.cosine_similarity(tree.embeddings[4].weight[:,None,:], 
                                          tree.embeddings[4].weight[None,:,:], dim=-1)
                avg_l4c = l4c[mask_l4].mean().item()
                elapsed = time.time() - t0
                print(f"  ep {ep:4d} gold={gold:.1f}% ms={ms:.0f}% rank={avg_rank:.1f} cos={avg_cos:.4f} L4cos={avg_l4c:.4f} {elapsed:.0f}s")
    
    gold1 = eval_gold()
    results[scale] = {'gold': gold1, 'ms': ms, 'cos': avg_cos, 'L4cos': avg_l4c}

print(f"\n{'='*60}")
print("DiffTree 实验结果")
print(f"{'='*60}")
print(f"{'scale':>8s}  {'gold':>6s}  {'ms':>6s}  {'cos':>7s}  {'L4cos':>7s}")
for s in SCALES:
    r = results[s]
    print(f"  {s:4.1f}   {r['gold']:5.1f}%  {r['ms']:5.0f}%  {r['cos']:6.4f}  {r['L4cos']:6.4f}")
print(f"\ninit: L4cos={l4_cos0:.4f}")
print(f"\nTotal: {time.time()-t0:.0f}s")
