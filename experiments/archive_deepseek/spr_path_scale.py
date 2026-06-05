import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, random, json, time, os
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
t0 = time.time()

sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)

ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
print(f"Loaded checkpoint in {time.time()-t0:.1f}s")

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

# ── ECDICT ──
ecd_senses = {}
try:
    with open('/workspace/multi_sense_anchors.json') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list): ecd_senses[k.lower()] = [str(x) for x in v]
            elif isinstance(v, dict):
                zh = v.get('zh', v.get('senses', v.get('translations', [])))
                if isinstance(zh, list): ecd_senses[k.lower()] = [str(x) for x in zh]
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                en = str(item.get('en', item.get('word', ''))).lower()
                zh = item.get('zh', item.get('senses', []))
                if en and isinstance(zh, list) and len(zh) > 1: ecd_senses[en] = [str(x) for x in zh]
    print(f"ECDICT: {len(ecd_senses)} multi-sense words")
except Exception as e:
    print(f"ECDICT load failed ({e}), using fallback")
    ecd_senses = {
        'light': ['光','轻','灯'],'spring': ['春','泉'],'right': ['正确','右边','权利'],
        'bank': ['银行','岸'],'bear': ['熊','忍受'],'charge': ['充电','收费'],
        'check': ['检查','支票'],'class': ['班级','阶级'],'draw': ['画','抽'],
        'fair': ['公平','展览会'],'fall': ['秋天','落下'],'figure': ['数字','身材'],
        'fly': ['飞','苍蝇'],'foot': ['脚','英尺'],'head': ['头','领导'],
        'iron': ['铁','熨斗'],'key': ['钥匙','关键'],'kind': ['善良','种类'],
        'lead': ['领导','铅'],'lie': ['说谎','躺'],'like': ['喜欢','像'],
        'mark': ['标记','分数'],'match': ['比赛','匹配'],'mean': ['意思','平均'],
        'mine': ['我的','矿'],'miss': ['想念','错过'],'object': ['物体','反对'],
        'order': ['顺序','命令'],'park': ['公园','停车'],'party': ['派对','政党'],
        'patient': ['病人','耐心'],'plant': ['植物','工厂'],'play': ['玩','戏剧'],
        'present': ['现在','礼物'],'produce': ['生产','农产品'],'race': ['比赛','种族'],
        'record': ['记录','唱片'],'ring': ['戒指','铃声'],'rock': ['岩石','摇滚'],
        'row': ['行','划船'],'run': ['跑','运行'],'safe': ['安全','保险箱'],
        'season': ['季节','调味'],'second': ['第二','秒'],'sentence': ['句子','判决'],
        'sign': ['标志','签名'],'sink': ['水槽','下沉'],'sound': ['声音','健全'],
        'spirit': ['精神','烈酒'],'stamp': ['邮票','盖章'],'stand': ['站','摊位'],
        'state': ['状态','州'],'stick': ['棍子','粘贴'],'stock': ['股票','库存'],
        'store': ['商店','存储'],'strike': ['罢工','打击'],'table': ['桌子','表格'],
        'tear': ['眼泪','撕'],'tie': ['领带','系'],'tip': ['小费','尖端'],
        'train': ['火车','训练'],'trip': ['旅行','绊倒'],'watch': ['手表','观看'],
        'wave': ['波浪','挥手'],'yard': ['院子','码'],
    }
    print(f"  Fallback: {len(ecd_senses)} words")

# ── LaBSE anchors ──
labse_pairs = []
try:
    with open('/workspace/labse_anchors.py') as f:
        labse_pairs = eval(f.read().split('=', 1)[1].strip())
    print(f"LaBSE anchors: {len(labse_pairs)}")
except:
    print("LaBSE: 0 (not found)")

# ── Manual anchors ──
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

anchor_list = []; seen_en = set()
for e, z in labse_pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi) and ei[0].item() not in seen_en:
        seen_en.add(ei[0].item())
        anchor_list.append((e, z, torch.tensor(ei), torch.tensor(zi)))
for e, z in MANUAL:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi) and ei[0].item() not in seen_en:
        seen_en.add(ei[0].item())
        anchor_list.append((e, z, torch.tensor(ei), torch.tensor(zi)))
N_single = len(anchor_list)
print(f"Single-sense anchors: {N_single}")

# ── Build en_word → index/zh map ──
en_to_zh = {}
for en_w, zh_w, ei, zi in anchor_list:
    en_to_zh[en_w] = zh_w

# ── Find overlap with ECDICT ──
candidates = []
for en_w, senses in ecd_senses.items():
    if en_w not in en_to_zh:
        continue
    existing_zh = en_to_zh[en_w]
    additional = []
    for s in senses:
        s = s.strip()
        if not s: continue
        if s == existing_zh: continue
        zi = sp.encode_as_ids(s)
        if ok(zi):
            additional.append((s, zi))
    if additional:
        candidates.append((en_w, existing_zh, additional))
print(f"ECDICT overlap candidates: {len(candidates)}")

# ── Score distinctness ──
with torch.no_grad():
    cand_scores = []
    for en_w, existing_zh, additional in candidates:
        zi_ex = torch.tensor(sp.encode_as_ids(existing_zh), device=device, dtype=torch.long)
        hw_ex = F.normalize(heap_world(zi_ex).mean(0), dim=-1)
        valid = []
        total_dist = 0.0
        for zh_s, zi in additional:
            zi_t = torch.tensor(zi, device=device, dtype=torch.long)
            hw_z = F.normalize(heap_world(zi_t).mean(0), dim=-1)
            d = (1.0 - F.cosine_similarity(hw_ex.unsqueeze(0), hw_z.unsqueeze(0))).item()
            valid.append((zh_s, zi, d))
            total_dist += d
        if valid:
            avg_dist = total_dist / len(valid)
            cand_scores.append((avg_dist, en_w, existing_zh, valid))
cand_scores.sort(key=lambda x: -x[0])

TOP_K = min(50, len(cand_scores))
selected = cand_scores[:TOP_K]
print(f"Selected top {TOP_K} multi-sense words:")
for avg_d, en_w, ezh, senses in selected[:12]:
    print(f"  {en_w:15s} [{ezh}] d={avg_d:.3f}  +{len(senses)}: {[s[0] for s in senses[:3]]}")
if TOP_K > 12:
    print(f"  ... and {TOP_K-12} more")

# ── Create path vectors ──
MAX_PATHS = 3
multi_paths = nn.ParameterDict()
multi_info = {}  # en_w -> {'en_ids':..., 'L0_vec':..., 'anchor_entries': [(ai, path_k, zh_w)]}

for avg_d, en_w, existing_zh, senses in selected:
    n_paths = min(MAX_PATHS, len(senses))
    multi_paths[en_w] = nn.Parameter(torch.randn(n_paths, d, device=device) * 0.001)
    info = {
        'en_ids': torch.tensor(sp.encode_as_ids(en_w), device=device),
        'L0_vec': F.normalize(L0.weight[torch.tensor(sp.encode_as_ids(en_w)[0], device=device)], dim=-1).detach(),
        'entries': []
    }
    for k in range(n_paths):
        zh_s, zi, dist = senses[k]
        zi_t = torch.tensor(zi, device=device, dtype=torch.long)
        anchor_list.append((en_w, zh_s, info['en_ids'], zi_t))
        ai = len(anchor_list) - 1
        info['entries'].append((ai, k, zh_s))
    multi_info[en_w] = info

N = len(anchor_list)
total_paths = sum(p.shape[0] for p in multi_paths.values())
print(f"Total anchors: {N} ({N_single} single + {N-N_single} multi-sense)")
print(f"Learnable path vectors: {total_paths} across {len(multi_paths)} words")

# ── Precompute zh_world ──
zh_world_mat = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_list):
        zh_world_mat[ai] = F.normalize(heap_world(zi.to(device)).mean(0), dim=-1)

# ── Precompute en_world for single-sense ──
en_world_pre = torch.zeros(N_single, d, device=device)
with torch.no_grad():
    for ai in range(N_single):
        ei = anchor_list[ai][2].to(device)
        en_world_pre[ai] = F.normalize(heap_world(ei).mean(0), dim=-1)

# ── Gold eval ──
GOLD = [
    ('i','我'),('one','一'),('two','二'),('hand','手'),('eye','眼'),('head','头'),('heart','心'),
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
    ('dark','黑暗'),('empty','空'),('kind','善良'),('brave','勇敢'),
]
ge_ids, gz_ids = [], []
for e, z in GOLD:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi):
        ge_ids.append(torch.tensor(ei, device=device))
        gz_ids.append(torch.tensor(zi, device=device))
gw = torch.zeros(len(gz_ids), d, device=device)
with torch.no_grad():
    for i, zi in enumerate(gz_ids):
        gw[i] = F.normalize(heap_world(zi).mean(0), dim=-1)

def eval_gold():
    with torch.no_grad():
        c = 0
        for i, ei in enumerate(ge_ids):
            hw = F.normalize(heap_world(ei).mean(0), dim=-1)
            if (hw.unsqueeze(0) @ gw.T / tau).squeeze(0).argmax() == i: c += 1
        return 100 * c / len(ge_ids)

# ── Freeze model ──
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False
for tn in t_nodes:
    for p in tn.parameters(): p.requires_grad = False

# ── Train ──
all_path_params = list(multi_paths.parameters())
opt = torch.optim.Adam(all_path_params, lr=0.01)
BATCH_SINGLE = 28; BATCH_MULTI = 4; EPOCHS = 3000
single_ids = list(range(N_single))
multi_words = list(multi_info.keys())
gold0 = eval_gold()

# Init multi_acc
with torch.no_grad():
    init_correct, init_total = 0, 0
    for en_w, info in multi_info.items():
        pvecs = multi_paths[en_w]
        for ai, pk, zh_w in info['entries']:
            q = cmul(info['L0_vec'].unsqueeze(0), pvecs[pk:pk+1])
            logits = (q @ zh_world_mat.T / tau).squeeze(0)
            if logits.argmax() == ai: init_correct += 1
            init_total += 1
    init_multi_acc = 100 * init_correct / init_total if init_total > 0 else 0.0

print(f"\nINIT: gold={gold0:.1f}% multi_acc={init_multi_acc:.1f}% ({init_correct}/{init_total})")

for ep in range(EPOCHS):
    random.shuffle(single_ids)
    ep_loss, ep_n = 0.0, 0
    for bi in range(0, len(single_ids), BATCH_SINGLE):
        batch_s = single_ids[bi:bi + BATCH_SINGLE]
        opt.zero_grad()
        total_loss = torch.tensor(0.0, device=device); total_n = 0

        q_s = en_world_pre[batch_s]
        logits_s = (q_s @ zh_world_mat.T) / tau
        targets_s = torch.tensor(batch_s, device=device)
        total_loss += F.cross_entropy(logits_s, targets_s) * len(batch_s)
        total_n += len(batch_s)

        for _ in range(BATCH_MULTI):
            en_w = random.choice(multi_words)
            info = multi_info[en_w]; pvecs = multi_paths[en_w]
            ai, pk, _ = random.choice(info['entries'])
            q_m = cmul(info['L0_vec'].unsqueeze(0), pvecs[pk:pk+1])
            logits_m = (q_m @ zh_world_mat.T) / tau
            total_loss += F.cross_entropy(logits_m, torch.tensor([ai], device=device))
            total_n += 1

        if total_n > 0:
            (total_loss / total_n).backward()
            torch.nn.utils.clip_grad_norm_(all_path_params, 1.0)
            opt.step()
            ep_loss += (total_loss / total_n).item()
            ep_n += 1

    if ep % 100 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            gold = eval_gold()
            cos_paths = []
            for en_w in multi_info:
                pn = F.normalize(multi_paths[en_w], dim=-1)
                if pn.shape[0] >= 2:
                    for i in range(pn.shape[0]):
                        for j in range(i + 1, pn.shape[0]):
                            cos_paths.append((pn[i] * pn[j]).sum().item())
            avg_cos = sum(cos_paths) / len(cos_paths) if cos_paths else 0.0
            corr, tot = 0, 0
            for en_w, info in multi_info.items():
                pvecs = multi_paths[en_w]
                for ai, pk, _ in info['entries']:
                    q = cmul(info['L0_vec'].unsqueeze(0), pvecs[pk:pk+1])
                    logits = (q @ zh_world_mat.T / tau).squeeze(0)
                    if logits.argmax() == ai: corr += 1
                    tot += 1
            multi_acc = 100 * corr / tot if tot > 0 else 0.0
            print(f"  ep {ep:4d} loss={ep_loss/max(ep_n,1):.4f} gold={gold:.1f}% cos_path={avg_cos:.4f} multi_acc={multi_acc:.1f}% ({corr}/{tot})")

print(f"\n=== Results ===")
gold_final = eval_gold()
with torch.no_grad():
    far_cos = []
    for en_w in multi_info:
        pn = F.normalize(multi_paths[en_w], dim=-1)
        if pn.shape[0] >= 2:
            for i in range(pn.shape[0]):
                for j in range(i + 1, pn.shape[0]):
                    far_cos.append((pn[i] * pn[j]).sum().item())
    final_avg_cos = sum(far_cos) / len(far_cos) if far_cos else 0.0
    fcorr, ftot = 0, 0
    for en_w, info in multi_info.items():
        pvecs = multi_paths[en_w]
        for ai, pk, _ in info['entries']:
            q = cmul(info['L0_vec'].unsqueeze(0), pvecs[pk:pk+1])
            logits = (q @ zh_world_mat.T / tau).squeeze(0)
            if logits.argmax() == ai: fcorr += 1
            ftot += 1
    final_multi_acc = 100 * fcorr / ftot if ftot > 0 else 0.0

print(f"gold:            {gold0:.1f}% → {gold_final:.1f}%")
print(f"cos_path:        {final_avg_cos:.4f}")
print(f"multi_acc:       {init_multi_acc:.1f}% → {final_multi_acc:.1f}% ({fcorr}/{ftot})")
print(f"Done in {time.time()-t0:.1f}s")
