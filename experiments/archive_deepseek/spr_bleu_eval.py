"""
独立 path BLEU 评测: 逐词翻译 → BLEU vs 参考 ZH
比较: L0 单义 (baseline) vs L0+多义 path (new)
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json
import sentencepiece as spm, re
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07; MAX_LEN = 50
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

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
def cmul(a, b):
    aN = F.normalize(a, dim=-1); bN = F.normalize(b, dim=-1)
    aL, aR = aN[..., :d//2], aN[..., d//2:]
    bL, bR = bN[..., :d//2], bN[..., d//2:]
    return torch.cat([aL*bL - aR*bR, aL*bR + aR*bL], -1)

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

# ── 构建锚点 + 独立 128D path ──
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
print(f"Multi-sense words: {len(multis)}")

multi_en = set(multis.keys())
anchor_raw = []
for e, z in ALL:
    if e.lower() in multi_en: continue
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchor_raw.append((e, z, ei, zi))

# Paths: direct nn.ParameterDict
paths = nn.ParameterDict()
path_meta = {}
en_to_ais = defaultdict(list)
for en, zhs in multis.items():
    en_ids = sp.encode_as_ids(en)
    if not ok(en_ids): continue
    for pi, zh in enumerate(zhs[:3]):
        zh_ids = sp.encode_as_ids(zh)
        if ok(en_ids) and ok(zh_ids):
            anchor_raw.append((en, zh, en_ids, zh_ids))
            ai = len(anchor_raw) - 1
            pname = f"{en}_{pi}"
            paths[pname] = nn.Parameter(torch.randn(d, device=device) * 0.001)
            path_meta[pname] = (en, zh, ai)
            en_to_ais[en].append((ai, zh))

anchor_ts = [(e, z, torch.tensor(ei, device=device), torch.tensor(zi, device=device)) for e, z, ei, zi in anchor_raw]
N = len(anchor_ts)

# ZH world matrix
zh_world = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (_, _, _, zi) in enumerate(anchor_ts):
        zh_world[ai] = F.normalize(heap_world(zi).mean(dim=0), dim=-1)

# Pre-compute L0 vecs for multi-sense words
l0_vecs = {}
for en in multi_en:
    ids = sp.encode_as_ids(en)
    if ok(ids): l0_vecs[en] = F.normalize(L0.weight[ids[0]], dim=-1).detach()

# ── 训练独立 path (快速训练, 100 epochs) ──
print(f"Training {len(paths)} independent paths...")
opt = torch.optim.Adam(list(paths.parameters()), lr=0.01)
pnames = list(paths.keys())
single_ai = [ai for ai, (en, _, _, _) in enumerate(anchor_ts) if en not in multi_en]

for ep in range(100):
    random.shuffle(single_ai)
    for bi in range(0, len(single_ai), 32):
        batch = single_ai[bi:bi+32]; opt.zero_grad()
        losses = []
        for ai in batch:
            ei = anchor_ts[ai][2]
            hw = F.normalize(heap_world(ei).mean(dim=0), dim=-1)
            logits = (hw.unsqueeze(0) @ zh_world.T) / tau
            losses.append(F.cross_entropy(logits, torch.tensor([ai], device=device)))
        for pname in random.sample(pnames, min(4, len(pnames))):
            en, zh, ai = path_meta[pname]
            if en in l0_vecs:
                p = F.normalize(paths[pname], dim=-1)
                out = cmul(l0_vecs[en].unsqueeze(0), p.unsqueeze(0)).squeeze(0)
                logits = (out.unsqueeze(0) @ zh_world.T) / tau
                losses.append(F.cross_entropy(logits, torch.tensor([ai], device=device)))
        if losses:
            (sum(losses) / len(losses)).backward()
            torch.nn.utils.clip_grad_norm_(list(paths.parameters()), 1.0)
            opt.step()
    if ep % 50 == 0 or ep == 99:
        with torch.no_grad():
            c = 0
            for pname in pnames:
                en, zh, ai = path_meta[pname]
                if en not in l0_vecs: continue
                p = F.normalize(paths[pname], dim=-1)
                out = cmul(l0_vecs[en].unsqueeze(0), p.unsqueeze(0)).squeeze(0)
                logits = (out.unsqueeze(0) @ zh_world.T / tau).squeeze(0)
                if logits.argmax() == ai: c += 1
            print(f"  ep {ep:3d} ms_train_acc={100*c/len(pnames):.0f}%")

print(f"Paths trained: ms_train_acc={100*c/len(pnames):.0f}%")

# ── BLEU 评测 ──
print(f"\n=== BLEU 评测 ===")

# 加载测试集 (newstest)
test_src = []
test_ref = []
try:
    with open('/mnt/nas/datasets/wmt17/newstest2017.tc.en') as f:
        test_src = [l.strip() for l in f]
    with open('/mnt/nas/datasets/wmt17/newstest2017.tc.zh') as f:
        test_ref = [l.strip() for l in f]
except:
    pass

if not test_src:
    # Fallback: use last 200 parallel sentences as test
    test_pairs = []
    with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
        for l in f:
            if '\t' not in l: continue
            zh, en = l.strip().split('\t', 1)
            test_pairs.append((en.strip(), zh.strip()))
    test_src = [en for en, zh in test_pairs[-200:]]
    test_ref = [zh for en, zh in test_pairs[-200:]]
    print(f"Using last {len(test_src)} parallel sentences as test set")

# ── 翻译策略 ──
# 预构建 EN word → ZH target map (单义用)
en_to_zh_known = {e.lower(): z for e, z in ALL}

# 预构建 EN first BPE token → anchor indices map
tok_to_ais = defaultdict(list)
for ai, (en, zh, ei, zi) in enumerate(anchor_ts):
    tok_to_ais[ei[0].item()].append(ai)

# 预构建 ZH token → word (decode)  
zh_cache = {}
def get_zh_word(ai):
    if ai not in zh_cache:
        zi = anchor_ts[ai][3]
        zh_cache[ai] = sp.decode(zi.tolist())
    return zh_cache[ai]

def translate_sentence(en_str, use_paths=True):
    """逐词翻译: 每个 EN token → 最近 ZH 词"""
    toks = sp.encode_as_ids(en_str.lower())
    en_t = torch.tensor(toks[:MAX_LEN], device=device)
    
    zh_tokens = []
    for pos, tok in enumerate(toks[:MAX_LEN]):
        if tok < 3: continue  # skip PAD/BOS/EOS
        
        # Get EN world position
        hw = F.normalize(heap_world(torch.tensor([tok], device=device)).mean(dim=0), dim=-1)
        
        # Check if this token has multi-sense paths
        en_word = sp.decode([tok]).strip()
        best_ai = None
        best_sim = -1
        
        if use_paths and en_word in l0_vecs:
            # Try all multi-sense paths for this word
            for ai, zh in en_to_ais.get(en_word, []):
                # Find which path matches this sense
                for pname in pnames:
                    en_p, zh_p, ai_p = path_meta[pname]
                    if en_p == en_word and ai_p == ai:
                        p = F.normalize(paths[pname], dim=-1)
                        cw = cmul(l0_vecs[en_word].unsqueeze(0), p.unsqueeze(0)).squeeze(0)
                        sim = F.cosine_similarity(F.normalize(cw, dim=-1).unsqueeze(0),
                                                  zh_world[ai].unsqueeze(0)).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_ai = ai
                        break
        
        if best_ai is None:
            # Fall back to L0 single-sense
            logits = (hw.unsqueeze(0) @ zh_world.T / tau).squeeze(0)
            best_ai = logits.argmax().item()
        
        if best_ai is not None:
            zh_w = get_zh_word(best_ai)
            zh_tokens.append(zh_w)
    
    return ''.join(zh_tokens)

def compute_bleu(hypotheses, references):
    """简易 BLEU: n-gram precision, n=1..4, brevity penalty"""
    from collections import Counter
    
    total_prec = [0.0] * 4
    total_hyp_len = 0
    total_ref_len = 0
    
    for hyp, ref in zip(hypotheses, references):
        hyp_words = list(hyp)
        ref_words = list(ref)
        total_hyp_len += len(hyp_words)
        total_ref_len += len(ref_words)
        
        for n in range(1, 5):
            hyp_ngrams = Counter([''.join(hyp_words[i:i+n]) for i in range(len(hyp_words)-n+1)])
            ref_ngrams = Counter([''.join(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hyp_words) - n + 1, 1)
            if total > 0:
                total_prec[n-1] += min(overlap / total, 1.0)
    
    if total_hyp_len == 0: return 0.0
    
    # Brevity penalty
    bp = min(1.0, total_hyp_len / max(total_ref_len, 1))
    if bp <= 0: return 0.0
    
    # Geometric mean of n-gram precisions
    import math
    precs = [p / max(len(hypotheses), 1) for p in total_prec]
    if any(p == 0 for p in precs[:2]): return 0.0
    score = bp * math.exp(sum(math.log(p) for p in precs) / 4)
    return score

# ── 跑评测 ──
N_TEST = min(100, len(test_src))
print(f"\nTesting on {N_TEST} sentences...")

# L0 单义 (baseline)
hyp_L0 = []
for i in range(N_TEST):
    hyp_L0.append(translate_sentence(test_src[i], use_paths=False))
    if i == 0:
        print(f"\n  Example 1:")
        print(f"    EN: {test_src[i][:80]}")
        print(f"    ZH (ref): {test_ref[i][:80]}")
        print(f"    L0:       {hyp_L0[-1][:80]}")

bleu_L0 = compute_bleu(hyp_L0, test_ref[:N_TEST])

# L0 + 多义 path
hyp_multi = []
for i in range(N_TEST):
    hyp_multi.append(translate_sentence(test_src[i], use_paths=True))
    if i == 0:
        print(f"    L0+path:  {hyp_multi[-1][:80]}")

bleu_multi = compute_bleu(hyp_multi, test_ref[:N_TEST])

# 随机选候选作为上界参考
import random
random.seed(42)
hyp_random = []
for i in range(N_TEST):
    zh_tok = []
    toks = sp.encode_as_ids(test_src[i].lower())
    for tok in toks[:MAX_LEN]:
        if tok < 3: continue
        # Random ZH word
        ai = random.randint(0, N-1)
        zh_tok.append(get_zh_word(ai))
    hyp_random.append(''.join(zh_tok))
bleu_rand = compute_bleu(hyp_random, test_ref[:N_TEST])

print(f"\n{'='*50}")
print(f"BLEU Results ({N_TEST} sentences)")
print(f"{'='*50}")
print(f"  L0 single-sense:        BLEU = {bleu_L0*100:.2f}")
print(f"  L0 + multi-sense paths: BLEU = {bleu_multi*100:.2f}")
print(f"  Random baseline:        BLEU = {bleu_rand*100:.2f}")
print(f"\n  Δ = {bleu_multi*100 - bleu_L0*100:+.2f} BLEU points")
print("Done.")
