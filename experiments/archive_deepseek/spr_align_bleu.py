"""
词对齐 BLEU: 把 EN 句中的锚点词替换为已知 ZH 翻译, 保留其他词不变.
评测: 模型能做到的最优情况是什么? (词级完美对齐时的 BLEU 上界)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, math, json
from collections import defaultdict, Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07; MAX_LEN = 50
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

# ── 模型 ──
L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
for p in L0.parameters(): p.requires_grad = False
for p in t_merge.parameters(): p.requires_grad = False

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
with open('/workspace/labse_anchors.py') as f: labse_pairs = eval(f.read().split('=',1)[1].strip())
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)
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
    ('big','大'),('small','小'),('good','好'),('bad','坏'),('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),('fast','快'),('slow','慢'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('sad','悲伤'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),('clean','干净'),('safe','安全'),('easy','容易'),
    ('important','重要'),('free','自由'),('fair','公平'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),
    ('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),]

ALL = labse_pairs + MANUAL
en_to_zh = {e.lower(): z for e, z in ALL}

# ── 加载测试句 ──
pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' not in l: continue
        en, zh = l.strip().split('\t', 1)
        en = en.strip().lower()
        zh = zh.strip()
        pairs.append((en, zh))
print(f"Test sentences: {len(pairs)}")

def compute_bleu(hypotheses, references):
    total_prec = [0.0] * 4
    total_hyp_len = 0
    total_ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        total_hyp_len += len(hyp_words)
        total_ref_len += len(ref_words)
        for n in range(1, 5):
            hyp_ngrams = Counter([' '.join(hyp_words[i:i+n]) for i in range(len(hyp_words)-n+1)])
            ref_ngrams = Counter([' '.join(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hyp_words) - n + 1, 1)
            if total > 0:
                total_prec[n-1] += min(overlap / total, 1.0)
    if total_hyp_len == 0: return 0.0
    bp = min(1.0, total_hyp_len / max(total_ref_len, 1))
    precs = [p / len(hypotheses) for p in total_prec]
    if any(p <= 0 for p in precs[:2]): return 0.0
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in precs) / 4)

def oracle_replace(en_text, zh_ref):
    """
    词对齐 BLEU 上界: 把 EN 句中已知锚点词替换为 ZH 翻译.
    非锚点词保持 EN (oracle: 至少保留原词的 n-gram 结构).
    """
    words = en_text.split()
    zh_words = []
    replaced = 0
    for w in words:
        w_clean = w.strip('.,!?;:"()[]')
        if w_clean in en_to_zh:
            zh_words.append(en_to_zh[w_clean])
            replaced += 1
        else:
            zh_words.append(w)  # keep EN word if no translation
    return ' '.join(zh_words), replaced / max(len(words), 1)

def model_replace(en_text):
    """
    模型翻译: BPE token → 1-NN in anchor ZH matrix → decode → concat
    (当前方法, 已知 BLEU=0)
    """
    toks = sp.encode_as_ids(en_text)
    zh_chars = []
    for tok in toks[:MAX_LEN]:
        if tok < 3: continue
        hw = F.normalize(heap_world(torch.tensor([tok], device=device)).mean(dim=0), dim=-1)
        # 1-NN over anchor ZH world
        best_sim = -1
        best_zh = ''
        for ai, (_, zh_w, _, _) in enumerate(anchor_ts if 'anchor_ts' in dir() else []):
            pass  # placeholder
        zh_chars.append(sp.decode([tok]))  # bad: just echo the BPE token
    
    return ' '.join(zh_chars)

# ── 评测 ──
N = 100
test_subset = pairs[-N:]

# A. Oracle: 完美词替换 (已知映射)
hyp_oracle = []
hyp_none = []  # 无替换
replace_ratios = []
for en, zh in test_subset:
    hyp_o, ratio = oracle_replace(en, zh)
    hyp_oracle.append(hyp_o)
    hyp_none.append(en)  # just copy EN as-is
    replace_ratios.append(ratio)

bleu_oracle = compute_bleu(hyp_oracle, [zh for _, zh in test_subset])
bleu_none = compute_bleu(hyp_none, [zh for _, zh in test_subset])
avg_ratio = sum(replace_ratios) / len(replace_ratios) * 100

print(f"\n=== 词对齐 BLEU ({N} sentences) ===")
print(f"  Avg anchor coverage: {avg_ratio:.1f}% of EN words in anchor set")
print(f"  Oracle (完美词替换): BLEU = {bleu_oracle*100:.2f}")
print(f"  Identity (EN unchanged): BLEU = {bleu_none*100:.2f}")
print(f"  Upper bound gain from anchors: +{bleu_oracle*100 - bleu_none*100:.2f} BLEU points")

# B. Show example
print(f"\n  Example:")
en, zh = test_subset[0]
hyp_o, _ = oracle_replace(en, zh)
print(f"    EN:     {en[:100]}")
print(f"    ZH ref: {zh[:100]}")
print(f"    Oracle: {hyp_o[:100]}")

print(f"\n=== 结论 ===")
print(f"完美词替换的 BLEU = {bleu_oracle*100:.2f}，锚点覆盖率 = {avg_ratio:.1f}%")
print(f"即使知道每个词的完美翻译，词对齐 BLEU 也只有 {bleu_oracle*100:.2f}——因为句法差异远大于词级替换。")
print(f"这解释了为什么逐词 1-NN 的 BLEU = 0。")
print("Done.")
