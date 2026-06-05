"""
词级翻译: 每个 EN 词通过 heap_world 查最近 ZH 词 → 是否命中参考
"""
import torch, torch.nn as nn, torch.nn.functional as F, re
import sentencepiece as spm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5; V = 16000; tau = 0.07
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x != 0 for x in ids)

L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])

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

# ── 构建 ZH 候选词表 (手选锚点) ──
MANUAL = [
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),('we','我们'),('they','他们'),
    ('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),('ten','十'),
    ('hand','手'),('eye','眼'),('head','头'),('heart','心'),('water','水'),('fire','火'),
    ('sun','太阳'),('moon','月'),('mountain','山'),('river','河'),('sky','天'),('earth','地'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),('tree','树'),('flower','花'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),('fish','鱼'),('bird','鸟'),
    ('horse','马'),('red','红'),('green','绿'),('white','白'),('black','黑'),
    ('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
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
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),('manage','管理'),('save','保存'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),('happen','发生'),('appear','出现'),
    ('continue','继续'),('reduce','减少'),('increase','增加'),
    ('big','大'),('small','小'),('good','好'),('bad','坏'),('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),('fast','快'),('slow','慢'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),('sad','悲伤'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),('clean','干净'),('safe','安全'),('easy','容易'),
    ('important','重要'),('free','自由'),('fair','公平'),('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),
    ('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
]

zh_words = []
zh_worlds = []
en_to_zh_true = {}
for en, zh in MANUAL:
    ei = sp.encode_as_ids(en); zi = sp.encode_as_ids(zh)
    if ok(ei) and ok(zi):
        zh_words.append(zh)
        zh_worlds.append(F.normalize(heap_world(torch.tensor(zi, device=device)).mean(dim=0), dim=-1))
        en_to_zh_true[en] = zh
zh_mat = torch.stack(zh_worlds)
print(f"ZH candidates: {len(zh_words)}")

def model_prediction(en_word):
    """用 heap_world 查最近 ZH 词"""
    ei = sp.encode_as_ids(en_word)
    if not ok(ei): return None
    hw = F.normalize(heap_world(torch.tensor([ei[0]], device=device)).mean(dim=0), dim=-1)
    sims = (hw.unsqueeze(0) @ zh_mat.T).squeeze(0)
    best = sims.argmax().item()
    return zh_words[best]

# ── 词级评测 ──
test_sents = [
    ("the sun is hot today",         "太阳很热今天"),
    ("i love to eat bread",          "我爱吃面包"),
    ("he drinks milk every day",     "他每天喝牛奶"),
    ("the river is long and deep",   "河又长又深"),
    ("she gave me a red flower",     "她给了我一朵红花"),
    ("we walk to the big mountain",  "我们走到大山"),
    ("they live in a new house",     "他们住在新房子里"),
    ("my mother bought rice and fish","我母亲买了米和鱼"),
    ("the old man died last night",  "老人昨晚死了"),
    ("i want to see the moon and stars","我想看月亮和星星"),
    ("fire is hot and water is cold","火是热的水是冷的"),
    ("the bird flies in the sky",    "鸟在天上飞"),
    ("he cut the bread with a knife","他用刀切面包"),
    ("she has a strong heart",       "她有一颗坚强的心"),
    ("we drink tea every morning",   "我们每天早上喝茶"),
    ("it is a good day",             "今天是个好天"),
    ("they kill the big fish",       "他们杀大鱼"),
    ("come and see the gold star",   "来看金星"),
    ("the new year is here",         "新年到了"),
    ("love is strong and true",      "爱是坚强而真实的"),
]

def zh_chars(s):
    return re.findall(r'[\u4e00-\u9fff]+', s)

total = 0
correct = 0

for en_s, zh_ref_full in test_sents:
    ref_chars = zh_chars(zh_ref_full)  # all ZH words in reference
    ref_str = zh_ref_full  # use full string for checking
    for word in en_s.split():
        w = word.strip('.,!?;:()[]"')
        pred = model_prediction(w)
        if pred is None: continue
        total += 1
        hit = pred in ref_str  # check if predicted ZH appears anywhere in reference
        if hit: correct += 1
        true_zh = en_to_zh_true.get(w, '?')
        mark = '✓' if hit else '✗'
        print(f"{w:12s} → {pred:10s} | {mark} | true={true_zh}")

print(f"\n=== 词级模型准确率 ===")
print(f"  Correct: {correct}/{total} = {100*correct/max(total,1):.1f}%")
print(f"  (Each EN word → heap_world → 1-NN over {len(zh_words)} ZH candidates)")
