"""
按树路径层级展示 token 分布 — 每个节点显示代表性 token
"""
import torch, torch.nn as nn, sentencepiece as spm
from collections import defaultdict

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size(); td = 5

def is_zh(ids):
    for tid in ids:
        try:
            t = sp.decode([tid])
            for ch in t:
                if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                    return True
        except:
            pass
    return False

def is_en(ids):
    try:
        t = sp.decode(ids)
        return all(ord(ch) < 128 and (ch.isalpha() or ch.isspace()) for ch in t if ch != ' ')
    except:
        return False

# 收集所有 token 的路径
paths = defaultdict(list)         # 完整路径 → token 列表
node_tokens = defaultdict(lambda: defaultdict(list))  # node_tokens[level][node_idx] → token 列表

for tid in range(3, V):
    path = []
    for l in range(td):
        if l == 0:
            nidx = 0
        else:
            nidx = (tid // (V // (2 ** l))) % (2 ** l)
        path.append(nidx)
        node_tokens[l][(l, nidx)].append(tid)
    paths[tuple(path)].append(tid)

print("=" * 80)
print("堆树路径 — Token 聚类分析")
print(f"V={V} tokens={V-3} levels={td} nodes={sum(2**l for l in range(td))}")
print("=" * 80)

# ── 树形展示 ──
print(f"\n{'── L0 ROOT (#0)':20s} all {V-3} tokens")
print(f"    │")
print(f"    ├── {'[L1] node#0':15s} ⋯  (left  branch)")
print(f"    └── {'[L1] node#1':15s} ⋯  (right branch)")

# ── 每层统计数据 ──
print("\n" + "=" * 80)
print("各层 token 分布统计")
print("=" * 80)

for l in range(td):
    n_nodes = 2 ** l
    print(f"\n─── Level {l} ({n_nodes:2d} nodes) ───")
    if l == 0:
        print(f"  node #{0:2d}: {V-3} tokens (ALL)")
        continue

    for n_idx in range(n_nodes):
        tids = node_tokens[l][(l, n_idx)]
        n_tok = len(tids)
        zh_count = sum(1 for tid in tids if is_zh([tid]))
        en_count = sum(1 for tid in tids if is_en([tid]))
        
        # 找代表性 token — 按频率排 (token_id 越小越常见)
        top_tids = sorted(tids)[:15]
        top_strs = []
        for tid in top_tids:
            word = sp.decode([tid])
            if len(word) <= 6:
                top_strs.append(word)
            if len(top_strs) >= 5:
                break
        if not top_strs:
            for tid in top_tids:
                word = sp.decode([tid])
                top_strs.append(word)
                if len(top_strs) >= 5:
                    break
        
        zh_ratio = zh_count / n_tok * 100 if n_tok > 0 else 0
        en_ratio = en_count / n_tok * 100 if n_tok > 0 else 0
        
        depth_prefix = "  " * (l + 1)
        print(f"{depth_prefix}node #{n_idx:2d}: {n_tok:5d} tokens  "
              f"ZH={zh_ratio:4.1f}% EN={en_ratio:4.1f}%  "
              f"samples: {top_strs}")

# ── 聚焦 L1 split (左右) — 权重相等分析 ──
print("\n" + "=" * 80)
print("L1 左右分叉 — 权重相等视角")
print("=" * 80)

L1_left = node_tokens[1][(1, 0)]
L1_right = node_tokens[1][(1, 1)]

for side_name, tids in [("LEFT (node#0)", L1_left), ("RIGHT (node#1)", L1_right)]:
    zh_count = sum(1 for tid in tids if is_zh([tid]))
    en_count = sum(1 for tid in tids if is_en([tid]))
    
    # 抽样展示: 每 1000 个取 1 个
    tids_sorted = sorted(tids)
    
    all_words = []
    for tid in tids_sorted:
        word = sp.decode([tid])
        if len(word) <= 8:
            all_words.append(word)
    
    print(f"\n  {side_name}: {len(tids)} tokens  ZH={zh_count}({zh_count/len(tids)*100:.1f}%) EN={en_count}({en_count/len(tids)*100:.1f}%)")
    print(f"    抽样展示 ({len(all_words)} words ≤8 chars):")
    for i in range(0, min(len(all_words), 40), 8):
        print(f"      {all_words[i:i+8]}")

# ── L2 分叉 — 16 nodes × 权重相等 ──
print("\n" + "=" * 80)
print("L2 四分叉 — 权重相等视角")
print("=" * 80)

for n_idx in range(4):
    tids = node_tokens[2][(2, n_idx)]
    zh_count = sum(1 for tid in tids if is_zh([tid]))
    en_count = sum(1 for tid in tids if is_en([tid]))
    
    tids_sorted = sorted(tids)
    all_words = [sp.decode([tid]) for tid in tids_sorted if len(sp.decode([tid])) <= 8]
    
    print(f"\n  node #{n_idx}: {len(tids)} tokens  ZH={zh_count}({zh_count/len(tids)*100:.1f}%) EN={en_count}({en_count/len(tids)*100:.1f}%)")
    for i in range(0, min(len(all_words), 40), 10):
        print(f"    {all_words[i:i+10]}")

# ── 锚点对分布 — 看 EN/ZH 各自落在哪个节点 ──
print("\n" + "=" * 80)
print("锚点对的路径分布 — EN 左 vs ZH 右？")
print("=" * 80)

ANCHORS = [
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),
    ('we','我们'),('they','他们'),('me','我'),('him','他'),
    ('my','我'),('your','你'),('his','他'),('her','她'),
    ('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),
    ('six','六'),('seven','七'),('eight','八'),('nine','九'),('ten','十'),
    ('hand','手'),('eye','眼'),('head','头'),('heart','心'),
    ('blood','血'),('body','身体'),('mouth','口'),('ear','耳'),
    ('foot','脚'),('arm','手臂'),('leg','腿'),('hair','头发'),
    ('water','水'),('fire','火'),('sun','太阳'),('moon','月'),
    ('mountain','山'),('river','河'),('sky','天'),('earth','地'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('wood','木'),('stone','石'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),
    ('fish','鱼'),('bird','鸟'),('horse','马'),('cow','牛'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),
    ('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),
    ('milk','牛奶'),('egg','蛋'),('fruit','水果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),
    ('spring','春'),('summer','夏'),('autumn','秋'),('winter','冬'),
    ('go','去'),('come','来'),('eat','吃'),('drink','喝'),
    ('see','看'),('hear','听'),('say','说'),('read','读'),
    ('write','写'),('walk','走'),('sit','坐'),('stand','站'),
    ('give','给'),('take','拿'),('make','做'),('know','知道'),
    ('think','想'),('want','要'),('love','爱'),('like','喜欢'),
    ('work','工作'),('play','玩'),('buy','买'),('sell','卖'),
    ('live','生活'),('die','死'),('kill','杀'),('speak','说'),
    ('talk','谈'),('find','找'),('learn','学'),('teach','教'),
    ('help','帮助'),('open','开'),('close','关'),('start','开始'),
    ('stop','停'),('run','跑'),('fly','飞'),
    ('remember','记住'),('forget','忘记'),('believe','相信'),
    ('understand','理解'),('explain','解释'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),
    ('push','推'),('pull','拉'),('carry','带'),('throw','扔'),
    ('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('need','需要'),('keep','保持'),
    ('agree','同意'),('return','返回'),('leave','离开'),
    ('enter','进入'),('arrive','到达'),('stay','停留'),('move','移动'),
    ('pass','通过'),('rise','上升'),('fall','下降'),('drop','掉'),
    ('lift','举起'),('burn','烧'),('choose','选择'),('decide','决定'),
    ('prepare','准备'),('finish','完成'),('enjoy','享受'),
    ('suffer','受苦'),('worry','担心'),('fear','恐惧'),
    ('protect','保护'),('attack','攻击'),('defend','防御'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),
    ('manage','管理'),('organize','组织'),('save','保存'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),
    ('happen','发生'),('appear','出现'),('continue','继续'),('reduce','减少'),
    ('increase','增加'),('raise','提高'),('avoid','避免'),('expect','期望'),
    ('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),
    ('fast','快'),('slow','慢'),('beautiful','美'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),
    ('sad','悲伤'),('young','年轻'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),
    ('free','自由'),('fair','公平'),('bright','明亮'),('dark','黑暗'),
    ('dry','干'),('full','满'),('empty','空'),
    ('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
]

def ok(ids): return all(x != 0 for x in ids)

for l in range(1, td):
    print(f"\nL{l} level ({2**l} nodes):")
    node_pairs = defaultdict(list)
    for en_word, zh_word in ANCHORS:
        ei = sp.encode_as_ids(en_word); zi = sp.encode_as_ids(zh_word)
        if not ok(ei) or not ok(zi): continue
        en_node = (ei[0] // (V // (2 ** l))) % (2 ** l)
        zh_node = (zi[0] // (V // (2 ** l))) % (2 ** l)
        node_pairs[(en_node, zh_node)].append((en_word, zh_word))
    
    for (en_n, zh_n), pairs_list in sorted(node_pairs.items()):
        show = pairs_list[:5]
        if len(pairs_list) > 5: show.append(f"...+{len(pairs_list)-5}")
        print(f"  EN_node#{en_n:2d} ←→ ZH_node#{zh_n:2d}: {len(pairs_list):3d} pairs  samples={show}")
