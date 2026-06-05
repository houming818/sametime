"""
Token 聚类分析——按 L0 堆树路径　
路径 = (token_id // V/2^l) % 2^l for l=0..4
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, numpy as np
from collections import defaultdict, Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128; td = 5
sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()

L0 = nn.Embedding(V, d).to(device)
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
t_merge = nn.Linear(d, d).to(device)

ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt',
                  map_location=device, weights_only=True)
L0.load_state_dict(ckpt['L0'])
t_merge.load_state_dict(ckpt['t_merge'])
for i, tn in enumerate(t_nodes):
    tn.load_state_dict(ckpt['t_nodes'][i])

def get_path(tok_id):
    path = []
    for l in range(td):
        nidx = (tok_id // (V // (2 ** l))) % (2 ** l) if l > 0 else 0
        path.append(nidx)
    return tuple(path)

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    return w

def heap_world(tok_ids):
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    w = F.normalize(tw_vec(tok_ids), dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    return t_merge(torch.cat([tL * wL - tR * wR, tL * wR + tR * wL], -1))

# ── Path distribution ──
print("=== L0 路径分布 (token_id hard routing) ===\n")
path_counts = Counter()
path_tokens = defaultdict(list)
for tid in range(3, V):
    p = get_path(tid)
    path_counts[p] += 1
    path_tokens[p].append(tid)

n_unique = len(path_counts)
print(f"Total tokens: {V - 3}  (PAD=0, BOS=1, EOS=2 excluded)")
print(f"Unique paths: {n_unique}  (max possible: 2^0 × 2^1 × 2^2 × 2^3 × 2^4 = 1×2×4×8×16 = 1024)")
print(f"Tokens per path: mean={np.mean(list(path_counts.values())):.1f} median={np.median(list(path_counts.values())):.0f} max={max(path_counts.values())}")
print()

# Top paths (most tokens)
print("Top 10 paths by token count:")
for p, cnt in path_counts.most_common(10):
    samples = [sp.decode([tid]) for tid in path_tokens[p][:5]]
    print(f"  path={p} n={cnt:4d} samples={samples}")

# ── Path vector cosine matrix ──
print(f"\n=== 路径向量的余弦相似度 ===")
unique_paths = sorted(path_counts.keys())[:200]
path_vecs = []
for p in unique_paths:
    token_ids = torch.tensor(path_tokens[p][:5], device=device)  # take up to 5
    pv = F.normalize(tw_vec(token_ids).mean(dim=0), dim=-1)
    path_vecs.append(pv)
pv_mat = torch.stack(path_vecs)  # [n_paths, d]
cos_mat = pv_mat @ pv_mat.T
mask = ~torch.eye(len(unique_paths), dtype=torch.bool, device=device)
off_diag = cos_mat[mask]
print(f"  {len(unique_paths)} paths: mean_off_cos={off_diag.mean():.4f} max={off_diag.max():.4f} min={off_diag.min():.4f}")

# ── Node norms ──
print(f"\n=== 节点 norm ===")
for l in range(td):
    n = t_nodes[l].weight
    norms = n.norm(dim=-1)
    print(f"  L{l} ({2**l:2d} nodes): norm=[{norms.min():.4f}, {norms.max():.4f}] mean={norms.mean():.4f}")

# ── Node pairwise cosine per level ──
print(f"\n=== 节点 pairwise cosine within each level ===")
for l in range(td):
    if 2**l == 1: continue
    nodes = t_nodes[l].weight  # [n, d]
    cos_w = F.cosine_similarity(nodes[:, None, :], nodes[None, :, :], dim=-1)
    mask_l = ~torch.eye(2**l, dtype=torch.bool, device=device)
    off = cos_w[mask_l]
    print(f"  L{l} ({2**l:2d} nodes): pairwise_cos=[{off.min():.4f}, {off.max():.4f}] mean={off.mean():.4f}")

# ── Inter-level cosine ──
print(f"\n=== 跨层节点 cosine (root vs others) ===")
root = t_nodes[0].weight[0]
for l in range(1, td):
    children = t_nodes[l].weight
    cos_val = F.cosine_similarity(root.unsqueeze(0), children)
    print(f"  root↔L{l} ({2**l:2d} children): cos=[{cos_val.min():.4f}, {cos_val.max():.4f}] mean={cos_val.mean():.4f}")

# ── EN/ZH token path overlap ──
print(f"\n=== EN/ZH 共享同一路径的锚点词对 ===")
from collections import defaultdict

def ok(ids):
    return all(x != 0 for x in ids)

MANUAL = [
    ('i', '我'), ('you', '你'), ('he', '他'), ('she', '她'), ('it', '它'),
    ('we', '我们'), ('they', '他们'), ('my', '我'), ('your', '你'), ('his', '他'),
    ('one', '一'), ('two', '二'), ('three', '三'), ('four', '四'), ('five', '五'),
    ('hand', '手'), ('eye', '眼'), ('head', '头'), ('heart', '心'),
    ('blood', '血'), ('body', '身体'), ('mouth', '口'), ('ear', '耳'),
    ('foot', '脚'), ('arm', '手臂'), ('leg', '腿'), ('hair', '头发'),
    ('water', '水'), ('fire', '火'), ('sun', '太阳'), ('moon', '月'),
    ('mountain', '山'), ('river', '河'), ('sky', '天'), ('earth', '地'),
    ('wind', '风'), ('rain', '雨'), ('snow', '雪'), ('sea', '海'),
    ('tree', '树'), ('flower', '花'), ('wood', '木'), ('stone', '石'),
    ('gold', '金'), ('iron', '铁'), ('star', '星'), ('cloud', '云'),
    ('fish', '鱼'), ('bird', '鸟'), ('horse', '马'), ('cow', '牛'),
    ('red', '红'), ('green', '绿'), ('white', '白'), ('black', '黑'),
    ('yellow', '黄'),
    ('mother', '母亲'), ('father', '父亲'), ('son', '儿子'), ('daughter', '女儿'),
    ('food', '食物'), ('rice', '米'), ('meat', '肉'), ('bread', '面包'),
    ('milk', '牛奶'), ('egg', '蛋'), ('fruit', '水果'),
    ('day', '天'), ('night', '夜'), ('year', '年'), ('month', '月'),
    ('spring', '春'), ('summer', '夏'), ('autumn', '秋'), ('winter', '冬'),
    ('go', '去'), ('come', '来'), ('eat', '吃'), ('drink', '喝'),
    ('see', '看'), ('hear', '听'), ('say', '说'), ('read', '读'),
    ('write', '写'), ('walk', '走'), ('sit', '坐'), ('stand', '站'),
    ('give', '给'), ('take', '拿'), ('make', '做'), ('know', '知道'),
    ('think', '想'), ('want', '要'), ('love', '爱'), ('like', '喜欢'),
    ('work', '工作'), ('play', '玩'), ('buy', '买'), ('sell', '卖'),
    ('live', '生活'), ('die', '死'), ('kill', '杀'), ('speak', '说'),
    ('talk', '谈'), ('find', '找'), ('learn', '学'), ('teach', '教'),
    ('help', '帮助'), ('open', '开'), ('close', '关'), ('start', '开始'),
    ('stop', '停'), ('run', '跑'), ('fly', '飞'), ('sing', '唱'),
    ('fight', '战斗'), ('win', '赢'), ('send', '送'), ('receive', '收'),
    ('remember', '记住'), ('forget', '忘记'), ('believe', '相信'),
    ('understand', '理解'), ('explain', '解释'), ('change', '改变'),
    ('grow', '生长'), ('build', '建设'), ('cut', '切'), ('break', '打破'),
    ('push', '推'), ('pull', '拉'), ('carry', '带'), ('throw', '扔'),
    ('catch', '抓'), ('follow', '跟随'), ('lead', '领导'), ('meet', '见面'),
    ('wait', '等'), ('need', '需要'), ('keep', '保持'),
    ('agree', '同意'), ('return', '返回'), ('leave', '离开'),
    ('enter', '进入'), ('arrive', '到达'), ('stay', '停留'), ('move', '移动'),
    ('pass', '通过'), ('rise', '上升'), ('fall', '下降'), ('drop', '掉'),
    ('lift', '举起'), ('burn', '烧'), ('choose', '选择'), ('decide', '决定'),
    ('prepare', '准备'), ('finish', '完成'), ('enjoy', '享受'),
    ('suffer', '受苦'), ('worry', '担心'), ('fear', '恐惧'),
    ('protect', '保护'), ('attack', '攻击'), ('defend', '防御'), ('destroy', '破坏'),
    ('create', '创造'), ('produce', '生产'), ('develop', '发展'), ('improve', '改善'),
    ('manage', '管理'), ('organize', '组织'), ('save', '保存'),
    ('disappear', '消失'), ('survive', '幸存'), ('own', '拥有'), ('exist', '存在'),
    ('happen', '发生'), ('appear', '出现'), ('continue', '继续'), ('reduce', '减少'),
    ('increase', '增加'), ('raise', '提高'), ('avoid', '避免'), ('expect', '期望'),
    ('big', '大'), ('small', '小'), ('good', '好'), ('bad', '坏'),
    ('new', '新'), ('old', '老'), ('hot', '热'), ('cold', '冷'),
    ('long', '长'), ('short', '短'), ('high', '高'), ('low', '低'),
    ('fast', '快'), ('slow', '慢'), ('beautiful', '美'), ('rich', '富'),
    ('strong', '强'), ('weak', '弱'), ('true', '真'), ('happy', '幸福'),
    ('sad', '悲伤'), ('young', '年轻'), ('heavy', '重'), ('light', '轻'),
    ('hard', '硬'), ('soft', '软'), ('deep', '深'), ('wide', '宽'),
    ('clean', '干净'), ('safe', '安全'), ('easy', '容易'), ('important', '重要'),
    ('free', '自由'), ('fair', '公平'), ('bright', '明亮'), ('dark', '黑暗'),
    ('dry', '干'), ('full', '满'), ('empty', '空'),
    ('kind', '善良'), ('brave', '勇敢'), ('afraid', '害怕'), ('angry', '愤怒'),
]

same_path = 0
diff_path = 0
for en_word, zh_word in MANUAL:
    ei = sp.encode_as_ids(en_word)
    zi = sp.encode_as_ids(zh_word)
    if not ok(ei) or not ok(zi): continue
    ep = get_path(ei[0])
    zp = get_path(zi[0])
    if ep == zp:
        same_path += 1
    else:
        diff_path += 1

print(f"  same path: {same_path}  diff path: {diff_path}")
print(f"  = {same_path / (same_path + diff_path) * 100:.1f}% of anchor pairs share the same tree path")

# ── Path diversity of all en/zh tokens ──
print(f"\n=== EN vs ZH 词性的路径分布 ===")
for label, tokens in [("EN", "the a an to in on is are was were of and it that for as with be at by from or this not but".split()),
                       ("ZH", "的是在和我人有来大自己不个们上到时就会说去也子能下过对可得里后那要这".split())]:
    paths = defaultdict(list)
    for word in tokens:
        ids = sp.encode_as_ids(word)
        if ok(ids):
            p = get_path(ids[0])
            paths[p].append(word)
    print(f"  {label}: {len(paths)} unique paths for {len(tokens)} words")
    for p, words in sorted(paths.items()):
        print(f"    path={p} → {words}")

print("\nDone.")
