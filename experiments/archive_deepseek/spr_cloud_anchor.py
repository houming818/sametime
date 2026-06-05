"""
Cloud Anchor Codebook — sentence-context hidden states for K-sample anchor clouds
Step 1: Extract K=20 hidden samples per anchor from training sentences
Step 2: NN matching via cloud distances (min of K-sample distances)
Step 3: Evaluate anchor self-test + sentence-level token accuracy
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, numpy as np, random, math, time, sys, os
from collections import defaultdict, Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {}; i = 1
while i < len(sys.argv):
    if sys.argv[i].startswith('--'):
        k = sys.argv[i][2:]; v = sys.argv[i+1] if i+1 < len(sys.argv) else ''
        args[k] = v; i += 2
    else: i += 1

K = int(args.get('K', '20'))          # samples per anchor cloud
MAX_LEN = 50
SAVE_DIR = '/mnt/nas/datasets/wmt17/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── BPE ───
sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128

def ok(ids): return all(x != 0 for x in ids)

# ─── Model ───
class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
    def fd(self, x): return self.dp(self.dec(x)[0])

L0 = nn.Embedding(V, d).to(device)
L1 = BiGRU(d).to(device)

# ─── Anchor Words ───
ANCHOR_WORDS = [
    ('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),
    ('six','六'),('seven','七'),('eight','八'),('nine','九'),('ten','十'),
    ('hundred','百'),('thousand','千'),
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),
    ('we','我们'),('they','他们'),('me','我'),('him','他'),
    ('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),
    ('fast','快'),('slow','慢'),('rich','富'),('poor','穷'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),
    ('young','年轻'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),
    ('important','重要'),('free','自由'),('bright','明亮'),
    ('dark','黑暗'),('dry','干'),('full','满'),('empty','空'),
    ('same','相同'),('different','不同'),('real','真实'),
    ('correct','正确'),('wrong','错误'),
    ('water','水'),('fire','火'),('sun','太阳'),('moon','月'),
    ('mountain','山'),('river','河'),('sky','天'),('earth','地'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('wood','木'),('stone','石'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),
    ('lake','湖'),('ice','冰'),('sand','沙'),
    ('fish','鱼'),('bird','鸟'),('horse','马'),('cow','牛'),
    ('chicken','鸡'),('dragon','龙'),('bear','熊'),('wolf','狼'),
    ('red','红'),('green','绿'),('white','白'),('black','黑'),
    ('yellow','黄'),
    ('mother','母亲'),('father','父亲'),('son','儿子'),
    ('daughter','女儿'),('husband','丈夫'),('wife','妻子'),
    ('family','家'),('child','孩子'),('brother','兄弟'),
    ('parent','父母'),('baby','婴儿'),
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),
    ('milk','牛奶'),('egg','蛋'),('fruit','水果'),('sugar','糖'),
    ('salt','盐'),('oil','油'),('wine','酒'),('tea','茶'),
    ('coffee','咖啡'),('apple','苹果'),
    ('day','天'),('night','夜'),('year','年'),('month','月'),
    ('week','周'),('hour','小时'),('time','时间'),
    ('today','今天'),('tomorrow','明天'),('morning','早上'),
    ('evening','晚上'),('spring','春'),('summer','夏'),
    ('autumn','秋'),('winter','冬'),
    ('go','去'),('come','来'),('eat','吃'),('drink','喝'),
    ('see','看'),('hear','听'),('say','说'),('read','读'),
    ('write','写'),('walk','走'),('sit','坐'),('stand','站'),
    ('give','给'),('take','拿'),('make','做'),('know','知道'),
    ('think','想'),('want','要'),('love','爱'),('like','喜欢'),
    ('work','工作'),('play','玩'),('buy','买'),('sell','卖'),
    ('live','生活'),('die','死'),('kill','杀'),('speak','说'),
    ('talk','谈'),('find','找'),('learn','学'),('teach','教'),
    ('help','帮助'),('open','开'),('close','关'),('start','开始'),
    ('stop','停'),('run','跑'),('fly','飞'),('sing','唱'),
    ('fight','战斗'),('win','赢'),('send','送'),
    ('remember','记住'),('forget','忘记'),('believe','相信'),
    ('understand','理解'),('change','改变'),('grow','生长'),
    ('build','建设'),('cut','切'),('break','打破'),
    ('push','推'),('pull','拉'),('carry','带'),('throw','扔'),
    ('catch','抓'),('follow','跟随'),('lead','领导'),
    ('meet','见面'),('wait','等'),('hope','希望'),('try','尝试'),
    ('need','需要'),('keep','保持'),('agree','同意'),
    ('return','返回'),('leave','离开'),('enter','进入'),
    ('choose','选择'),('decide','决定'),('prepare','准备'),
    ('finish','完成'),('join','加入'),('share','分享'),
    ('protect','保护'),('attack','攻击'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),
    ('improve','改善'),('manage','管理'),('save','保存'),
    ('happen','发生'),('appear','出现'),('continue','继续'),
    ('reduce','减少'),('increase','增加'),
    ('book','书'),('door','门'),('window','窗'),('house','家'),
    ('car','车'),('road','路'),('city','城市'),('country','国家'),
    ('school','学校'),('hospital','医院'),('table','桌'),
    ('chair','椅'),('bed','床'),('phone','电话'),
    ('money','钱'),('war','战争'),('peace','和平'),
    ('man','男人'),('woman','女人'),('boy','男孩'),('girl','女孩'),
    ('name','名字'),('language','语言'),('world','世界'),
    ('people','人民'),('life','生命'),('death','死亡'),
    ('king','国王'),('queen','女王'),('friend','朋友'),
    ('enemy','敌人'),('computer','电脑'),('machine','机器'),
    ('music','音乐'),('art','艺术'),('science','科学'),
    ('history','历史'),('story','故事'),('paper','纸'),
    ('wall','墙'),('garden','花园'),('market','市场'),
    ('bank','银行'),('student','学生'),('teacher','老师'),
    ('doctor','医生'),('police','警察'),('army','军队'),
    ('president','总统'),('government','政府'),('law','法律'),
    ('power','权力'),('system','系统'),('problem','问题'),
    ('answer','答案'),('result','结果'),('game','游戏'),
    ('team','团队'),('news','新闻'),('company','公司'),
    ('business','商业'),('product','产品'),('price','价格'),
    ('air','空气'),('weapon','武器'),('ship','船'),
    ('train','火车'),('plane','飞机'),('bridge','桥'),
    ('pain','痛苦'),('joy','快乐'),('hope','希望'),
    ('freedom','自由'),('justice','正义'),('respect','尊重'),
    ('trust','信任'),('duty','责任'),('effort','努力'),
    ('skill','技能'),('knowledge','知识'),('memory','回忆'),
    ('habit','习惯'),('rule','规则'),('goal','目标'),
    ('purpose','目的'),('reason','原因'),('voice','声音'),
    ('weight','重量'),('speed','速度'),('force','力'),
    ('distance','距离'),('direction','方向'),('position','位置'),
    ('level','水平'),('area','地区'),('land','土地'),
    ('method','方法'),('process','过程'),('standard','标准'),
    ('quality','质量'),('source','来源'),('resource','资源'),
    ('material','材料'),('tool','工具'),('medicine','药物'),
    ('disease','疾病'),('crime','犯罪'),
    ('up','上'),('down','下'),('left','左'),('right','右'),
    ('north','北'),('south','南'),('east','东'),('west','西'),
    ('in','里'),('out','外'),('front','前'),('back','后'),
    ('inside','内'),('top','顶'),('middle','中'),('side','边'),
    ('yes','是'),('no','不'),('why','为什么'),('how','如何'),
    ('what','什么'),('who','谁'),('here','这里'),('there','那里'),
    ('now','现在'),('again','再'),('always','总是'),('never','从不'),
    ('because','因为'),('but','但是'),('or','或'),('and','和'),('if','如果'),
    ('very','很'),('more','更多'),('less','更少'),
    ('only','只有'),('also','也'),('still','仍然'),('already','已经'),
    ('after','之后'),('before','之前'),('about','关于'),('with','与'),
    ('from','从'),('for','为了'),('over','超过'),('under','下'),
]

# ─── BLEU ───
def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values()); mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bpv = [1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    bp = min(2.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

# ═══════════════════════════════════════════════════════════
# PHASE 1: Extract sentence-context hidden states for each anchor
# ═══════════════════════════════════════════════════════════
print(f"Cloud Anchor Codebook | K={K}")
print("=" * 60)

ckpt = torch.load(f'{SAVE_DIR}/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); L1.load_state_dict(ckpt['L1'])
L0.eval(); L1.eval()
print(f"Loaded anchor_auto.pt  repair_BLEU={ckpt.get('repair_bleu','?'):.1f}")

# Build valid anchors + BPE-ID → anchor_index map
valid_anchors = []
for en_word, zh_word in ANCHOR_WORDS:
    en_ids = sp.encode_as_ids(en_word)
    zh_ids = sp.encode_as_ids(zh_word)
    if ok(en_ids) and ok(zh_ids):
        valid_anchors.append((en_word, zh_word, en_ids, zh_ids))

print(f"Valid anchors: {len(valid_anchors)}")

# Token-ID → anchor index map for fast scanning
token_to_anchor = defaultdict(list)  # token_id → [(anchor_idx, 'en'|'zh', first_token_id)]
for ai, (en_word, zh_word, en_ids, zh_ids) in enumerate(valid_anchors):
    token_to_anchor[en_ids[0]].append((ai, 'en'))
    token_to_anchor[zh_ids[0]].append((ai, 'zh'))

tokens_covered = len(token_to_anchor)
print(f"  BPE tokens covered: {tokens_covered}")

# Load data
print("\nloading sentences...")
train_en = []
with open("/data/datasets/wmt14/wmt14.train.de-en") as f:
    for i, l in enumerate(f):
        if i >= 50000: break
        if "\t" in l: train_en.append(sp.encode_as_ids(l.split("\t",1)[1].strip().lower()))
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh, en = l.strip().split("\t", 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 2 and len(en_t) >= 2: pairs.append((zh_t, en_t))

auto_en = [ids[:MAX_LEN] for ids in train_en[:80000] if len(ids) >= 2]
auto_zh = [zh[:MAX_LEN] for zh, _ in pairs[:80000] if len(zh) >= 2]
print(f"EN sents={len(auto_en)}  ZH sents={len(auto_zh)}")

# Pre-allocate cloud buffers: [n_anchors, K, d]
N = len(valid_anchors)
cloud_en = torch.zeros(N, K, d, device=device)
cloud_en_count = torch.zeros(N, dtype=torch.int, device=device)
cloud_zh = torch.zeros(N, K, d, device=device)
cloud_zh_count = torch.zeros(N, dtype=torch.int, device=device)

def pad_to_heap(ids, T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    nl = 1 << (k - 1)
    return torch.tensor(ids + [0] * (nl - T), device=device), nl

# ─── Scan EN sentences ───
print("\n=== Scanning EN sentences for anchor contexts ===")
t_start = time.time()
seen_en = 0
for si, ids in enumerate(auto_en):
    if si % 5000 == 0 and si > 0:
        t_elapsed = time.time() - t_start
        print(f"  EN: {si}/{len(auto_en)}  hits={seen_en}  {t_elapsed:.0f}s")
    # Check which anchors appear in this sentence
    appeared = set()
    for t, tok in enumerate(ids[:MAX_LEN]):
        if tok not in token_to_anchor: continue
        for ai, lang in token_to_anchor[tok]:
            if lang != 'en': continue
            if ai in appeared: continue  # only first occurrence per anchor per sentence
            appeared.add(ai)
            if cloud_en_count[ai] >= K: continue

            T = min(len(ids), MAX_LEN)
            ids_pad, _ = pad_to_heap(ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad).unsqueeze(0)
                he = L1.fe(emb).squeeze(0)  # [T, d]
                h_t = he[t]
            k = cloud_en_count[ai].item()
            cloud_en[ai, k] = h_t
            cloud_en_count[ai] += 1
            seen_en += 1
    # Early stop if all filled
    if (cloud_en_count >= K).all(): break

print(f"  EN done: {cloud_en_count.sum().item()} samples collected")

# ─── Scan ZH sentences ───
print("\n=== Scanning ZH sentences for anchor contexts ===")
t_start = time.time()
seen_zh = 0
for si, ids in enumerate(auto_zh):
    if si % 5000 == 0 and si > 0:
        t_elapsed = time.time() - t_start
        print(f"  ZH: {si}/{len(auto_zh)}  hits={seen_zh}  {t_elapsed:.0f}s")
    appeared = set()
    for t, tok in enumerate(ids[:MAX_LEN]):
        if tok not in token_to_anchor: continue
        for ai, lang in token_to_anchor[tok]:
            if lang != 'zh': continue
            if ai in appeared: continue
            appeared.add(ai)
            if cloud_zh_count[ai] >= K: continue

            T = min(len(ids), MAX_LEN)
            ids_pad, _ = pad_to_heap(ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad).unsqueeze(0)
                he = L1.fe(emb).squeeze(0)
                h_t = he[t]
            k = cloud_zh_count[ai].item()
            cloud_zh[ai, k] = h_t
            cloud_zh_count[ai] += 1
            seen_zh += 1
    if (cloud_zh_count >= K).all(): break

print(f"  ZH done: {cloud_zh_count.sum().item()} samples collected")

# Report coverage
en_covered = (cloud_en_count > 0).sum().item()
zh_covered = (cloud_zh_count > 0).sum().item()
en_full = (cloud_en_count >= K).sum().item()
zh_full = (cloud_zh_count >= K).sum().item()
print(f"\n  Coverage: EN {en_covered}/{N} (K-full: {en_full})  ZH {zh_covered}/{N} (K-full: {zh_full})")

# Fill Missing: for anchors with < K samples, repeat existing or use isolated encoding
for ai in range(N):
    for cloud, count in [(cloud_en, cloud_en_count), (cloud_zh, cloud_zh_count)]:
        if count[ai] == 0:
            # Fallback: isolated word encoding
            en_ids, zh_ids = valid_anchors[ai][2], valid_anchors[ai][3]
            ids = en_ids if cloud is cloud_en else zh_ids
            with torch.no_grad():
                e = L0(torch.tensor(ids, device=device)).unsqueeze(0)
                h = L1.fe(e).squeeze(0).mean(dim=0)
            for k in range(K):
                cloud[ai, k] = h
            count[ai] = K
        elif count[ai] < K:
            # Repeat existing samples
            existing = count[ai].item()
            for k in range(existing, K):
                cloud[ai, k] = cloud[ai, k % existing]
            count[ai] = K

print(f"  All anchors filled to K={K}")

# L2 normalize for cosine similarity
cloud_en_flat = cloud_en.reshape(-1, d)  # [N*K, d]
cloud_zh_flat = cloud_zh.reshape(-1, d)
cloud_en_flat = F.normalize(cloud_en_flat, dim=-1)
cloud_zh_flat = F.normalize(cloud_zh_flat, dim=-1)

# ═══════════════════════════════════════════════════════════
# PHASE 2: Anchor self-test — EN anchor cloud → nearest ZH anchor cloud
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: Anchor self-test (cloud NN)")
print("=" * 60)

# For each anchor, take each of K EN samples, find nearest ZH cloud
# Nearest cloud = argmin_{j} min_{k} ||en_sample - zh_cloud[j][k]||

# Pre-compute: for each ZH anchor, its K samples as [N, K, d]
zh_cloud_norm = F.normalize(cloud_zh, dim=-1)  # [N, K, d]

# Test: for each EN anchor, sample one EN hidden, find nearest ZH cloud
# Use leave-one-out: test on samples not in cloud (or just use all for diagnostic)
correct = 0; total = 0
per_anchor_correct = [0] * N  # number of correct out of 5 per anchor
with torch.no_grad():
    for ai in range(N):
        en_samples = F.normalize(cloud_en[ai], dim=-1)  # [K, d]
        for k in range(min(K, 5)):  # test 5 samples per anchor
            h = en_samples[k:k+1]  # [1, d]
            c = torch.mm(h, cloud_zh_flat.T).squeeze(0)
            c_anchors = c.reshape(N, K).max(dim=1).values
            best_j = c_anchors.argmax().item()
            ok = (best_j == ai)
            correct += ok; total += 1
            if ok:
                per_anchor_correct[ai] += 1

cloud_self_acc = 100 * correct / total
print(f"\n  Cloud NN self-test: {correct}/{total} = {cloud_self_acc:.1f}%")

# Show per-anchor results
correct_pa = sum(1 for c in per_anchor_correct if c > 0)
print(f"  Per-anchor (any sample correct): {correct_pa}/{N} = {100*correct_pa/N:.1f}%")

# List anchors where < 5 test samples correct
print(f"\n  Partial/failed anchors (<5/5 correct):")
shown = 0
for ai in range(N):
    n_ok = per_anchor_correct[ai]
    if n_ok < 5:
        en_w, zh_w = valid_anchors[ai][0], valid_anchors[ai][1]
        print(f"  {en_w:12s} → {zh_w:8s}  ok={n_ok}/5")
        shown += 1
        if shown >= 40:
            print(f"  ... and {N - shown - correct_pa} more")
            break

# ═══════════════════════════════════════════════════════════
# PHASE 3: Q-enhanced cloud NN test
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 3: Q-enhanced cloud NN test")
print("=" * 60)

bridge_path = f'{SAVE_DIR}/anchor_bridge.pt'
if os.path.exists(bridge_path):
    br = torch.load(bridge_path, map_location=device)
    Q = br['Q'].to(device)
    print(f"  Q loaded, Procrustes cos={br.get('procrustes_cos','?')}")

    # Apply Q to EN clouds: EN_hidden @ Q → aligned to ZH space
    cloud_en_q = cloud_en @ Q  # [N, K, d]
    cloud_en_q = F.normalize(cloud_en_q, dim=-1)

    correct_q = 0; total_q = 0
    with torch.no_grad():
        for ai in range(N):
            en_samples = cloud_en_q[ai]  # [K, d] already normalized
            for k in range(min(K, 5)):
                h = en_samples[k:k+1]
                c = torch.mm(h, cloud_zh_flat.T).squeeze(0)
                c_anchors = c.reshape(N, K).max(dim=1).values
                best_j = c_anchors.argmax().item()
                correct_q += (best_j == ai); total_q += 1

    q_self_acc = 100 * correct_q / total_q
    print(f"  Q-enhanced cloud self-test: {correct_q}/{total_q} = {q_self_acc:.1f}%")

    # Detailed comparison
    print(f"\n  {'Anchor':20s} {'raw':6s} {'+Q':6s} {'Δ':8s}")
    for ai in range(N):
        en_samples_raw = F.normalize(cloud_en[ai][:1], dim=-1)
        en_samples_q = cloud_en_q[ai][:1]
        c_raw = torch.mm(en_samples_raw, cloud_zh_flat.T).squeeze(0).reshape(N, K).max(dim=1).values
        c_q = torch.mm(en_samples_q, cloud_zh_flat.T).squeeze(0).reshape(N, K).max(dim=1).values
        best_raw = c_raw.argmax().item()
        best_q = c_q.argmax().item()
        ok_raw = '✓' if best_raw == ai else '✗'
        ok_q = '✓' if best_q == ai else '✗'
        cos_best = c_q[ai].item()
        if ok_raw != ok_q or cos_best < 0.85:
            en_w, zh_w = valid_anchors[ai][0], valid_anchors[ai][1]
            print(f"  {en_w+'>'+zh_w:20s} {ok_raw:6s} {ok_q:6s} cos={cos_best:.3f}")

else:
    print(f"  No bridge checkpoint at {bridge_path}, skipping Q test")

# ═══════════════════════════════════════════════════════════
# PHASE 4: Sentence-level NN translation test
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 4: Sentence-level cloud NN translation")
print("=" * 60)

bridge_pairs = pairs[-50000:]
val_sents = random.sample(bridge_pairs, min(10, len(bridge_pairs)))

# Use Q-enhanced EN cloud if available, else raw
en_cloud_for_nn = cloud_en_q.reshape(N, K, d) if os.path.exists(bridge_path) else cloud_en.reshape(N, K, d)
en_cloud_for_nn = F.normalize(en_cloud_for_nn, dim=-1)
zh_tokens_for_nn = torch.tensor([valid_anchors[ai][3][0] for ai in range(N)], device=device)  # first BPE tok

correct_tok = 0; total_tok = 0
rf, hp = [], []
for zh_ids, en_ids in val_sents:
    T = min(len(en_ids), min(len(zh_ids), MAX_LEN))
    if T < 3: continue
    en_ids, zh_ids = en_ids[:T], zh_ids[:T]
    ids_pad_en, _ = pad_to_heap(en_ids, T)
    with torch.no_grad():
        emb = L0(ids_pad_en).unsqueeze(0)
        he = L1.fe(emb).squeeze(0)[:T]  # [T, d]
        if os.path.exists(bridge_path): he = he @ Q
        he = F.normalize(he, dim=-1)  # [T, d]

        # For each position, find nearest cloud
        # he [T, d] × cloud_flat.T [d, N*K] → [T, N*K]
        c = torch.mm(he, cloud_zh_flat.T)  # [T, N*K]
        c_anchors = c.reshape(T, N, K).max(dim=2).values  # [T, N]
        best_j = c_anchors.argmax(dim=1)  # [T]
        pred_ids = zh_tokens_for_nn[best_j].cpu().tolist()

        for t in range(T):
            correct_tok += (pred_ids[t] == zh_ids[t])
            total_tok += 1

        print(f"  EN: {sp.decode_ids(en_ids)[:70]}")
        print(f"  ZH: {sp.decode_ids(zh_ids)[:70]}")
        print(f"  NN: {sp.decode_ids(pred_ids)[:70]}")
        print()

        rf.append(zh_ids); hp.append(pred_ids)

sent_bleu = compute_bleu(rf, hp)
sent_acc = 100 * correct_tok / max(1, total_tok)
print(f"Sentence token acc: {correct_tok}/{total_tok} = {sent_acc:.1f}%")
print(f"Sentence BLEU: {sent_bleu:.1f}")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print("=" * 60)
print(f"  Cloud K:             {K}")
print(f"  Anchors:             {N}")
print(f"  Cloud self-acc:      {cloud_self_acc:.1f}%")
if os.path.exists(bridge_path):
    print(f"  Q cloud self-acc:    {q_self_acc:.1f}%  (Δ: {q_self_acc - cloud_self_acc:+.1f}%)")
print(f"  Sentence token acc:  {sent_acc:.1f}%")
print(f"  Sentence BLEU:       {sent_bleu:.1f}")

# Save cloud data for future use
torch.save({
    'cloud_en': cloud_en.cpu(), 'cloud_zh': cloud_zh.cpu(),
    'cloud_en_count': cloud_en_count.cpu(), 'cloud_zh_count': cloud_zh_count.cpu(),
    'valid_anchors': valid_anchors, 'N': N, 'K': K,
    'cloud_self_acc': cloud_self_acc,
}, f'{SAVE_DIR}/anchor_cloud_{K}.pt')
print(f"\nSaved: {SAVE_DIR}/anchor_cloud_{K}.pt")
