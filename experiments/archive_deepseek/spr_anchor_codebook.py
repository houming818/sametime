"""Anchor Codebook: nearest-neighbor translation via anchor hidden states."""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, numpy as np, random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d, MAX_LEN = sp.get_piece_size(), 128, 50

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)

    def fe(self, x):
        return self.ep(self.enc(x)[0])

    def fd(self, x):
        return self.dp(self.dec(x)[0])

L0 = nn.Embedding(V, d).to(device)
L1 = BiGRU(d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt['L0'])
L1.load_state_dict(ckpt['L1'])
L0.eval()
L1.eval()

def ok(ids):
    return all(x != 0 for x in ids)

# Build anchor codebook: EN hidden → ZH hidden → ZH token
ANCHOR_WORDS = [
    ('one', '一'), ('two', '二'), ('three', '三'), ('four', '四'), ('five', '五'),
    ('six', '六'), ('seven', '七'), ('eight', '八'), ('nine', '九'), ('ten', '十'),
    ('hundred', '百'), ('thousand', '千'),
    ('i', '我'), ('you', '你'), ('he', '他'), ('she', '她'), ('it', '它'),
    ('we', '我们'), ('they', '他们'), ('me', '我'), ('him', '他'),
    ('big', '大'), ('small', '小'), ('good', '好'), ('bad', '坏'),
    ('new', '新'), ('old', '老'), ('hot', '热'), ('cold', '冷'),
    ('long', '长'), ('short', '短'), ('high', '高'), ('low', '低'),
    ('fast', '快'), ('slow', '慢'), ('rich', '富'), ('poor', '穷'),
    ('strong', '强'), ('weak', '弱'), ('true', '真'), ('happy', '幸福'),
    ('young', '年轻'), ('heavy', '重'), ('light', '轻'),
    ('hard', '硬'), ('soft', '软'), ('deep', '深'), ('wide', '宽'),
    ('clean', '干净'), ('safe', '安全'), ('easy', '容易'),
    ('important', '重要'), ('free', '自由'), ('bright', '明亮'),
    ('dark', '黑暗'), ('dry', '干'), ('full', '满'), ('empty', '空'),
    ('same', '相同'), ('different', '不同'), ('real', '真实'),
    ('correct', '正确'), ('wrong', '错误'), ('useful', '有用'),
    ('great', '伟大'), ('serious', '严重'), ('major', '主要'),
    ('unique', '独特'), ('special', '特殊'), ('basic', '基本'),
    ('popular', '流行'), ('famous', '著名'),
    ('water', '水'), ('fire', '火'), ('sun', '太阳'), ('moon', '月'),
    ('mountain', '山'), ('river', '河'), ('sky', '天'), ('earth', '地'),
    ('wind', '风'), ('rain', '雨'), ('snow', '雪'), ('sea', '海'),
    ('tree', '树'), ('flower', '花'), ('wood', '木'), ('stone', '石'),
    ('gold', '金'), ('iron', '铁'), ('star', '星'), ('cloud', '云'),
    ('lake', '湖'), ('ice', '冰'), ('sand', '沙'),
    ('fish', '鱼'), ('bird', '鸟'), ('horse', '马'), ('cow', '牛'),
    ('chicken', '鸡'), ('dragon', '龙'), ('bear', '熊'), ('wolf', '狼'),
    ('red', '红'), ('green', '绿'), ('white', '白'), ('black', '黑'),
    ('yellow', '黄'),
    ('mother', '母亲'), ('father', '父亲'), ('son', '儿子'),
    ('daughter', '女儿'), ('husband', '丈夫'), ('wife', '妻子'),
    ('family', '家'), ('child', '孩子'), ('brother', '兄弟'),
    ('parent', '父母'), ('baby', '婴儿'),
    ('food', '食物'), ('rice', '米'), ('meat', '肉'), ('bread', '面包'),
    ('milk', '牛奶'), ('egg', '蛋'), ('fruit', '水果'), ('sugar', '糖'),
    ('salt', '盐'), ('oil', '油'), ('wine', '酒'), ('tea', '茶'),
    ('coffee', '咖啡'), ('fish', '鱼'), ('apple', '苹果'),
    ('day', '天'), ('night', '夜'), ('year', '年'), ('month', '月'),
    ('week', '周'), ('hour', '小时'), ('time', '时间'),
    ('today', '今天'), ('tomorrow', '明天'), ('morning', '早上'),
    ('evening', '晚上'), ('spring', '春'), ('summer', '夏'),
    ('autumn', '秋'), ('winter', '冬'),
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
    ('fight', '战斗'), ('win', '赢'), ('send', '送'),
    ('remember', '记住'), ('forget', '忘记'), ('believe', '相信'),
    ('understand', '理解'), ('change', '改变'), ('grow', '生长'),
    ('build', '建设'), ('cut', '切'), ('break', '打破'),
    ('push', '推'), ('pull', '拉'), ('carry', '带'), ('throw', '扔'),
    ('catch', '抓'), ('follow', '跟随'), ('lead', '领导'),
    ('meet', '见面'), ('wait', '等'), ('hope', '希望'), ('try', '尝试'),
    ('need', '需要'), ('keep', '保持'), ('agree', '同意'),
    ('return', '返回'), ('leave', '离开'), ('enter', '进入'),
    ('choose', '选择'), ('decide', '决定'), ('prepare', '准备'),
    ('finish', '完成'), ('join', '加入'), ('share', '分享'),
    ('protect', '保护'), ('attack', '攻击'), ('destroy', '破坏'),
    ('create', '创造'), ('produce', '生产'), ('develop', '发展'),
    ('improve', '改善'), ('manage', '管理'), ('save', '保存'),
    ('happen', '发生'), ('appear', '出现'), ('continue', '继续'),
    ('reduce', '减少'), ('increase', '增加'),
    ('book', '书'), ('door', '门'), ('window', '窗'), ('house', '家'),
    ('car', '车'), ('road', '路'), ('city', '城市'), ('country', '国家'),
    ('school', '学校'), ('hospital', '医院'), ('table', '桌'),
    ('chair', '椅'), ('bed', '床'), ('phone', '电话'),
    ('money', '钱'), ('war', '战争'), ('peace', '和平'),
    ('man', '男人'), ('woman', '女人'), ('boy', '男孩'), ('girl', '女孩'),
    ('name', '名字'), ('language', '语言'), ('world', '世界'),
    ('people', '人民'), ('life', '生命'), ('death', '死亡'),
    ('king', '国王'), ('queen', '女王'), ('friend', '朋友'),
    ('enemy', '敌人'), ('computer', '电脑'), ('machine', '机器'),
    ('music', '音乐'), ('art', '艺术'), ('science', '科学'),
    ('history', '历史'), ('story', '故事'), ('paper', '纸'),
    ('wall', '墙'), ('garden', '花园'), ('market', '市场'),
    ('bank', '银行'), ('student', '学生'), ('teacher', '老师'),
    ('doctor', '医生'), ('police', '警察'), ('army', '军队'),
    ('president', '总统'), ('government', '政府'), ('law', '法律'),
    ('power', '权力'), ('system', '系统'), ('problem', '问题'),
    ('answer', '答案'), ('result', '结果'), ('game', '游戏'),
    ('team', '团队'), ('news', '新闻'), ('company', '公司'),
    ('business', '商业'), ('product', '产品'), ('price', '价格'),
    ('air', '空气'), ('weapon', '武器'), ('ship', '船'),
    ('train', '火车'), ('plane', '飞机'), ('bridge', '桥'),
    ('hospital', '医院'), ('pain', '痛苦'), ('joy', '快乐'),
    ('hope', '希望'), ('freedom', '自由'), ('justice', '正义'),
    ('respect', '尊重'), ('trust', '信任'), ('duty', '责任'),
    ('effort', '努力'), ('skill', '技能'), ('knowledge', '知识'),
    ('memory', '回忆'), ('habit', '习惯'), ('rule', '规则'),
    ('goal', '目标'), ('purpose', '目的'), ('reason', '原因'),
    ('voice', '声音'), ('weight', '重量'), ('speed', '速度'),
    ('temperature', '温度'), ('pressure', '压力'), ('force', '力'),
    ('distance', '距离'), ('direction', '方向'), ('position', '位置'),
    ('level', '水平'), ('area', '地区'), ('land', '土地'),
    ('method', '方法'), ('process', '过程'), ('standard', '标准'),
    ('quality', '质量'), ('source', '来源'), ('resource', '资源'),
    ('material', '材料'), ('tool', '工具'), ('medicine', '药物'),
    ('disease', '疾病'), ('crime', '犯罪'),
    ('up', '上'), ('down', '下'), ('left', '左'), ('right', '右'),
    ('north', '北'), ('south', '南'), ('east', '东'), ('west', '西'),
    ('in', '里'), ('out', '外'), ('front', '前'), ('back', '后'),
    ('inside', '内'), ('top', '顶'), ('middle', '中'), ('side', '边'),
    ('yes', '是'), ('no', '不'), ('why', '为什么'), ('how', '如何'),
    ('what', '什么'), ('who', '谁'), ('here', '这里'), ('there', '那里'),
    ('now', '现在'), ('again', '再'), ('always', '总是'), ('never', '从不'),
    ('because', '因为'), ('but', '但是'), ('or', '或'), ('and', '和'), ('if', '如果'),
    ('very', '很'), ('more', '更多'), ('less', '更少'),
    ('only', '只有'), ('also', '也'), ('still', '仍然'), ('already', '已经'),
    ('after', '之后'), ('before', '之前'), ('about', '关于'), ('with', '与'),
    ('from', '从'), ('for', '为了'), ('over', '超过'), ('under', '下'),
]

# Build EN anchor codebook: en_hidden → nearest zh_hidden → zh_token
print("Building anchor codebook...")
with torch.no_grad():
    # Collect all anchor hidden states: EN hidden, ZH hidden, ZH token
    anchor_en_h, anchor_zh_h, anchor_zh_tok = [], [], []
    for en_word, zh_word in ANCHOR_WORDS:
        en_ids = sp.encode_as_ids(en_word)
        zh_ids = sp.encode_as_ids(zh_word)
        if not ok(en_ids) or not ok(zh_ids):
            continue
        e_en = L0(torch.tensor(en_ids, device=device)).unsqueeze(0)
        e_zh = L0(torch.tensor(zh_ids, device=device)).unsqueeze(0)
        h_en = L1.fe(e_en).squeeze(0).mean(dim=0)  # [d]
        h_zh = L1.fe(e_zh).squeeze(0).mean(dim=0)  # [d]
        anchor_en_h.append(h_en)
        anchor_zh_h.append(h_zh)
        anchor_zh_tok.append(zh_ids[0])  # first BPE token of ZH word

    anchor_en_h = torch.stack(anchor_en_h)  # [K, d]
    anchor_zh_h = torch.stack(anchor_zh_h)  # [K, d]
    print(f"  Anchors: {anchor_en_h.shape[0]} pairs")

    # Test: for each EN anchor, find nearest ZH anchor by cosine similarity
    print("\n=== Anchor self-test: EN anchor → nearest ZH anchor ===")
    correct = 0
    for i in range(min(50, len(anchor_en_h))):
        h_en_i = anchor_en_h[i]
        # Find nearest ZH anchor hidden
        cos_sim = F.cosine_similarity(h_en_i.unsqueeze(0), anchor_zh_h, dim=-1)
        best_j = cos_sim.argmax().item()
        best_zh_tok = sp.decode_ids([anchor_zh_tok[best_j]])
        gold_zh_tok = sp.decode_ids([anchor_zh_tok[i]])
        match = (best_j == i)
        correct += match
        if i < 20:
            en_word = ANCHOR_WORDS[i][0]
            print(f"  {en_word:10s} → nn={best_zh_tok:10s} (gold={gold_zh_tok:10s}) {'✓' if match else '✗'}  cos={cos_sim[best_j]:.3f}")
    print(f"  Anchor self-accuracy: {correct}/{min(50,len(anchor_en_h))} = {100*correct/min(50,len(anchor_en_h)):.1f}%")

    # Now test on sentence-level tokens
    print("\n=== Sentence-level NN translation ===")
    # Load some parallel data
    pairs = []
    with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
        for l in f:
            if "\t" in l:
                zh, en = l.strip().split("\t", 1)
                zh_toks = sp.encode_as_ids(zh.strip())
                en_toks = sp.encode_as_ids(en.strip().lower())
                if len(zh_toks) >= 2 and len(en_toks) >= 2:
                    pairs.append((zh_toks, en_toks))

    def pad_to_heap(ids, T):
        k = 1
        while (1 << (k - 1)) < T:
            k += 1
        nl = 1 << (k - 1)
        return torch.tensor(ids + [0] * (nl - T), device=device), nl

    # Test 10 sentence pairs
    correct_tok, total_tok = 0, 0
    for zh_ids, en_ids in random.sample(pairs[-5000:], 10):
        T = min(len(en_ids), min(len(zh_ids), MAX_LEN))
        if T < 3: continue
        en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
        ids_pad_en, _ = pad_to_heap(en_ids, T)
        with torch.no_grad():
            emb = L0(ids_pad_en).unsqueeze(0)
            he = L1.fe(emb).squeeze(0)[:T]  # [T, d]
            # For each position, find nearest ZH anchor
            pred_ids = []
            for t in range(T):
                h_t = he[t]  # [d]
                cos_sim = F.cosine_similarity(h_t.unsqueeze(0), anchor_zh_h, dim=-1)
                best_j = cos_sim.argmax().item()
                pred_ids.append(anchor_zh_tok[best_j])
                if anchor_zh_tok[best_j] == zh_ids[t]:
                    correct_tok += 1
                total_tok += 1
            print(f"  EN: {sp.decode_ids(en_ids)[:70]}")
            print(f"  ZH: {sp.decode_ids(zh_ids)[:70]}")
            print(f"  NN: {sp.decode_ids(pred_ids)[:70]}")
            print()

    print(f"Sentence token acc: {correct_tok}/{total_tok} = {100*correct_tok/max(1,total_tok):.1f}%")
