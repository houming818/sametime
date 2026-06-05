"""
SPR Anchor Bridge — 1000 锚点词注入 + Procrustes 对齐
L0/L1 anchor MSE 强制 EN-ZH 词向量对齐 → SVD 解桥 → 冻 decoder 翻译

Usage:
  python3 spr_anchor_bridge.py --phase auto --epochs 100
  python3 spr_anchor_bridge.py --phase bridge
  python3 spr_anchor_bridge.py --phase eval   (加载已有 ckpt 诊断)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sentencepiece as spm, sys, os
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {}; i = 1
while i < len(sys.argv):
    if sys.argv[i].startswith('--'):
        k = sys.argv[i][2:]; v = sys.argv[i+1] if i+1 < len(sys.argv) else ''; args[k] = v; i += 2
    else: i += 1

phase = args.get('phase', 'auto')
d, MAX_LEN = 128, 50
save_dir = '/mnt/nas/datasets/wmt17/checkpoints'; os.makedirs(save_dir, exist_ok=True)

sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
print(f"device={device} SPR Anchor Bridge | phase={phase} vocab={V}")
print("=" * 60)

# ═══════════════════════════════════════════════
# Anchor Dictionary: 1000 EN-ZH word pairs
# ═══════════════════════════════════════════════
def ok(ids): return all(x != 0 for x in ids)

ANCHOR_WORDS = [
    # ── pronouns ──
    ('i','我'),('you','你'),('he','他'),('she','她'),('it','它'),
    ('we','我们'),('they','他们'),('me','我'),('him','他'),
    ('my','我'),('your','你'),('his','他'),('her','她'),
    # ── numbers ──
    ('one','一'),('two','二'),('three','三'),('four','四'),('five','五'),
    ('six','六'),('seven','七'),('eight','八'),('nine','九'),('ten','十'),
    ('hundred','百'),('thousand','千'),('first','第一'),('second','第二'),
    # ── body ──
    ('hand','手'),('eye','眼'),('head','头'),('heart','心'),
    ('blood','血'),('body','身体'),('mouth','口'),('ear','耳'),
    ('foot','脚'),('arm','手臂'),('leg','腿'),('hair','头发'),
    ('brain','脑'),('tongue','舌'),('neck','颈'),
    # ── nature ──
    ('water','水'),('fire','火'),('sun','太阳'),('moon','月'),
    ('mountain','山'),('river','河'),('sky','天'),('earth','地'),
    ('wind','风'),('rain','雨'),('snow','雪'),('sea','海'),
    ('tree','树'),('flower','花'),('wood','木'),('stone','石'),
    ('gold','金'),('iron','铁'),('star','星'),('cloud','云'),
    ('lake','湖'),('ice','冰'),('sand','沙'),
    # ── animals ──
    ('fish','鱼'),('bird','鸟'),('horse','马'),('cow','牛'),
    ('chicken','鸡'),('dragon','龙'),
    ('bear','熊'),('elephant','象'),('wolf','狼'),
    # ── colors ──
    ('red','红'),('green','绿'),('white','白'),('black','黑'),
    ('yellow','黄'),
    # ── family ──
    ('mother','母亲'),('father','父亲'),('son','儿子'),('daughter','女儿'),
    ('husband','丈夫'),('wife','妻子'),('family','家'),('child','孩子'),
    ('brother','兄弟'),('parent','父母'),('baby','婴儿'),
    # ── food ──
    ('food','食物'),('rice','米'),('meat','肉'),('bread','面包'),
    ('milk','牛奶'),('egg','蛋'),('fruit','水果'),('sugar','糖'),
    ('salt','盐'),('oil','油'),('wine','酒'),('tea','茶'),
    ('coffee','咖啡'),('fish','鱼'),('apple','苹果'),
    # ── time ──
    ('day','天'),('night','夜'),('year','年'),('month','月'),
    ('week','周'),('hour','小时'),('time','时间'),('today','今天'),
    ('tomorrow','明天'),('morning','早上'),('evening','晚上'),
    ('spring','春'),('summer','夏'),('autumn','秋'),('winter','冬'),
    # ── verbs ──
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
    ('fight','战斗'),('win','赢'),('send','送'),('receive','收'),
    ('remember','记住'),('forget','忘记'),('believe','相信'),
    ('understand','理解'),('explain','解释'),('change','改变'),
    ('grow','生长'),('build','建设'),('cut','切'),('break','打破'),
    ('push','推'),('pull','拉'),('carry','带'),('throw','扔'),
    ('catch','抓'),('follow','跟随'),('lead','领导'),('meet','见面'),
    ('wait','等'),('hope','希望'),('try','尝试'),('allow','允许'),
    ('need','需要'),('keep','保持'),('agree','同意'),('promise','承诺'),
    ('refuse','拒绝'),('accept','接受'),('return','返回'),('leave','离开'),
    ('enter','进入'),('arrive','到达'),('stay','停留'),('move','移动'),
    ('pass','通过'),('rise','上升'),('fall','下降'),('drop','掉'),
    ('lift','举起'),('burn','烧'),('melt','融化'),
    ('choose','选择'),('decide','决定'),('prepare','准备'),('finish','完成'),
    ('join','加入'),('share','分享'),('divide','分'),('separate','分开'),
    ('enjoy','享受'),('suffer','受苦'),('worry','担心'),('fear','恐惧'),
    ('protect','保护'),('attack','攻击'),('defend','防御'),('destroy','破坏'),
    ('create','创造'),('produce','生产'),('develop','发展'),('improve','改善'),
    ('manage','管理'),('organize','组织'),('save','保存'),('remain','保持'),
    ('disappear','消失'),('survive','幸存'),('own','拥有'),('exist','存在'),
    ('happen','发生'),('appear','出现'),('continue','继续'),('reduce','减少'),
    ('increase','增加'),('raise','提高'),('avoid','避免'),('expect','期望'),
    # ── adjectives ──
    ('big','大'),('small','小'),('good','好'),('bad','坏'),
    ('new','新'),('old','老'),('hot','热'),('cold','冷'),
    ('long','长'),('short','短'),('high','高'),('low','低'),
    ('fast','快'),('slow','慢'),('beautiful','美'),('rich','富'),
    ('strong','强'),('weak','弱'),('true','真'),('happy','幸福'),
    ('sad','悲伤'),('young','年轻'),('heavy','重'),('light','轻'),
    ('hard','硬'),('soft','软'),('deep','深'),('wide','宽'),
    ('clean','干净'),('safe','安全'),('easy','容易'),('important','重要'),
    ('free','自由'),('fair','公平'),('bright','明亮'),('dark','黑暗'),
    ('dry','干'),('full','满'),('empty','空'),('nice','好'),
    ('kind','善良'),('brave','勇敢'),('afraid','害怕'),('angry','愤怒'),
    ('tired','累'),('sick','病'),('healthy','健康'),('dead','死'),
    ('same','相同'),('different','不同'),('similar','相似'),
    ('entire','整个'),('real','真实'),('correct','正确'),
    ('wrong','错误'),('useful','有用'),('great','伟大'),
    ('wonderful','美好'),('excellent','优秀'),('perfect','完美'),
    ('serious','严重'),('major','主要'),('final','最后'),
    ('original','原始'),('unique','独特'),('special','特殊'),
    ('specific','具体'),('general','一般'),('basic','基本'),
    ('essential','必要'),('critical','关键'),('popular','流行'),
    ('famous','著名'),('successful','成功'),('effective','有效'),
    ('powerful','强大'),('meaningful','有意义'),('positive','积极'),
    ('negative','消极'),('direct','直接'),('formal','正式'),
    ('legal','合法'),('moral','道德'),('political','政治'),
    ('social','社会'),('cultural','文化'),('historical','历史'),
    ('scientific','科学'),('technical','技术'),
    ('possible','可能'),('necessary','必要'),('certain','确定'),
    # ── common nouns ──
    ('book','书'),('door','门'),('window','窗'),('house','家'),
    ('car','车'),('road','路'),('city','城市'),('country','国家'),
    ('school','学校'),('hospital','医院'),('table','桌'),('chair','椅'),
    ('bed','床'),('phone','电话'),('money','钱'),('war','战争'),
    ('peace','和平'),('man','男人'),('woman','女人'),('boy','男孩'),
    ('girl','女孩'),('name','名字'),('word','字'),('language','语言'),
    ('world','世界'),('people','人民'),('life','生命'),('death','死亡'),
    ('god','神'),('king','王'),('queen','女王'),('friend','朋友'),
    ('enemy','敌人'),('computer','电脑'),('machine','机器'),
    ('music','音乐'),('art','艺术'),('science','科学'),('history','历史'),
    ('story','故事'),('letter','信'),('paper','纸'),('glass','玻璃'),
    ('wall','墙'),('floor','地板'),('garden','花园'),('park','公园'),
    ('market','市场'),('bank','银行'),('church','教堂'),
    ('office','办公室'),('student','学生'),('teacher','老师'),
    ('doctor','医生'),('police','警察'),('army','军队'),
    ('president','总统'),('government','政府'),('law','法律'),
    ('power','权力'),('system','系统'),('problem','问题'),
    ('answer','答案'),('question','问题'),('result','结果'),
    ('game','游戏'),('team','团队'),('news','新闻'),
    ('company','公司'),('business','商业'),('product','产品'),
    ('price','价格'),('value','价值'),('air','空气'),
    ('weapon','武器'),('gun','枪'),('ship','船'),
    ('train','火车'),('plane','飞机'),('bridge','桥'),
    ('pain','痛苦'),('joy','快乐'),('anger','愤怒'),
    ('hope','希望'),('faith','信念'),('courage','勇气'),
    ('freedom','自由'),('justice','正义'),('honor','荣誉'),
    ('respect','尊重'),('trust','信任'),('duty','责任'),
    ('effort','努力'),('skill','技能'),('knowledge','知识'),
    ('memory','回忆'),('habit','习惯'),('rule','规则'),
    ('goal','目标'),('purpose','目的'),('reason','原因'),('effect','效果'),
    ('voice','声音'),('silence','安静'),('weight','重量'),('speed','速度'),
    ('temperature','温度'),('pressure','压力'),('force','力'),
    ('distance','距离'),('direction','方向'),('position','位置'),('level','水平'),
    ('area','地区'),('land','土地'),('method','方法'),('process','过程'),
    ('stage','阶段'),('standard','标准'),('quality','质量'),
    ('source','来源'),('resource','资源'),('material','材料'),('tool','工具'),
    ('medicine','药物'),('disease','疾病'),('crime','犯罪'),('punishment','惩罚'),
    # ── roles ──
    ('leader','领导'),('manager','经理'),('worker','工人'),('farmer','农民'),
    ('soldier','士兵'),('writer','作家'),('artist','艺术家'),('scientist','科学家'),
    ('engineer','工程师'),('driver','司机'),('nurse','护士'),
    ('lawyer','律师'),('judge','法官'),('king','国王'),
    # ── places ──
    ('room','房间'),('kitchen','厨房'),('corner','角落'),
    ('entrance','入口'),('exit','出口'),('island','岛'),
    ('valley','山谷'),('desert','沙漠'),('factory','工厂'),
    ('station','车站'),('airport','机场'),('port','港口'),
    ('library','图书馆'),('museum','博物馆'),('theater','剧院'),
    ('hotel','酒店'),('store','商店'),
    # ── abstract ──
    ('thought','思想'),('feeling','感觉'),('emotion','情感'),('spirit','精神'),
    ('mind','心智'),('nature','自然'),('environment','环境'),
    ('weather','天气'),('climate','气候'),('design','设计'),
    ('style','风格'),('beauty','美'),('truth','真理'),
    ('wisdom','智慧'),('luck','运气'),('fate','命运'),
    ('danger','危险'),('progress','进步'),('revolution','革命'),('reform','改革'),
    # ── activities ──
    ('travel','旅行'),('visit','访问'),('discover','发现'),
    ('invent','发明'),('paint','画'),('wash','洗'),('clean','清洁'),
    ('repair','修理'),('celebrate','庆祝'),('welcome','欢迎'),('invite','邀请'),
    ('thank','感谢'),('apologize','道歉'),('forgive','原谅'),
    # ── more adj/adv ──
    ('strange','奇怪'),('normal','正常'),('common','常见'),
    ('simple','简单'),('complex','复杂'),('clear','清楚'),('obvious','明显'),
    ('secret','秘密'),('private','私人'),('public','公共'),('local','本地'),
    ('national','国家'),('international','国际'),('foreign','外国'),
    ('natural','自然'),('modern','现代'),
    ('western','西方'),('eastern','东方'),
    # ── directions ──
    ('up','上'),('down','下'),('left','左'),('right','右'),
    ('north','北'),('south','南'),('east','东'),('west','西'),
    ('in','里'),('out','外'),('front','前'),('back','后'),
    ('inside','内'),('top','顶'),('middle','中'),('center','中心'),
    ('side','边'),
    # ── misc ──
    ('yes','是'),('no','不'),('why','为什么'),('how','如何'),
    ('what','什么'),('who','谁'),('here','这里'),('there','那里'),
    ('now','现在'),('again','再'),('always','总是'),('never','从不'),
    ('because','因为'),('but','但是'),('or','或'),('and','和'),('if','如果'),
    ('very','很'),('more','更多'),('less','更少'),
    ('only','只有'),('also','也'),('still','仍然'),('already','已经'),
    ('between','之间'),('without','没有'),('through','通过'),
    ('after','之后'),('before','之前'),('about','关于'),('against','反对'),
    ('with','与'),('from','从'),('for','为了'),('into','进入'),
    ('over','超过'),('under','下'),('near','附近'),
    ('part','部分'),('place','地方'),('point','点'),
    ('case','情况'),('fact','事实'),('example','例子'),
    ('kind','种类'),('group','组'),('number','数字'),
    # ── more nouns ──
    ('poem','诗'),('song','歌'),('film','电影'),('picture','图片'),
    ('image','图像'),('video','视频'),('signal','信号'),('message','消息'),
    ('information','信息'),('data','数据'),('evidence','证据'),
    ('document','文件'),('record','记录'),('contract','合同'),
    ('agreement','协议'),('plan','计划'),('strategy','战略'),
    ('policy','政策'),('program','程序'),('project','项目'),
    ('sport','体育'),('competition','竞赛'),('victory','胜利'),
    ('threat','威胁'),('risk','风险'),('crisis','危机'),('conflict','冲突'),
    ('violence','暴力'),('vision','愿景'),('mission','使命'),
    ('role','角色'),('status','地位'),('title','标题'),
    ('task','任务'),('job','工作'),('career','职业'),
    ('patient','病人'),('client','客户'),('customer','顾客'),('audience','观众'),
    ('population','人口'),('generation','代'),('community','社区'),('society','社会'),
    ('union','联盟'),('organization','组织'),
    # ── more verbs ──
    ('express','表达'),('describe','描述'),('mention','提到'),('suggest','建议'),
    ('propose','提议'),('demand','要求'),('request','请求'),('claim','声称'),
    ('declare','宣布'),('announce','宣布'),('release','释放'),('deliver','交付'),
    ('supply','供应'),('provide','提供'),('offer','提供'),('present','呈现'),
    ('represent','代表'),('contain','包含'),('include','包括'),('involve','涉及'),
    ('require','需要'),('consider','考虑'),('assume','假设'),('imagine','想象'),
    ('realize','意识'),('recognize','认识'),('identify','识别'),
    ('compare','比较'),('prefer','偏好'),('apply','应用'),('adopt','采用'),
    ('pay','支付'),('cost','花费'),('spend','花费'),('earn','赚'),
    ('invest','投资'),('save','节省'),('waste','浪费'),('collect','收集'),
    # ── academic ──
    ('theory','理论'),('practice','实践'),('concept','概念'),('principle','原则'),
    ('philosophy','哲学'),('physics','物理学'),('chemistry','化学'),
    ('mathematics','数学'),('literature','文学'),('poetry','诗歌'),
    ('analysis','分析'),('evaluation','评价'),
    # ── Pos-aligned from parallel corpus (position-approximate, 267 clean pairs) ──
    ('但是','But'),('如果','If'),('中国','China'),('事实上','Indeed'),
    ('此外','Moreover'),('我们','We'),('他们','They'),('当然','Of'),
    ('因此','So'),('这些','These'),('即便','Even'),('比如','For'),
    ('这一','This'),('毕竟','After'),('但这','But'),('即使','Even'),
    ('随着','As'),('首先','First'),('但在','But'),('最后','Finally'),
    ('结果','As'),('与此同时','Meanwhile'),('许多','Many'),('为了','To'),
    ('但如果','But'),('不幸的是','Unfortunately'),('而且','And'),('它们','They'),
    ('其次','Second'),('纽约','NEW'),('相反','Instead'),('一些','Some'),
    ('正如','As'),('伦敦','LONDON'),('特朗普','Trump'),('虽然','While'),
    ('例如','For'),('这种','This'),('现在','Now'),('今天','Today'),
    ('根据','According'),('幸运的是','Fortunately'),('通过','By'),('作为','As'),
    ('那么','So'),('换句话说','In'),('世界','The'),('英国','The'),
    ('在美国','In'),('类似地','Likewise'),('只有','Only'),('这是','This'),
    ('欧盟','The'),('对于','For'),('同样','Likewise'),('但这一','But'),
    ('这意味着','This'),('也许','Perhaps'),('华盛顿','WASHINGTON'),('去年','Last'),
    ('所以','So'),('所有这些','All'),('第三','Third'),('问题在于','The'),
    ('第二','The'),('实际上','Indeed'),('尽管如此','Nonetheless'),('奥巴马','Obama'),
    ('印度','India'),('简言之','In'),('他们的','Their'),('其他','Other'),
    ('除了','Beyond'),('有些','Some'),('所有','All'),('日本','Japan'),
    ('好消息是','The'),('而在','And'),('但如今','But'),('联合国','The'),
    ('美国的','America'),('除非','Un'),('问题','The'),('确实','Indeed'),
    ('可以肯定','To'),('他的','His'),('这样的','Such'),('大部分','Most'),
    ('要想','To'),('在过去','the'),('为什么','Why'),('由此','The'),
    ('显然','Clear'),('这就是','That'),('非洲','Africa'),('或者','Or'),
    ('答案','answer'),('俄罗斯','Russia'),('的确','Indeed'),('那些','Those'),
    ('今年','This'),('但尽管','But'),('一旦','On'),('甚至','Even'),
    ('这就','This'),('但随着','But'),('但这种','But'),('有人','Some'),
    ('德国','Germany'),('我们必须','We'),('不过','But'),('不仅如此','Moreover'),
    ('只要','As'),('否则','Otherwise'),('这是一个','This'),('世界银行','The'),
    ('在这方面','Here'),('问题是','is'),('而尽管','And'),('或许','Perhaps'),
    ('西方','The'),('相比之下','By'),('发自伦敦','LONDON'),('国际','The'),
    ('而这','And'),('普京','Putin'),('某些','Some'),('在许多','In'),
    ('美国总统','US'),('在欧洲','In'),('东京','TOKYO'),('这也是','That'),
    ('韩国','South'),('欧元区','The'),('我的','My'),('原因','The'),
    ('意大利','Italy'),('当前','The'),('在这一','In'),('多年来','For'),
    ('公共','Public'),('但这并不意味着','But'),('一种','One'),('另外','Moreover'),
    ('新的','New'),('第一','The'),('其结果是','As'),('土耳其','Turkey'),
    ('但日本','But'),('有了','With'),('非洲的','Africa'),('但这个','But'),
    ('即使是','even'),('并不是说','not'),('贸易','Trade'),('按照','According'),
    ('其他国家','Other'),('世界上最','the'),('但不','But'),('这样做','Doing'),
    ('直到最近','recently'),('今年的','This'),('阿富汗','Afghanistan'),('安倍','Abe'),
    ('这些国家','These'),('公司','Companies'),('我们现在','We'),('教育','Education'),
    ('值得','It'),('该计划','The'),('西班牙','The'),('最近的','The'),
    ('截止','By'),('它们可以','they'),('农民','Farmers'),('近年来','In'),
    ('埃塞俄比亚','Ethiopia'),('巴勒斯坦','Palestinian'),('这或许','This'),('她们','They'),
    ('女性','Women'),('约翰内斯堡','JOHANNESBURG'),('尼日利亚','Nigeria'),('约翰','John'),
    ('南非','South'),('因此我','So'),('他们需要','they'),('农业','Agriculture'),
    ('发达国家','The'),('非洲国家','African'),('但非洲','But'),('建设','Building'),
]

def build_anchors(use_labse_only=False):
    """Build deduplicated anchor pairs with BPE token IDs."""
    anchors = []
    seen = set()
    if not use_labse_only:
        for en_word, zh_word in ANCHOR_WORDS:
            en_ids = sp.encode_as_ids(en_word)
            zh_ids = sp.encode_as_ids(zh_word)
            if ok(en_ids) and ok(zh_ids):
                key = (tuple(en_ids), tuple(zh_ids))
                if key not in seen:
                    seen.add(key)
                    anchors.append((en_ids, zh_ids))
    # Load LaBSE-guided cross-language anchor pairs
    try:
        from labse_anchors import LABSE_ANCHORS
        for en_word, zh_word in LABSE_ANCHORS:
            en_ids = sp.encode_as_ids(en_word)
            zh_ids = sp.encode_as_ids(zh_word)
            if ok(en_ids) and ok(zh_ids):
                key = (tuple(en_ids), tuple(zh_ids))
                if key not in seen:
                    seen.add(key)
                    anchors.append((en_ids, zh_ids))
    except ImportError:
        pass
    return anchors

ANCHORS = build_anchors(use_labse_only=True)  # train on LaBSE-guided only, eval on manual
print(f"Anchor pairs: {len(ANCHORS)}")

# ═══════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════
print("loading data...")
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh, en = l.strip().split("\t", 1)
            zh_toks = sp.encode_as_ids(zh.strip()); en_toks = sp.encode_as_ids(en.strip().lower())
            if len(zh_toks) >= 2 and len(en_toks) >= 2:
                pairs.append((zh_toks, en_toks))

train_en = []
with open("/data/datasets/wmt14/wmt14.train.de-en") as f:
    for i, l in enumerate(f):
        if i >= 50000: break
        if "\t" in l: train_en.append(sp.encode_as_ids(l.split("\t", 1)[1].strip().lower()))
val_en = [sp.encode_as_ids(l.split("\t", 1)[1].strip().lower()) for l in
          open("/data/datasets/wmt14/wmt14.validation.de-en")][:300]

n_auto_data = int(args.get("data", "100000"))
auto_en = [ids[:MAX_LEN] for ids in train_en[:n_auto_data // 2] if len(ids) >= 3]
auto_zh = [zh[:MAX_LEN] for zh, _ in pairs[:n_auto_data // 2] if len(zh) >= 2]

# Anchor TLM sentences: each anchor pair becomes a mini bilingual "sentence"
# "cat" [SEP] "猫" → model learns both languages share the same space
anchor_sents = []
for en_ids, zh_ids in ANCHORS:
    if not ok(en_ids) or not ok(zh_ids): continue
    concat = en_ids + [2] + zh_ids  # 2 = BPE </s> as language separator
    if len(concat) <= MAX_LEN:
        anchor_sents.append(concat)
repeat_tlm = 10  # reduced repeat — more anchors means less repetition needed
all_auto = auto_en + auto_zh + anchor_sents * repeat_tlm
print(f"auto={len(all_auto)} EN={len(auto_en)} ZH={len(auto_zh)} TLM_anchors={len(anchor_sents)}×{repeat_tlm}")

# ═══════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════
def heap_size(T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    return 1 << (k - 1), k

def pad_to_heap(ids, T):
    nl, _ = heap_size(T)
    return torch.tensor(ids + [0] * (nl - T), device=device), nl

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
nn.init.normal_(L0.weight, 0, 0.02)
L1 = BiGRU(d).to(device)

# Per-token world embedding (for heap/heap_alt phases)
world_emb = nn.Embedding(V, d).to(device)
nn.init.normal_(world_emb.weight, 0, 0.5)
world_merge = nn.Linear(d, d).to(device)
nn.init.eye_(world_merge.weight); nn.init.zeros_(world_merge.bias)

# Tree world decomposition: token ID → binary path → shared nodes → world vector
# Depth 5: root(1) + L1(2) + L2(4) + L3(8) + L4(16) = 31 shared nodes total
TREE_DEPTH = 5
tree_nodes = nn.ModuleList([nn.Embedding(2 ** i, d) for i in range(TREE_DEPTH)])
for tn in tree_nodes:
    nn.init.normal_(tn.weight, 0, 0.1)
tree_merge = nn.Linear(d, d).to(device)
nn.init.eye_(tree_merge.weight); nn.init.zeros_(tree_merge.bias)

print(f"params={sum(p.numel() for m in [L0, L1, world_merge, tree_merge] + list(tree_nodes) for p in m.parameters()) / 1e6:.1f}M (L0+L1+tree)")

def tree_world_vec(token_ids):
    """Token ID → binary path → sum of shared node vectors → world embedding [K, 128]."""
    w = torch.zeros(len(token_ids), d, device=device)
    for level in range(TREE_DEPTH):
        stride = V // (2 ** level)
        node_idx = torch.clamp(token_ids // stride, 0, (2 ** level) - 1)
        w = w + tree_nodes[level](node_idx)
    return w


def tree_world_pos(token_ids):
    """Token emb × tree world emb → world position [K, 128]."""
    t = F.normalize(L0.weight[token_ids], dim=-1)
    w = F.normalize(tree_world_vec(token_ids), dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    left = tL * wL - tR * wR
    right = tL * wR + tR * wL
    return tree_merge(torch.cat([left, right], dim=-1))

# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════
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


def anchor_loss_l0(subset=None):
    """MSE between L0 embeddings of anchor EN-ZH pairs."""
    if len(ANCHORS) == 0: return torch.tensor(0.0, device=device)
    anchors = ANCHORS if subset is None else random.sample(ANCHORS, min(subset, len(ANCHORS)))
    loss = torch.tensor(0.0, device=device)
    for en_ids, zh_ids in anchors:
        e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
        e_zh = L0.weight[torch.tensor(zh_ids, device=device)].mean(dim=0)
        loss = loss + F.mse_loss(e_en, e_zh)
    return loss / len(anchors)


def anchor_loss_l1(subset=None):
    """MSE between L1 hidden states of anchor EN-ZH pairs."""
    if len(ANCHORS) == 0: return torch.tensor(0.0, device=device)
    anchors = ANCHORS if subset is None else random.sample(ANCHORS, min(subset, len(ANCHORS)))
    loss = torch.tensor(0.0, device=device)
    for en_ids, zh_ids in anchors:
        e_en = L0(torch.tensor(en_ids, device=device)).unsqueeze(0)
        e_zh = L0(torch.tensor(zh_ids, device=device)).unsqueeze(0)
        h_en = L1.fe(e_en).squeeze(0).mean(dim=0)
        h_zh = L1.fe(e_zh).squeeze(0).mean(dim=0)
        loss = loss + F.mse_loss(h_en, h_zh)
    return loss / len(anchors)


def eval_repair_bleu():
    """Repair BLEU: mask 25% EN tokens → autoencode → recover."""
    L0.eval(); L1.eval()
    rf, hp = [], []
    for s in val_en[:50]:
        ids = s[:MAX_LEN]; T = min(len(ids), MAX_LEN); ids = ids[:T]
        ids_b = ids[:]; k = 0
        for i in range(len(ids_b)):
            if random.random() > 0.25 or k < 2: k += 1
            else: ids_b[i] = 1
        with torch.no_grad():
            ids_pad, _ = pad_to_heap(ids_b, T)
            emb = L0(ids_pad).unsqueeze(0)
            ctx = L1.fe(emb)
            dec = L1.fd(ctx)
            logits = dec.squeeze(0)[:T] @ L0.weight.T
            pred = logits.argmax(dim=-1).cpu().tolist()
        rf.append(ids[:T]); hp.append(pred)
    L1.train(); L0.train()
    return compute_bleu(rf, hp)


def eval_anchor_cos():
    """Cosine similarity at L0 and L1 for anchor pairs."""
    L0.eval(); L1.eval()
    cos_l0_vals, cos_l1_vals = [], []
    with torch.no_grad():
        for en_ids, zh_ids in ANCHORS[:200]:  # sample 200
            e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
            e_zh = L0.weight[torch.tensor(zh_ids, device=device)].mean(dim=0)
            cos_l0_vals.append(F.cosine_similarity(e_en.unsqueeze(0), e_zh.unsqueeze(0)).item())

            e_en_b = L0(torch.tensor(en_ids, device=device)).unsqueeze(0)
            e_zh_b = L0(torch.tensor(zh_ids, device=device)).unsqueeze(0)
            h_en = L1.fe(e_en_b).squeeze(0).mean(dim=0)
            h_zh = L1.fe(e_zh_b).squeeze(0).mean(dim=0)
            cos_l1_vals.append(F.cosine_similarity(h_en.unsqueeze(0), h_zh.unsqueeze(0)).item())
    L1.train(); L0.train()
    return np.mean(cos_l0_vals), np.mean(cos_l1_vals)


def world_pos_fn(token_ids):
    """Token emb × per-token world emb → world position [K, d]."""
    t = F.normalize(L0.weight[token_ids], dim=-1)       # [K, 128] unit norm
    w = F.normalize(world_emb(token_ids), dim=-1)       # [K, 128] unit norm
    tL, tR = t[..., :d // 2], t[..., d // 2:]  # [K, 64]
    wL, wR = w[..., :d // 2], w[..., d // 2:]  # [K, 64]
    left  = tL * wL - tR * wR    # [K, 64]  cos=1 → magnitude ~0.1 each
    right = tL * wR + tR * wL    # [K, 64]
    return world_merge(torch.cat([left, right], dim=-1))  # [K, 128]


def eval_world_nn(n_samples=None):
    """World-space nearest-neighbor accuracy on anchor pairs.
    Uses mean embedding across all BPE tokens per anchor word."""
    if len(ANCHORS) == 0: return 0.0
    anchors = ANCHORS if n_samples is None else random.sample(ANCHORS, min(n_samples, len(ANCHORS)))
    valid = [(e, z) for e, z in anchors if ok(e) and ok(z)]
    if len(valid) == 0: return 0.0
    L0.eval(); world_merge.eval()
    with torch.no_grad():
        all_zh_pos = F.normalize(world_pos_fn(torch.arange(V, device=device)), dim=-1)  # [V, 128]
        en_mean_pos, zh_gold = [], []
        for en_ids, zh_ids in valid:
            e_emb = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)  # mean across BPE tokens
            w_emb = world_emb.weight[torch.tensor(en_ids, device=device)].mean(dim=0)  # per-token world
            # Apply Multiply with this mean embedding
            el, er = e_emb[:d//2], e_emb[d//2:]
            wl, wr = w_emb[:d//2], w_emb[d//2:]
            left  = el * wl - er * wr
            right = el * wr + er * wl
            en_pos = F.normalize(world_merge(torch.cat([left, right])).unsqueeze(0), dim=-1)
            en_mean_pos.append(en_pos)
            zh_gold.append(zh_ids[0])
        en_mean_pos = torch.cat(en_mean_pos)  # [K, 128]
        zh_gold = torch.tensor(zh_gold, device=device)
        cos = en_mean_pos @ all_zh_pos.T  # [K, V]
        pred = cos.argmax(dim=-1)
        correct = (pred == zh_gold).sum().item()
    L0.train(); world_merge.train()
    return 100 * correct / len(valid)


def eval_flat_nn(n_samples=None):
    """Flat embedding nearest-neighbor accuracy on anchor pairs."""
    if len(ANCHORS) == 0: return 0.0
    anchors = ANCHORS if n_samples is None else random.sample(ANCHORS, min(n_samples, len(ANCHORS)))
    valid = [(e, z) for e, z in anchors if ok(e) and ok(z)]
    if len(valid) == 0: return 0.0
    L0.eval()
    with torch.no_grad():
        zh_flat = F.normalize(L0.weight, dim=-1)
        en_mean_emb, zh_gold = [], []
        for en_ids, zh_ids in valid:
            en_mean_emb.append(L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0))
            zh_gold.append(zh_ids[0])
        en_mean_emb = F.normalize(torch.stack(en_mean_emb), dim=-1)
        zh_gold = torch.tensor(zh_gold, device=device)
        cos = en_mean_emb @ zh_flat.T
        pred = cos.argmax(dim=-1)
        correct = (pred == zh_gold).sum().item()
    L0.train()
    return 100 * correct / len(valid)


# ═══════════════════════════════════════════════
# PHASE A+B: Autoencode + Anchor Injection
# ═══════════════════════════════════════════════
if phase == 'auto':
    EPOCHS = int(args.get('epochs', '50'))
    lr = float(args.get('lr', '0.003'))
    anchor_every = int(args.get('anchor_every', '500'))
    opt = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"PHASE AUTO: epochs={EPOCHS} lr={lr} anchor_every={anchor_every}")
    print("=" * 60)

    for ep in range(EPOCHS):
        L0.train(); L1.train()
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            batch = all_auto[bi:bi + 16]
            if not batch: continue
            opt.zero_grad()
            bl, ns = torch.tensor(0.0, device=device), 0
            for ids in batch:
                T = min(len(ids), MAX_LEN); ids = ids[:T]
                ids_d = ids[:]; k = 0
                for i in range(len(ids_d)):
                    if random.random() > 0.25 or k < 2: k += 1
                    else: ids_d[i] = 1
                ids_pad, _ = pad_to_heap(ids_d, T)
                ids_tgt, _ = pad_to_heap(ids, T)
                with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                ctx = L1.fe(emb)
                dec = L1.fd(ctx)
                loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                bl += loss; ns += 1

            if ns == 0: continue
            base_loss = (bl / ns)

            # Periodic L0/L1 anchor losses
            if ti % anchor_every == 0:
                al0 = anchor_loss_l0()
                al1 = anchor_loss_l1(subset=128)
                total = base_loss + 0.5 * al0 + 0.5 * al1
            else:
                total = base_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            ti += 1; tl += total.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            flat_acc = eval_flat_nn(n_samples=200)
            world_acc = eval_world_nn(n_samples=200)
            elapsed = time.time() - t0
            print(f"  ep {ep:3d} loss={tl / max(ti, 1):.4f} repair_BLEU={rb:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} flat_NN={flat_acc:.1f}% world_NN={world_acc:.1f}% {elapsed:.0f}s")

            if ep % max(EPOCHS // 2, 1) == 0:
                L0.eval(); L1.eval()
                with torch.no_grad():
                    for en_str, zh_str in [('one', '一'), ('big', '大'), ('good', '好'), ('water', '水')]:
                        en_ids = sp.encode_as_ids(en_str); zh_ids = sp.encode_as_ids(zh_str)
                        if ok(en_ids) and ok(zh_ids):
                            e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
                            e_zh = L0.weight[torch.tensor(zh_ids, device=device)].mean(dim=0)
                            cos = F.cosine_similarity(e_en.unsqueeze(0), e_zh.unsqueeze(0)).item()
                            print(f"    cos({en_str},{zh_str})={cos:.4f}")
                L0.train(); L1.train()

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            'repair_bleu': eval_repair_bleu(), 'epochs': EPOCHS, 'n_anchors': len(ANCHORS)}
    torch.save(ckpt, f'{save_dir}/anchor_auto.pt')
    cos0, cos1 = eval_anchor_cos()
    flat_acc = eval_flat_nn(n_samples=200)
    world_acc = eval_world_nn(n_samples=200)
    print(f"Saved: anchor_auto.pt  cos_L0={cos0:.4f} cos_L1={cos1:.4f} flat_NN={flat_acc:.1f}% world_NN={world_acc:.1f}%")

# ═══════════════════════════════════════════════
# PHASE C: Procrustes Bridge (modes: direct | decoder | finetune)
# ═══════════════════════════════════════════════
elif phase == 'bridge':
    bridge_mode = args.get('bridge_mode', 'both')  # direct, decoder, finetune, both
    ckpt_path = f'{save_dir}/anchor_auto.pt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}. Run --phase auto first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    L0.load_state_dict(ckpt['L0'])
    L1.load_state_dict(ckpt['L1'])

    print(f"Loaded {ckpt_path} repair_BLEU={ckpt.get('repair_bleu', '?')}")
    cos0, cos1 = eval_anchor_cos()
    print(f"Pre-bridge: cos_L0={cos0:.4f} cos_L1={cos1:.4f}")

    # Procrustes SVD
    proc_mode = args.get('proc_mode', 'anchor')  # anchor | sentence | both
    print(f"\n=== Procrustes SVD alignment (mode={proc_mode}) ===")
    L0.eval(); L1.eval()
    Q = None

    if proc_mode in ('anchor', 'both'):
        with torch.no_grad():
            X_en, X_zh = [], []
            for en_ids, zh_ids in ANCHORS:
                e_en = L0(torch.tensor(en_ids, device=device)).unsqueeze(0)
                e_zh = L0(torch.tensor(zh_ids, device=device)).unsqueeze(0)
                h_en = L1.fe(e_en).squeeze(0).mean(dim=0)
                h_zh = L1.fe(e_zh).squeeze(0).mean(dim=0)
                X_en.append(h_en.cpu()); X_zh.append(h_zh.cpu())
            X_en = torch.stack(X_en).to(device)
            X_zh = torch.stack(X_zh).to(device)
            M = X_zh.T @ X_en
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            Q_anchor = Vt.T @ U.T
            pred = X_en @ Q_anchor
            mse = F.mse_loss(pred, X_zh).item()
            cos_anchor = F.cosine_similarity(pred, X_zh, dim=-1).mean().item()
            print(f"  Word-anchor Procrustes: MSE={mse:.6f} cos={cos_anchor:.4f}")
            Q = Q_anchor

    if proc_mode in ('sentence', 'both'):
        with torch.no_grad():
            X_en, X_zh = [], []
            n_sent = min(20000, len(bridge_pairs))
            for zh_ids, en_ids in bridge_pairs[:n_sent]:
                Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                T = min(Te, Tz)
                if T < 3: continue
                ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                ids_pad_zh, _ = pad_to_heap(zh_ids[:T], T)
                emb_en = L0(ids_pad_en).unsqueeze(0)
                emb_zh = L0(ids_pad_zh).unsqueeze(0)
                h_sent_en = L1.fe(emb_en).squeeze(0)[:T].mean(dim=0)  # sentence-mean
                h_sent_zh = L1.fe(emb_zh).squeeze(0)[:T].mean(dim=0)
                X_en.append(h_sent_en.cpu()); X_zh.append(h_sent_zh.cpu())
            X_en = torch.stack(X_en).to(device)
            X_zh = torch.stack(X_zh).to(device)
            M = X_zh.T @ X_en
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            Q_sent = Vt.T @ U.T
            pred = X_en @ Q_sent
            mse = F.mse_loss(pred, X_zh).item()
            cos_sent = F.cosine_similarity(pred, X_zh, dim=-1).mean().item()
            print(f"  Sentence-mean Procrustes: MSE={mse:.6f} cos={cos_sent:.4f}")
            Q = Q_sent

    if proc_mode == 'both':
        anchor_pred = X_en @ Q_anchor
        sent_pred = X_zh @ Q_sent.T  # inverse test
        print(f"  Cross-test: Q_anchor on sent data cos={F.cosine_similarity(anchor_pred[:min(100, len(X_en))], X_zh[:min(100, len(X_en))], dim=-1).mean():.4f}")
    if Q is None: Q = Q_anchor
    Q = Q.to(device)

    # ─── Diagnostic 1: Direct mapping (skip decoder) ───
    if bridge_mode in ('direct', 'both'):
        print("\n=== Direct mapping (no decoder) ===")
        for zh_ids, en_ids in bridge_pairs[-200:-190]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                logits = ha.squeeze(0)[:T] @ L0.weight.T  # direct, no decoder
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            print(f"  EN: {sp.decode_ids(en_ids[:T])[:70]}")
            print(f"  ZH: {sp.decode_ids(zh_ids[:T])[:70]}")
            print(f"  DR: {sp.decode_ids(pred_ids)[:70]}")
            print()
        rf_d, hp_d = [], []
        for zh_ids, en_ids in bridge_pairs[-500:]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                logits = ha.squeeze(0)[:T] @ L0.weight.T  # direct
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            rf_d.append(zh_ids[:T]); hp_d.append(pred_ids)
        direct_bleu = compute_bleu(rf_d, hp_d)
        print(f"Direct BLEU (Q→L0, no decoder): {direct_bleu:.1f}")

    # ─── Diagnostic 2: Frozen decoder (current path) ───
    if bridge_mode in ('decoder', 'finetune', 'both'):
        print("\n=== Decoder bridge samples ===")
        for zh_ids, en_ids in bridge_pairs[-200:-190]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                dz = L1.fd(ha)
                logits = dz.squeeze(0)[:T] @ L0.weight.T
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            print(f"  EN: {sp.decode_ids(en_ids[:T])[:70]}")
            print(f"  ZH: {sp.decode_ids(zh_ids[:T])[:70]}")
            print(f"  PR: {sp.decode_ids(pred_ids)[:70]}")
            print()
        rf, hp = [], []
        for zh_ids, en_ids in bridge_pairs[-500:]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                dz = L1.fd(ha)
                logits = dz.squeeze(0)[:T] @ L0.weight.T
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            rf.append(zh_ids[:T]); hp.append(pred_ids)
        dec_bleu = compute_bleu(rf, hp)
        print(f"Decoder BLEU (frozen): {dec_bleu:.1f}")

    # ─── Fine-tune decoder on Q-mapped hidden states ───
    if bridge_mode == 'finetune':
        print("\n=== Finetuning decoder on Q-mapped states ===")
        # Freeze all except decoder
        for p in L0.parameters(): p.requires_grad = False
        for p in L1.enc.parameters(): p.requires_grad = False
        for p in L1.ep.parameters(): p.requires_grad = False
        for p in L1.dec.parameters(): p.requires_grad = True
        for p in L1.dp.parameters(): p.requires_grad = True

        opt = torch.optim.Adam(list(L1.dec.parameters()) + list(L1.dp.parameters()), lr=0.0001)
        Q.requires_grad = False

        ft_epochs = int(args.get('ft_epochs', '10'))
        for ep in range(ft_epochs):
            L1.train(); tl, ti = 0, 0
            random.shuffle(bridge_pairs)
            for bi in range(0, 3000, 16):
                batch = bridge_pairs[bi:bi + 16]
                if not batch: continue
                opt.zero_grad()
                bl, ns = torch.tensor(0.0, device=device), 0
                for zh_ids, en_ids in batch:
                    Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                    T = min(Te, Tz)
                    if T < 2: continue
                    ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                    ids_tgt, _ = pad_to_heap(zh_ids[:T], T)
                    with torch.no_grad():
                        emb = L0(ids_pad_en).unsqueeze(0)
                        he = L1.fe(emb)
                        ha = he @ Q
                    dz = L1.fd(ha)
                    loss = F.cross_entropy(dz.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                    bl += loss; ns += 1
                if ns == 0: continue
                (bl / ns).backward()
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
                opt.step()
                ti += 1; tl += (bl / ns).item()
            print(f"  ft ep {ep:3d} loss={tl / max(ti, 1):.4f}")

        # Re-evaluate after finetuning
        L1.eval()
        rf_ft, hp_ft = [], []
        for zh_ids, en_ids in bridge_pairs[-500:]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                dz = L1.fd(ha)
                logits = dz.squeeze(0)[:T] @ L0.weight.T
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            rf_ft.append(zh_ids[:T]); hp_ft.append(pred_ids)
        ft_bleu = compute_bleu(rf_ft, hp_ft)
        print(f"Finetune BLEU: {ft_bleu:.1f}")

        # Show samples after finetuning
        print("\n=== Finetuned samples ===")
        for zh_ids, en_ids in bridge_pairs[-200:-193]:
            Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
            T = min(Te, Tz)
            if T < 3: continue
            ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
            with torch.no_grad():
                emb = L0(ids_pad_en).unsqueeze(0)
                he = L1.fe(emb)
                ha = he @ Q
                dz = L1.fd(ha)
                logits = dz.squeeze(0)[:T] @ L0.weight.T
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            print(f"  EN: {sp.decode_ids(en_ids[:T])[:70]}")
            print(f"  ZH: {sp.decode_ids(zh_ids[:T])[:70]}")
            print(f"  FT: {sp.decode_ids(pred_ids)[:70]}")
            print()

    # Save
    result = {'Q': Q, 'module': 'anchor_bridge', 'cos_before_l0': cos0,
              'cos_before_l1': cos1}
    if 'cos_after' in dir(): result['procrustes_cos'] = cos_after
    elif 'cos_sent' in dir(): result['procrustes_cos'] = cos_sent
    if bridge_mode in ('direct', 'both'): result['direct_bleu'] = direct_bleu
    if bridge_mode in ('decoder', 'both'): result['decoder_bleu'] = dec_bleu
    if bridge_mode == 'finetune': result['finetune_bleu'] = ft_bleu
    torch.save(result, f'{save_dir}/anchor_bridge.pt')
    print(f"Saved: {save_dir}/anchor_bridge.pt")

# ═══════════════════════════════════════════════
# PHASE D: Trainable Linear Bridge (no Procrustes)
# ═══════════════════════════════════════════════
elif phase == 'train_bridge':
    ckpt_path = f'{save_dir}/anchor_auto.pt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}. Run --phase auto first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    L0.load_state_dict(ckpt['L0'])
    L1.load_state_dict(ckpt['L1'])
    print(f"Loaded {ckpt_path} repair_BLEU={ckpt.get('repair_bleu', '?')}")

    cos0, cos1 = eval_anchor_cos()
    print(f"cos_L0={cos0:.4f}  cos_L1={cos1:.4f}")

    # Trainable bridge: Linear initialized near identity (cos_L1=0.82 means
    # identity is reasonable — encoder already maps EN+ZH close)
    Wb = nn.Linear(d, d).to(device)
    nn.init.eye_(Wb.weight)  # start from identity: cos_L1=0.82 ensures proximity
    nn.init.zeros_(Wb.bias)

    # Freeze encoder
    for p in L0.parameters(): p.requires_grad = False
    for p in L1.enc.parameters(): p.requires_grad = False
    for p in L1.ep.parameters(): p.requires_grad = False

    opt = torch.optim.Adam(list(Wb.parameters()) + list(L1.dec.parameters()) + list(L1.dp.parameters()), lr=0.001)
    tr_epochs = int(args.get('tr_epochs', '50'))
    print(f"Trainable bridge: epochs={tr_epochs} lr=0.001")

    t0 = time.time()
    for ep in range(tr_epochs):
        L1.train(); Wb.train()
        random.shuffle(bridge_pairs)
        tl, ti = 0, 0
        for bi in range(0, 3000, 16):
            batch = bridge_pairs[bi:bi + 16]
            if not batch: continue
            opt.zero_grad()
            bl, ns = torch.tensor(0.0, device=device), 0
            for zh_ids, en_ids in batch:
                Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                T = min(Te, Tz)
                if T < 2: continue
                ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                ids_tgt, _ = pad_to_heap(zh_ids[:T], T)
                with torch.no_grad():
                    emb = L0(ids_pad_en).unsqueeze(0)
                    he = L1.fe(emb)
                ha = Wb(he)  # trainable bridge
                dz = L1.fd(ha)
                logits = F.normalize(dz.squeeze(0)[:T], dim=-1) @ F.normalize(L0.weight, dim=-1).T * 10
                loss = F.cross_entropy(logits, ids_tgt[:T])
                # Entropy penalty to prevent collapse
                log_p = F.log_softmax(logits, dim=-1)
                entropy = -(torch.exp(log_p) * log_p).sum(-1).mean()
                loss = loss + 2.0 * F.relu(0.5 - entropy)
                bl += loss; ns += 1
            if ns == 0: continue
            (bl / ns).backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            ti += 1; tl += (bl / ns).item()

        if ep % max(tr_epochs // 10, 1) == 0 or ep == tr_epochs - 1:
            L1.eval(); Wb.eval()
            rf, hp = [], []
            with torch.no_grad():
                for zh_ids, en_ids in bridge_pairs[-500:]:
                    Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                    T = min(Te, Tz)
                    if T < 3: continue
                    ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                    emb = L0(ids_pad_en).unsqueeze(0)
                    he = L1.fe(emb)
                    ha = Wb(he)
                    dz = L1.fd(ha)
                    logits = F.normalize(dz.squeeze(0)[:T], dim=-1) @ F.normalize(L0.weight, dim=-1).T * 10
                    pred_ids = logits.argmax(dim=-1).cpu().tolist()
                    rf.append(zh_ids[:T]); hp.append(pred_ids)
            b = compute_bleu(rf, hp)
            elapsed = time.time() - t0
            print(f"  ep {ep:3d} loss={tl / max(ti, 1):.4f} BLEU={b:.1f} {elapsed:.0f}s")
            # Show samples
            if ep % max(tr_epochs // 5, 1) == 0:
                for zh_ids, en_ids in bridge_pairs[-203:-200]:
                    Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                    T = min(Te, Tz)
                    if T < 3: continue
                    ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                    with torch.no_grad():
                        emb = L0(ids_pad_en).unsqueeze(0)
                        he = L1.fe(emb)
                        ha = Wb(he)
                        dz = L1.fd(ha)
                        logits = F.normalize(dz.squeeze(0)[:T], dim=-1) @ F.normalize(L0.weight, dim=-1).T * 10
                        pred = sp.decode_ids(logits.argmax(dim=-1).cpu().tolist())
                    print(f"    EN: {sp.decode_ids(en_ids[:T])[:60]}")
                    print(f"    ZH: {sp.decode_ids(zh_ids[:T])[:60]}")
                    print(f"    PR: {pred[:60]}")
                    print()
            L1.train(); Wb.train()

    torch.save({'Wb': Wb.state_dict(), 'decoder': L1.dec.state_dict(), 'dp': L1.dp.state_dict(),
                'module': 'anchor_train_bridge'}, f'{save_dir}/anchor_train_bridge.pt')
    print(f"Saved: {save_dir}/anchor_train_bridge.pt")

# ═══════════════════════════════════════════════
# PHASE EVAL: Load checkpoint + diagnostics
# ═══════════════════════════════════════════════
elif phase == 'eval':
    auto_ckpt = f'{save_dir}/anchor_auto.pt'
    bridge_ckpt = f'{save_dir}/anchor_bridge.pt'

    if os.path.exists(auto_ckpt):
        ckpt = torch.load(auto_ckpt, map_location=device)
        L0.load_state_dict(ckpt['L0'])
        L1.load_state_dict(ckpt['L1'])
        print(f"Loaded {auto_ckpt}")
        print(f"  repair_BLEU={ckpt.get('repair_bleu', '?')} epochs={ckpt.get('epochs', '?')}")

        cos0, cos1 = eval_anchor_cos()
        print(f"  cos_L0={cos0:.4f}  cos_L1={cos1:.4f}")

        # Show anchor pair proximities
        print("\nAnchor pair L0 cos:")
        L0.eval(); L1.eval()
        test_words = [
            ('one', '一'), ('two', '二'), ('three', '三'),
            ('big', '大'), ('small', '小'), ('good', '好'), ('bad', '坏'),
            ('water', '水'), ('fire', '火'), ('sun', '太阳'), ('earth', '地'),
            ('go', '去'), ('come', '来'), ('eat', '吃'), ('see', '看'),
            ('man', '男人'), ('woman', '女人'), ('child', '孩子'),
            ('i', '我'), ('you', '你'), ('he', '他'),
        ]
        with torch.no_grad():
            for en_str, zh_str in test_words:
                en_ids = sp.encode_as_ids(en_str)
                zh_ids = sp.encode_as_ids(zh_str)
                if ok(en_ids) and ok(zh_ids):
                    e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
                    e_zh = L0.weight[torch.tensor(zh_ids, device=device)].mean(dim=0)
                    c = F.cosine_similarity(e_en.unsqueeze(0), e_zh.unsqueeze(0)).item()
                    print(f"  {en_str:10s} ↔ {zh_str:6s}  cos={c:+.4f}")

        # Repair BLEU
        rb = eval_repair_bleu()
        print(f"\n  Repair BLEU: {rb:.1f}")

    if os.path.exists(bridge_ckpt):
        br = torch.load(bridge_ckpt, map_location=device)
        print(f"\nLoaded {bridge_ckpt}")
        print(f"  bridge_BLEU={br.get('bridge_bleu', '?')}")
        print(f"  cos_before_L0={br.get('cos_before_l0', '?')}")
        print(f"  cos_before_L1={br.get('cos_before_l1', '?')}")
        print(f"  procrustes_cos={br.get('procrustes_cos', '?')}")

# ═══════════════════════════════════════════════
# PHASE E: Direct Linear translation (no decoder)
# ═══════════════════════════════════════════════
elif phase == 'new_dec' or phase == 'direct_lin':
    ckpt_path = f'{save_dir}/anchor_auto.pt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    L0.load_state_dict(ckpt['L0'])
    L1.load_state_dict(ckpt['L1'])
    print(f"Loaded {ckpt_path}")

    # Freeze L0 + encoder
    for p in L0.parameters(): p.requires_grad = False
    for p in L1.enc.parameters(): p.requires_grad = False
    for p in L1.ep.parameters(): p.requires_grad = False

    # Direct: Linear(d→V) — no Wb, no decoder, no L0.weight.T
    LinOut = nn.Linear(d, V, bias=False).to(device)
    nn.init.normal_(LinOut.weight, 0, 0.01)

    opt = torch.optim.Adam(LinOut.parameters(), lr=0.001)
    tr_epochs = int(args.get('tr_epochs', '50'))
    print(f"Direct Linear(d→V): epochs={tr_epochs}  (no decoder, no Wb)")

    t0 = time.time()
    for ep in range(tr_epochs):
        LinOut.train()
        random.shuffle(bridge_pairs)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            batch = bridge_pairs[bi:bi + 16]
            if not batch: continue
            opt.zero_grad()
            bl, ns = torch.tensor(0.0, device=device), 0
            for zh_ids, en_ids in batch:
                Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                T = min(Te, Tz)
                if T < 2: continue
                ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                ids_tgt, _ = pad_to_heap(zh_ids[:T], T)
                with torch.no_grad():
                    emb = L0(ids_pad_en).unsqueeze(0)
                    he = L1.fe(emb)  # frozen encoder, [1, T, 128]
                logits = LinOut(he).squeeze(0)[:T]  # [T, V]
                loss = F.cross_entropy(logits, ids_tgt[:T])
                bl += loss; ns += 1
            if ns == 0: continue
            (bl / ns).backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            ti += 1; tl += (bl / ns).item()

        if ep % max(tr_epochs // 10, 1) == 0 or ep == tr_epochs - 1:
            LinOut.eval()
            rf, hp = [], []
            with torch.no_grad():
                for zh_ids, en_ids in bridge_pairs[-500:]:
                    Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                    T = min(Te, Tz)
                    if T < 3: continue
                    ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                    emb = L0(ids_pad_en).unsqueeze(0)
                    he = L1.fe(emb)
                    lo = LinOut(he).squeeze(0)[:T]
                    hp.append(lo.argmax(dim=-1).cpu().tolist())
                    rf.append(zh_ids[:T])
            b = compute_bleu(rf, hp)
            elapsed = time.time() - t0
            print(f"  ep {ep:3d} loss={tl / max(ti, 1):.4f} BLEU={b:.1f} {elapsed:.0f}s")

            if ep % max(tr_epochs // 5, 1) == 0:
                for zh_ids, en_ids in bridge_pairs[-203:-200]:
                    Te, Tz = min(len(en_ids), MAX_LEN), min(len(zh_ids), MAX_LEN)
                    T = min(Te, Tz)
                    if T < 3: continue
                    ids_pad_en, _ = pad_to_heap(en_ids[:T], T)
                    with torch.no_grad():
                        emb = L0(ids_pad_en).unsqueeze(0)
                        he = L1.fe(emb)
                        lo = LinOut(he).squeeze(0)[:T]
                        pred_ids = lo.argmax(dim=-1).cpu().tolist()
                        pred = sp.decode_ids(pred_ids)
                    print(f"    EN: {sp.decode_ids(en_ids[:T])[:60]}")
                    print(f"    ZH: {sp.decode_ids(zh_ids[:T])[:60]}")
                    print(f"    PR: {pred[:60]}")
                    print()
            LinOut.train()

    torch.save({'LinOut': LinOut.state_dict(), 'module': 'direct_lin'},
               f'{save_dir}/anchor_direct_lin.pt')
    print(f"Saved: {save_dir}/anchor_direct_lin.pt")

# ═══════════════════════════════════════════════
# PHASE F: L0 Procrustes — space-to-space mapping
# ═══════════════════════════════════════════════
elif phase == 'l0_map':
    ckpt_path = f'{save_dir}/anchor_auto.pt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    L0.load_state_dict(ckpt['L0'])
    L1.load_state_dict(ckpt['L1'])
    L0.eval(); L1.eval()
    print(f"Loaded {ckpt_path} repair_BLEU={ckpt.get('repair_bleu', '?'):.1f}")

    # ─── Build anchor embedding matrices ───
    print("\n=== L0 Procrustes: E[anchor_en] → Q → E[anchor_zh] ===")
    with torch.no_grad():
        X_en, X_zh = [], []
        valid_idx = []
        for ai, (en_ids, zh_ids) in enumerate(ANCHORS):
            if not ok(en_ids) or not ok(zh_ids): continue
            # Use first BPE token's embedding (most anchors are single-token)
            e_en = L0.weight[torch.tensor(en_ids[0:1], device=device)].mean(dim=0)
            e_zh = L0.weight[torch.tensor(zh_ids[0:1], device=device)].mean(dim=0)
            X_en.append(e_en.cpu()); X_zh.append(e_zh.cpu())
            valid_idx.append(ai)

        X_en = torch.stack(X_en).to(device)  # [K, d]
        X_zh = torch.stack(X_zh).to(device)  # [K, d]
        K = X_en.shape[0]
        print(f"  Anchor pairs: {K}")

        # Natural cos before mapping
        cos_nat = F.cosine_similarity(X_en, X_zh, dim=-1).mean().item()
        print(f"  Natural cos (before Q): {cos_nat:.4f}")

        # Procrustes SVD
        M = X_zh.T @ X_en  # [d, d]
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        Q = Vt.T @ U.T  # Q = V @ U^T, orthogonal [d, d]

        # Quality check
        pred = X_en @ Q  # [K, d]
        mse = F.mse_loss(pred, X_zh).item()
        cos_q = F.cosine_similarity(pred, X_zh, dim=-1).mean().item()
        print(f"  Procrustes: MSE={mse:.6f} cos={cos_q:.4f}  (Δ={cos_q-cos_nat:+.4f})")

    Q = Q.to(device)

    # ─── L0 NN translation test ───
    print("\n=== L0 word translation (Q @ E[en] → nearest E[zh]) ===")
    # Normalize all ZH embeddings for fast NN
    E_zh_norm = F.normalize(L0.weight, dim=-1)  # [V, d]

    def l0_translate(en_ids):
        e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
        e_mapped = F.normalize(e_en @ Q, dim=-1)
        cos_scores = torch.mv(E_zh_norm, e_mapped)  # [V]
        return cos_scores.argmax().item()

    # Test anchor self-translation
    correct = 0
    for ai in range(min(K, 50)):
        en_ids, zh_ids = ANCHORS[valid_idx[ai]]
        if not ok(en_ids) or not ok(zh_ids): continue
        pred_id = l0_translate(en_ids[0:1])
        gold_id = zh_ids[0]
        ok_flag = (pred_id == gold_id)
        correct += ok_flag
        if ai < 30:
            en_w = sp.decode_ids(en_ids)
            gold_w = sp.decode_ids(zh_ids)
            pred_w = sp.decode_ids([pred_id])
            print(f"  {en_w:12s} → {pred_w:12s} (gold={gold_w:12s}) {'✓' if ok_flag else '✗'}")

    l0_nn_acc = 100 * correct / min(K, 50)
    print(f"\n  L0 NN anchor self-acc: {correct}/{min(K,50)} = {l0_nn_acc:.1f}%")

    # Test held-out pairs (distinct set)
    print("\n=== Held-out anchor test ===")
    # Split: use first K//2 for Procrustes, rest for testing
    train_k = K * 3 // 4
    with torch.no_grad():
        X_train_en, X_train_zh = X_en[:train_k], X_zh[:train_k]
        M2 = X_train_zh.T @ X_train_en
        U2, S2, Vt2 = torch.linalg.svd(M2, full_matrices=False)
        Q2 = Vt2.T @ U2.T

    E2_norm = F.normalize(L0.weight, dim=-1)
    def l0_translate_q2(en_ids):
        e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
        e_mapped = F.normalize(e_en @ Q2, dim=-1)
        return torch.mv(E2_norm, e_mapped).argmax().item()

    test_correct = 0; test_total = 0
    for ai in range(train_k, min(K, train_k + 50)):
        en_ids, zh_ids = ANCHORS[valid_idx[ai]]
        if not ok(en_ids) or not ok(zh_ids): continue
        pred_id = l0_translate_q2(en_ids[0:1])
        gold_id = zh_ids[0]
        test_correct += (pred_id == gold_id); test_total += 1
        if test_total <= 15:
            en_w = sp.decode_ids(en_ids); gold_w = sp.decode_ids(zh_ids)
            pred_w = sp.decode_ids([pred_id])
            print(f"  {en_w:12s} → {pred_w:12s} (gold={gold_w:12s}) {'✓' if pred_id==gold_id else '✗'}")

    if test_total > 0:
        print(f"\n  Held-out acc: {test_correct}/{test_total} = {100*test_correct/test_total:.1f}%")

    # ─── Sentence-level L0 translation ───
    print("\n=== Sentence-level L0 NN translation ===")
    bridge_pairs = pairs[-50000:]
    rf, hp = [], []
    correct_tok, total_tok = 0, 0
    for zh_ids, en_ids in random.sample(bridge_pairs, 10):
        T = min(len(en_ids), min(len(zh_ids), MAX_LEN))
        if T < 3: continue
        en_ids, zh_ids = en_ids[:T], zh_ids[:T]
        pred_ids = []
        for t in range(T):
            pred_id = l0_translate([en_ids[t]])
            pred_ids.append(pred_id)
            if pred_id == zh_ids[t]: correct_tok += 1
            total_tok += 1
        rf.append(zh_ids); hp.append(pred_ids)
        print(f"  EN: {sp.decode_ids(en_ids)[:70]}")
        print(f"  ZH: {sp.decode_ids(zh_ids)[:70]}")
        print(f"  L0: {sp.decode_ids(pred_ids)[:70]}")
        print()

    bl = compute_bleu(rf, hp)
    print(f"L0 sent token acc: {correct_tok}/{total_tok} = {100*correct_tok/max(1,total_tok):.1f}%")
    print(f"L0 sent BLEU: {bl:.1f}")

    torch.save({'Q': Q, 'L0_nn_acc': l0_nn_acc, 'cos_nat': cos_nat, 'cos_q': cos_q,
                'module': 'l0_procrustes'}, f'{save_dir}/anchor_l0_map.pt')
    print(f"\nSaved: {save_dir}/anchor_l0_map.pt")

# ═══════════════════════════════════════════════
# PHASE G: Heap experiment — token heap × world heap → world space
# ═══════════════════════════════════════════════
elif phase == 'heap':
    EPOCHS = int(args.get('epochs', '50'))
    lr = float(args.get('lr', '0.003'))
    anchor_every = int(args.get('anchor_every', '500'))
    opt = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()) +
                            list(world_merge.parameters()) + list(world_emb.parameters()), lr=lr)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"PHASE HEAP: epochs={EPOCHS} lr={lr} depth=2")
    print(f"  world_emb: Embedding({V},128) world_merge: Linear(128→128)")
    print("=" * 60)

    for ep in range(EPOCHS):
        L0.train(); L1.train(); world_merge.train()
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            batch = all_auto[bi:bi + 16]
            if not batch: continue
            opt.zero_grad()
            bl, ns = torch.tensor(0.0, device=device), 0
            for ids in batch:
                T = min(len(ids), MAX_LEN); ids = ids[:T]
                ids_d = ids[:]; k = 0
                for i in range(len(ids_d)):
                    if random.random() > 0.25 or k < 2: k += 1
                    else: ids_d[i] = 1
                ids_pad, _ = pad_to_heap(ids_d, T)
                ids_tgt, _ = pad_to_heap(ids, T)
                with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                ctx = L1.fe(emb); dec = L1.fd(ctx)
                loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                bl += loss; ns += 1

            if ns == 0: continue
            base_loss = (bl / ns)

            # InfoNCE on world positions for all anchor pairs
            en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
            zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
            n_pairs = min(len(en_ids), len(zh_ids))
            if n_pairs > 1:
                en_ids = en_ids[:n_pairs]; zh_ids = zh_ids[:n_pairs]
                en_w = world_pos_fn(en_ids)   # [K, 128]
                zh_w = world_pos_fn(zh_ids)   # [K, 128]
                en_w = F.normalize(en_w, dim=-1)
                zh_w = F.normalize(zh_w, dim=-1)
                logits = en_w @ zh_w.T / 0.07  # τ=0.07 (sharper contrast)
                labels = torch.arange(n_pairs, device=device)
                nce_loss = F.cross_entropy(logits, labels)

                total = base_loss + 1.0 * nce_loss
            else:
                total = base_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            ti += 1; tl += total.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            flat_nn = eval_flat_nn(n_samples=200)
            world_nn = eval_world_nn(n_samples=200)
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0

            # Compute mean cosine: use FULL BPE token embeddings (mean), not just first token
            en_ids = [e for e, _ in ANCHORS if ok(e) and ok(_)]
            zh_ids = [z for _, z in ANCHORS if ok(_) and ok(z)]
            n = min(len(en_ids), len(zh_ids), 100)
            if n > 0:
                with torch.no_grad():
                    en_emb = torch.stack([L0.weight[torch.tensor(e, device=device)].mean(dim=0) for e in en_ids[:n]])  # [n, 128]
                    zh_emb = torch.stack([L0.weight[torch.tensor(z, device=device)].mean(dim=0) for z in zh_ids[:n]])
                    f_flat = F.cosine_similarity(F.normalize(en_emb, dim=-1), F.normalize(zh_emb, dim=-1), dim=-1).mean().item()
                    # World pos on mean embeddings — use per-token world_emb
                    en_w, zh_w = [], []
                    for e in en_ids[:n]:
                        emb = L0.weight[torch.tensor(e, device=device)].mean(dim=0)
                        w_emb = world_emb.weight[torch.tensor(e, device=device)].mean(dim=0)
                        el, er = emb[:d//2], emb[d//2:]
                        wl, wr = w_emb[:d//2], w_emb[d//2:]
                        d2 = world_merge(torch.cat([el*wl - er*wr, el*wr + er*wl]))
                        en_w.append(F.normalize(d2.unsqueeze(0), dim=-1))
                    for z in zh_ids[:n]:
                        emb = L0.weight[torch.tensor(z, device=device)].mean(dim=0)
                        w_emb = world_emb.weight[torch.tensor(z, device=device)].mean(dim=0)
                        el, er = emb[:d//2], emb[d//2:]
                        wl, wr = w_emb[:d//2], w_emb[d//2:]
                        d2 = world_merge(torch.cat([el*wl - er*wr, el*wr + er*wl]))
                        zh_w.append(F.normalize(d2.unsqueeze(0), dim=-1))
                    en_w = torch.cat(en_w); zh_w = torch.cat(zh_w)
                    f_world = F.cosine_similarity(en_w, zh_w, dim=-1).mean().item()
            else:
                f_flat, f_world = 0.0, 0.0

            print(f"  ep {ep:3d} loss={tl / max(ti, 1):.4f} repair_BLEU={rb:.1f} "
                  f"flat_NN={flat_nn:.1f}% world_NN={world_nn:.1f}% "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} flat_cos={f_flat:.3f} world_cos={f_world:.3f} {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            'world_emb': world_emb.state_dict(), 'world_merge': world_merge.state_dict(),
            'epochs': EPOCHS, 'n_anchors': len(ANCHORS)}
    torch.save(ckpt, f'{save_dir}/anchor_heap.pt')
    flat_final = eval_flat_nn(n_samples=200)
    world_final = eval_world_nn(n_samples=200)
    print(f"Saved: anchor_heap.pt  flat_NN={flat_final:.1f}% world_NN={world_final:.1f}%")

# ═══════════════════════════════════════════════
# PHASE H: Alternating training — freeze L0 ↔ freeze world
# ═══════════════════════════════════════════════
elif phase == 'heap_alt':
    EPOCHS = int(args.get('epochs', '100'))
    lr = float(args.get('lr', '0.003'))
    lr_alt = float(args.get('lr_alt', '0.003'))

    # Separate optimizers for CE (L0+L1) and anchor (world)
    opt_ce = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    opt_world = torch.optim.Adam(list(world_emb.parameters()) + list(world_merge.parameters()), lr=lr_alt)

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"PHASE HEAP_ALT: epochs={EPOCHS} lr_ce={lr} lr_world={lr_alt}")
    print("=" * 60)

    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            # Determine mode: CE batch or anchor batch (alternating)
            do_anchor = (ti % 2 == 0)  # alternate every batch

            if do_anchor:
                # ── Anchor batch: train world_emb + world_merge on InfoNCE ──
                opt_world.zero_grad()
                L0.eval(); L1.eval()
                world_emb.train(); world_merge.train()

                en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
                zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
                n_pairs = min(len(en_ids), len(zh_ids))
                if n_pairs > 1:
                    en_w = world_pos_fn(en_ids[:n_pairs])
                    zh_w = world_pos_fn(zh_ids[:n_pairs])
                    en_w = F.normalize(en_w, dim=-1); zh_w = F.normalize(zh_w, dim=-1)
                    logits = en_w @ zh_w.T / 0.07
                    loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_world.param_groups[0]['params'], 2.0)
                    opt_world.step()
                    tl += loss.item(); ti += 1
            else:
                # ── CE batch: train L0 + L1 on autoencode ──
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt_ce.zero_grad()
                    L0.train(); L1.train()
                    world_emb.eval(); world_merge.eval()

                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T)
                        ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt_ce.param_groups[0]['params'], 2.0)
                        opt_ce.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            flat_nn = eval_flat_nn(n_samples=200)
            world_nn = eval_world_nn(n_samples=200)
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0

            # Quick cos check
            en_ids = [e for e, _ in ANCHORS if ok(e) and ok(_)]
            zh_ids = [z for _, z in ANCHORS if ok(_) and ok(z)]
            n = min(len(en_ids), len(zh_ids), 50)
            if n > 0:
                with torch.no_grad():
                    en_emb = torch.stack([L0.weight[torch.tensor(e, device=device)].mean(dim=0) for e in en_ids[:n]])
                    zh_emb = torch.stack([L0.weight[torch.tensor(z, device=device)].mean(dim=0) for z in zh_ids[:n]])
                    f_flat = F.cosine_similarity(F.normalize(en_emb, dim=-1), F.normalize(zh_emb, dim=-1), dim=-1).mean().item()
            else:
                f_flat = 0.0

            print(f"  ep {ep:3d} loss={tl / max(ti, 1):.4f} repair_BLEU={rb:.1f} "
                  f"flat_NN={flat_nn:.1f}% world_NN={world_nn:.1f}% "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} flat_cos={f_flat:.3f} {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            'world_emb': world_emb.state_dict(), 'world_merge': world_merge.state_dict(),
            'epochs': EPOCHS, 'n_anchors': len(ANCHORS)}
    torch.save(ckpt, f'{save_dir}/anchor_heap_alt.pt')
    flat_final = eval_flat_nn(n_samples=200)
    world_final = eval_world_nn(n_samples=200)
    print(f"Saved: anchor_heap_alt.pt  flat_NN={flat_final:.1f}% world_NN={world_final:.1f}%")

# ═══════════════════════════════════════════════
# PHASE I: Tree world — shared node decomposition + alternating training
# ═══════════════════════════════════════════════
elif phase == 'tree_alt':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))

    # Build tree dynamically based on depth
    td = int(args.get('tree_depth', '5'))
    print(f"  Tree depth={td}  nodes=" + "+".join([str(2**i) for i in range(td)]))

    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
    for tn in t_nodes:
        nn.init.normal_(tn.weight, 0, 0.1)
    t_merge = nn.Linear(d, d).to(device)
    nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

    def tw_vec(tok_ids):
        w = torch.zeros(len(tok_ids), d, device=device)
        for level in range(td):
            stride = V // (2 ** level)
            nidx = torch.clamp(tok_ids // stride, 0, (2 ** level) - 1)
            w = w + t_nodes[level](nidx)
        return w

    def wpos_single(tok_ids):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(tw_vec(tok_ids), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    # Pre-build anchor lists (FULL BPE token IDs)
    en_full = [e for e, _ in ANCHORS if ok(e) and ok(_)]
    zh_full = [z for _, z in ANCHORS if ok(_) and ok(z)]
    N_ank = min(len(en_full), len(zh_full))
    en_full, zh_full = en_full[:N_ank], zh_full[:N_ank]
    # Pad for batch mean computation
    max_ank_len = max(len(ids) for ids in en_full + zh_full)
    en_pad = torch.zeros(N_ank, max_ank_len, dtype=torch.long, device=device)
    zh_pad = torch.zeros(N_ank, max_ank_len, dtype=torch.long, device=device)
    en_mask = torch.zeros(N_ank, max_ank_len, device=device)
    zh_mask = torch.zeros(N_ank, max_ank_len, device=device)
    for i, ids in enumerate(en_full):
        L = len(ids); en_pad[i, :L] = torch.tensor(ids, device=device); en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_full):
        L = len(ids); zh_pad[i, :L] = torch.tensor(ids, device=device); zh_mask[i, :L] = 1.0 / L

    def wp_mean_batch(padded, mask):
        N_b, M = padded.shape
        t = F.normalize((L0.weight[padded] * mask.unsqueeze(-1)).sum(1), dim=-1)
        w_flat = tw_vec(padded.reshape(-1))
        w_3d = w_flat.reshape(N_b, M, d)
        w = F.normalize((w_3d * mask.unsqueeze(-1)).sum(1), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    opt = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()) +
                             list(t_merge.parameters()) + [p for tn in t_nodes for p in tn.parameters()], lr=lr)

    t0 = time.time()
    print(f"PHASE TREE_ALT (unfrozen L0): epochs={EPOCHS} depth={td}")
    print("=" * 60)

    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            do_anchor = (ti % 2 == 0)

            if do_anchor:
                opt.zero_grad()
                L0.train(); L1.eval()
                for tn in t_nodes: tn.train()
                t_merge.train()

                en_w = wp_mean_batch(en_pad, en_mask)
                zh_w = wp_mean_batch(zh_pad, zh_mask)
                logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
                loss = F.cross_entropy(logits, torch.arange(N_ank, device=device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
                opt.step()
                tl += loss.item(); ti += 1
            else:
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt.zero_grad()
                    L0.train(); L1.train()
                    for tn in t_nodes: tn.eval()
                    t_merge.eval()
                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T); ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
                        opt.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0

            # Eval with MEAN embeddings (not just e[0])
            with torch.no_grad():
                en_w = F.normalize(wp_mean_batch(en_pad, en_mask), dim=-1)
                zh_w = F.normalize(wp_mean_batch(zh_pad, zh_mask), dim=-1)
                logits = en_w @ zh_w.T / 0.07
                tree_acc = 100 * (logits.argmax(-1) == torch.arange(N_ank, device=device)).float().mean().item()
                # Open-set
                all_zh = F.normalize(wpos_single(torch.arange(V, device=device)), dim=-1)
                cos = en_w @ all_zh.T
                zh_gold = torch.tensor([zl[0] for zl in zh_full], device=device)
                open_acc = 100 * (cos.argmax(-1) == zh_gold).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} repair_BLEU={rb:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} tree_acc={tree_acc:.1f}% open={open_acc:.1f}% {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            't_merge': t_merge.state_dict(), 'epochs': EPOCHS, 'tree_depth': td}
    torch.save(ckpt, f'{save_dir}/anchor_tree_alt.pt')
    print(f"Saved: anchor_tree_alt.pt  tree_depth={td}")

# ═══════════════════════════════════════════════
# PHASE N: Pure InfoNCE — no CE, no L1, only anchor contrastive loss
# ═══════════════════════════════════════════════
elif phase == 'tree_nce':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    td = int(args.get('tree_depth', '5'))

    # Warm start: load L0 from pre-trained autoencode checkpoint
    warm_path = args.get('warm_from', '')
    if warm_path:
        ckpt = torch.load(warm_path, map_location=device)
        L0.load_state_dict(ckpt['L0'])
        rb = ckpt.get('repair_bleu', 0)
        print(f"  Warm start L0 from: {warm_path}")

    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
    for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
    if warm_path:
        # Root = random unit vector (non-zero → Multiply has meaningful magnitude)
        rv = torch.randn(1, d, device=device)
        t_nodes[0].weight.data = rv / rv.norm()
        for i in range(1, td):
            nn.init.zeros_(t_nodes[i].weight)
    t_merge = nn.Linear(d, d).to(device)
    nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)
    if warm_path: nn.init.zeros_(t_merge.bias)  # ensure bias doesn't skew

    def tw_vec(tok_ids):
        w = torch.zeros(len(tok_ids), d, device=device)
        for level in range(td):
            nidx = torch.clamp(tok_ids // (V // (2 ** level)), 0, (2 ** level) - 1) if level > 0 else torch.zeros_like(tok_ids)
            w = w + t_nodes[level](nidx)
        return w

    def world_pos_mean(ids_list):
        embs = torch.stack([L0.weight[torch.tensor(ids, device=device)].mean(dim=0) for ids in ids_list])
        t = F.normalize(embs, dim=-1)
        w_mean = torch.stack([tw_vec(torch.tensor(ids, device=device)).mean(dim=0) for ids in ids_list])
        w = F.normalize(w_mean, dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    opt = torch.optim.Adam(list(L0.parameters()) + list(t_merge.parameters()) +
                             [p for tn in t_nodes for p in tn.parameters()], lr=lr)

    # Pre-build anchor lists + vectorized batch computation
    en_ank_full = [e for e, _ in ANCHORS if ok(e) and ok(_)]
    zh_ank_full = [z for _, z in ANCHORS if ok(_) and ok(z)]
    N = min(len(en_ank_full), len(zh_ank_full))
    en_ank_full, zh_ank_full = en_ank_full[:N], zh_ank_full[:N]

    # Build gold eval set from MANUAL anchors only (not used in training)
    en_gold = []
    zh_gold = []
    for en_word, zh_word in ANCHOR_WORDS:
        en_ids = sp.encode_as_ids(en_word); zh_ids = sp.encode_as_ids(zh_word)
        if ok(en_ids) and ok(zh_ids): en_gold.append(en_ids); zh_gold.append(zh_ids)
    Ng = min(len(en_gold), len(zh_gold))
    en_gold, zh_gold = en_gold[:Ng], zh_gold[:Ng]
    # Build gold pads
    gold_max = max(len(ids) for ids in en_gold + zh_gold)
    g_en_pad = torch.zeros(Ng, gold_max, dtype=torch.long, device=device)
    g_zh_pad = torch.zeros(Ng, gold_max, dtype=torch.long, device=device)
    g_en_mask = torch.zeros(Ng, gold_max, device=device)
    g_zh_mask = torch.zeros(Ng, gold_max, device=device)
    for i, ids in enumerate(en_gold):
        L = len(ids); g_en_pad[i, :L] = torch.tensor(ids, device=device); g_en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_gold):
        L = len(ids); g_zh_pad[i, :L] = torch.tensor(ids, device=device); g_zh_mask[i, :L] = 1.0 / L

    # Build padded tensors for fast batch mean computation
    all_ids = en_ank_full + zh_ank_full
    max_len = max(len(ids) for ids in all_ids)
    en_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    zh_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    en_mask = torch.zeros(N, max_len, device=device)
    zh_mask = torch.zeros(N, max_len, device=device)
    for i, ids in enumerate(en_ank_full):
        L = len(ids); en_pad[i, :L] = torch.tensor(ids, device=device); en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_ank_full):
        L = len(ids); zh_pad[i, :L] = torch.tensor(ids, device=device); zh_mask[i, :L] = 1.0 / L

    def wpos_single(tok_ids):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(tw_vec(tok_ids), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    def wp_mean_batch(padded, mask):
        """Vectorized world position for mean of multi-BPE-token anchors."""
        N_b, M = padded.shape
        t = F.normalize((L0.weight[padded] * mask.unsqueeze(-1)).sum(1), dim=-1)
        # tw_vec on flattened tokens, then reshape
        w_flat = tw_vec(padded.reshape(-1))  # [N*M, 128]
        w_3d = w_flat.reshape(N_b, M, d)  # [N, M, 128]
        w = F.normalize((w_3d * mask.unsqueeze(-1)).sum(1), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    print(f"\n{'='*60}")
    print(f"PHASE TREE_NCE (pure InfoNCE): epochs={EPOCHS} depth={td} anchors={N}")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        L0.train(); t_merge.train()
        for tn in t_nodes: tn.train()
        tl = 0.0
        random.shuffle(all_auto)  # unused but keeps loop structure
        for bi in range(0, 500, 1):  # 500 InfoNCE steps per epoch
            opt.zero_grad()
            en_w = wp_mean_batch(en_pad, en_mask)
            zh_w = wp_mean_batch(zh_pad, zh_mask)
            logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
            loss = F.cross_entropy(logits, torch.arange(N, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            tl += loss.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            elapsed = time.time() - t0
            cos0, cos1 = eval_anchor_cos()
            with torch.no_grad():
                en_w = F.normalize(world_pos_mean(en_ank_full), dim=-1)
                zh_w = F.normalize(world_pos_mean(zh_ank_full), dim=-1)
                logits = en_w @ zh_w.T / 0.07
                nce_acc = 100 * (logits.argmax(-1) == torch.arange(N, device=device)).float().mean().item()
                # Gold-set eval (manual anchors, apples-to-apples with LaBSE teacher)
                en_gold_w = F.normalize(wp_mean_batch(g_en_pad, g_en_mask), dim=-1)
                zh_gold_w = F.normalize(wp_mean_batch(g_zh_pad, g_zh_mask), dim=-1)
                gold_logits = en_gold_w @ zh_gold_w.T / 0.07
                gold_acc = 100 * (gold_logits.argmax(-1) == torch.arange(Ng, device=device)).float().mean().item()
                # Open-set: among all BPE tokens
                all_zh = F.normalize(wpos_single(torch.arange(V, device=device)), dim=-1)
                cos_o = en_w @ all_zh.T
                zh_gold = torch.tensor([zl[0] for zl in zh_ank_full], device=device)
                open_acc = 100 * (cos_o.argmax(-1) == zh_gold).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/500:.4f} nce_acc={nce_acc:.1f}% gold={gold_acc:.1f}% open={open_acc:.1f}% "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} {elapsed:.0f}s", flush=True)

    ckpt = {'L0': L0.state_dict(), 't_merge': t_merge.state_dict(),
            't_nodes': {i: tn.state_dict() for i, tn in enumerate(t_nodes)},
            'epochs': EPOCHS, 'tree_depth': td}
    torch.save(ckpt, f'{save_dir}/anchor_tree_nce.pt')
    print(f"Saved: anchor_tree_nce.pt")

# ═══════════════════════════════════════════════
# PHASE Q: Tree + L1 cross-lingual training
# ═══════════════════════════════════════════════
elif phase == 'tree_l1':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    td = int(args.get('tree_depth', '5'))

    # L0 + tree (same as tree_nce)
    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
    for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
    rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
    for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
    t_merge = nn.Linear(d, d).to(device)
    nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

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
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    # L1 spatial: projection from GRU context to ZH world position
    L1_proj = nn.Linear(d, d).to(device)
    nn.init.eye_(L1_proj.weight); nn.init.zeros_(L1_proj.bias)

    # Pre-build anchor sentence index (hash-map accelerated)
    print("Building anchor sentence index (hash-accelerated)...")
    anchor_sents = [[] for _ in range(len(ANCHORS))]
    # Hash: first BPE token → list of anchor indices
    ank_hash = {}
    for ai, (en_ids, zh_ids) in enumerate(ANCHORS):
        ank_hash.setdefault(en_ids[0], []).append(ai)
    for si, (zh_sent, en_sent) in enumerate(pairs[-50000:]):
        if si % 10000 == 0 and si > 0:
            filled = sum(1 for a in anchor_sents if len(a) >= 3)
            print(f"  {si}/50000  filled={filled}/{len(ANCHORS)}")
            if filled >= len(ANCHORS) * 0.8: break
        for t, tok in enumerate(en_sent[:MAX_LEN]):
            if tok not in ank_hash: continue
            for ai in ank_hash[tok]:
                if len(anchor_sents[ai]) >= 3: continue
                en_ids = ANCHORS[ai][0]
                if t + len(en_ids) <= len(en_sent) and en_sent[t:t+len(en_ids)] == en_ids:
                    anchor_sents[ai].append((en_sent[:MAX_LEN], t, ANCHORS[ai][1]))
                    break

    # Build training pads (same as tree_nce)
    en_ank_full = [e for e, _ in ANCHORS if ok(e) and ok(_)]
    zh_ank_full = [z for _, z in ANCHORS if ok(_) and ok(z)]
    N = min(len(en_ank_full), len(zh_ank_full))
    en_ank_full, zh_ank_full = en_ank_full[:N], zh_ank_full[:N]
    max_len = max(len(ids) for ids in en_ank_full + zh_ank_full)
    en_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    zh_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    en_mask = torch.zeros(N, max_len, device=device); zh_mask = torch.zeros(N, max_len, device=device)
    for i, ids in enumerate(en_ank_full):
        L = len(ids); en_pad[i, :L] = torch.tensor(ids, device=device); en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_ank_full):
        L = len(ids); zh_pad[i, :L] = torch.tensor(ids, device=device); zh_mask[i, :L] = 1.0 / L

    # Gold eval
    en_gold, zh_gold = [], []
    for en_word, zh_word in ANCHOR_WORDS:
        ei = sp.encode_as_ids(en_word); zi = sp.encode_as_ids(zh_word)
        if ok(ei) and ok(zi): en_gold.append(ei); zh_gold.append(zi)
    Ng = min(len(en_gold), len(zh_gold)); en_gold, zh_gold = en_gold[:Ng], zh_gold[:Ng]
    gmax = max(len(ids) for ids in en_gold + zh_gold)
    g_en_pad = torch.zeros(Ng, gmax, dtype=torch.long, device=device)
    g_zh_pad = torch.zeros(Ng, gmax, dtype=torch.long, device=device)
    g_en_mask = torch.zeros(Ng, gmax, device=device); g_zh_mask = torch.zeros(Ng, gmax, device=device)
    for i, ids in enumerate(en_gold): L = len(ids); g_en_pad[i, :L] = torch.tensor(ids, device=device); g_en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_gold): L = len(ids); g_zh_pad[i, :L] = torch.tensor(ids, device=device); g_zh_mask[i, :L] = 1.0 / L

    def wp_mean_batch(padded, mask):
        N_b, M = padded.shape
        t = F.normalize((L0.weight[padded] * mask.unsqueeze(-1)).sum(1), dim=-1)
        w_flat = tw_vec(padded.reshape(-1)); w_3d = w_flat.reshape(N_b, M, d)
        w = F.normalize((w_3d * mask.unsqueeze(-1)).sum(1), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    all_tree_params = list(L0.parameters()) + list(t_merge.parameters())
    for tn in t_nodes: all_tree_params += list(tn.parameters())
    all_l1_params = list(L1.enc.parameters()) + list(L1.ep.parameters()) + list(L1_proj.parameters())

    opt_tree = torch.optim.Adam(all_tree_params, lr=lr, weight_decay=1e-4)
    opt_l1 = torch.optim.Adam(all_l1_params, lr=lr, weight_decay=1e-4)

    print(f"\n{'='*60}")
    print(f"PHASE TREE_L1 (alternating): epochs={EPOCHS} depth={td} anchors={N} gold={Ng}")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        L0.train(); t_merge.train(); L1_proj.train(); L1.train()
        for tn in t_nodes: tn.train()
        tl = 0.0
        random.shuffle(all_auto)
        for bi in range(0, 500, 1):
            if bi % 2 == 0:
                # Tree batch: InfoNCE on anchors (freeze L1)
                opt_tree.zero_grad()
                L1.eval()
                en_w = wp_mean_batch(en_pad, en_mask)
                zh_w = wp_mean_batch(zh_pad, zh_mask)
                loss = F.cross_entropy(F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07, torch.arange(N, device=device))
            else:
                # L1 batch: spatial MSE on anchor sentence contexts (freeze tree)
                opt_l1.zero_grad()
                L0.eval(); t_merge.eval()
                for tn in t_nodes: tn.eval()
                L1.train(); L1_proj.train()
                loss = torch.tensor(0.0, device=device)
                n_l1 = 0
                for _ in range(16):
                    ai = random.randrange(N)
                    if len(anchor_sents[ai]) == 0: continue
                    en_sent, pos, zh_ids = random.choice(anchor_sents[ai])
                    T = min(len(en_sent), MAX_LEN)
                    ids_pad, _ = pad_to_heap(en_sent[:T], T)
                    with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                    h_ctx = L1.fe(emb).squeeze(0)
                    l1_target = L1_proj(h_ctx[pos:pos+1]).squeeze(0)
                    zh_world = F.normalize(heap_world(torch.tensor(zh_ids, device=device)[:1]), dim=-1).squeeze(0)
                    loss = loss + F.mse_loss(l1_target, zh_world)
                    n_l1 += 1
                if n_l1 == 0: continue
                loss = loss / n_l1
            loss.backward()
            if bi % 2 == 0:
                torch.nn.utils.clip_grad_norm_(opt_tree.param_groups[0]['params'], 2.0)
                opt_tree.step()
            else:
                torch.nn.utils.clip_grad_norm_(opt_l1.param_groups[0]['params'], 2.0)
                opt_l1.step()
            tl += loss.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            elapsed = time.time() - t0
            with torch.no_grad():
                en_w = F.normalize(wp_mean_batch(en_pad, en_mask), dim=-1)
                zh_w = F.normalize(wp_mean_batch(zh_pad, zh_mask), dim=-1)
                nce_acc = 100 * ((en_w @ zh_w.T / 0.07).argmax(-1) == torch.arange(N, device=device)).float().mean().item()
                en_g = F.normalize(wp_mean_batch(g_en_pad, g_en_mask), dim=-1)
                zh_g = F.normalize(wp_mean_batch(g_zh_pad, g_zh_mask), dim=-1)
                gold_acc = 100 * ((en_g @ zh_g.T / 0.07).argmax(-1) == torch.arange(Ng, device=device)).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/500:.4f} nce={nce_acc:.1f}% gold={gold_acc:.1f}% {elapsed:.0f}s", flush=True)

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(), 'L1_proj': L1_proj.state_dict(),
            't_merge': t_merge.state_dict()}
    torch.save(ckpt, f'{save_dir}/anchor_tree_l1.pt')
    print(f"Saved: anchor_tree_l1.pt")

# ═══════════════════════════════════════════════
# PHASE O: Learned routing with Gumbel-Softmax + weight decay
# ═══════════════════════════════════════════════
elif phase == 'tree_learn':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    td = int(args.get('tree_depth', '5'))
    tau_start = float(args.get('tau_start', '1.0'))
    tau_end = float(args.get('tau_end', '0.1'))

    # Shared tree nodes
    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
    for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
    t_merge = nn.Linear(d, d).to(device)
    nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

    # Learned routing: compact code + shared projection gates
    route_dim = 16
    route_emb = nn.Embedding(V, route_dim).to(device)
    nn.init.normal_(route_emb.weight, 0, 0.05)
    route_proj = nn.ModuleList([nn.Linear(route_dim, 2 ** i).to(device) for i in range(td)])
    for rp in route_proj:
        nn.init.normal_(rp.weight, 0, 0.1); nn.init.zeros_(rp.bias)

    def tw_vec_learned(tok_ids, tau):
        code = route_emb(tok_ids)  # [K, route_dim]
        w = torch.zeros(len(tok_ids), d, device=device)
        for level in range(td):
            logits = route_proj[level](code)  # [K, 2**level]
            alpha = F.gumbel_softmax(logits, tau=tau, hard=True)  # one-hot straight-through
            nodes = t_nodes[level](torch.arange(2 ** level, device=device))
            w = w + alpha @ nodes
        return w

    def wp_single_lr(tok_ids, tau):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(tw_vec_learned(tok_ids, tau), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    # Build training anchors (LaBSE guided, exclude manual)
    en_ank_full = [e for e, _ in ANCHORS if ok(e) and ok(_)]
    zh_ank_full = [z for _, z in ANCHORS if ok(_) and ok(z)]
    N = min(len(en_ank_full), len(zh_ank_full))
    en_ank_full, zh_ank_full = en_ank_full[:N], zh_ank_full[:N]
    # Pad for batch mean
    max_len = max(len(ids) for ids in en_ank_full + zh_ank_full)
    en_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    zh_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    en_mask = torch.zeros(N, max_len, device=device)
    zh_mask = torch.zeros(N, max_len, device=device)
    for i, ids in enumerate(en_ank_full):
        L = len(ids); en_pad[i, :L] = torch.tensor(ids, device=device); en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_ank_full):
        L = len(ids); zh_pad[i, :L] = torch.tensor(ids, device=device); zh_mask[i, :L] = 1.0 / L

    # Gold eval: manual anchors only
    en_gold, zh_gold = [], []
    for en_word, zh_word in ANCHOR_WORDS:
        en_ids = sp.encode_as_ids(en_word); zh_ids = sp.encode_as_ids(zh_word)
        if ok(en_ids) and ok(zh_ids): en_gold.append(en_ids); zh_gold.append(zh_ids)
    Ng = min(len(en_gold), len(zh_gold))
    en_gold, zh_gold = en_gold[:Ng], zh_gold[:Ng]
    gold_max = max(len(ids) for ids in en_gold + zh_gold)
    g_en_pad = torch.zeros(Ng, gold_max, dtype=torch.long, device=device)
    g_zh_pad = torch.zeros(Ng, gold_max, dtype=torch.long, device=device)
    g_en_mask = torch.zeros(Ng, gold_max, device=device)
    g_zh_mask = torch.zeros(Ng, gold_max, device=device)
    for i, ids in enumerate(en_gold):
        L = len(ids); g_en_pad[i, :L] = torch.tensor(ids, device=device); g_en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_gold):
        L = len(ids); g_zh_pad[i, :L] = torch.tensor(ids, device=device); g_zh_mask[i, :L] = 1.0 / L

    all_params = list(L0.parameters()) + list(t_merge.parameters()) + list(route_emb.parameters())
    for tn in t_nodes: all_params += list(tn.parameters())
    for rp in route_proj: all_params += list(rp.parameters())
    opt = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-4)

    def wp_mean_lr(padded, mask, tau):
        N_b, M = padded.shape
        t = F.normalize((L0.weight[padded] * mask.unsqueeze(-1)).sum(1), dim=-1)
        flat_ids = padded.reshape(-1)
        w_flat = tw_vec_learned(flat_ids, tau)
        w_3d = w_flat.reshape(N_b, M, d)
        w = F.normalize((w_3d * mask.unsqueeze(-1)).sum(1), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    print(f"\n{'='*60}")
    print(f"PHASE TREE_LEARN: epochs={EPOCHS} depth={td} anchors={N} gold={Ng}")
    print(f"  tau: {tau_start}→{tau_end}  weight_decay=1e-4")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        L0.train(); t_merge.train(); route_emb.train()
        for tn in t_nodes: tn.train()
        for rp in route_proj: rp.train()
        tl = 0.0
        tau_now = max(tau_end, tau_start * (1 - ep / 100))
        random.shuffle(all_auto)
        for bi in range(0, 500, 1):
            opt.zero_grad()
            en_w = wp_mean_lr(en_pad, en_mask, tau_now)
            zh_w = wp_mean_lr(zh_pad, zh_mask, tau_now)
            logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
            loss = F.cross_entropy(logits, torch.arange(N, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            tl += loss.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            elapsed = time.time() - t0
            cos0, cos1 = eval_anchor_cos()
            with torch.no_grad():
                en_w = F.normalize(wp_mean_lr(en_pad, en_mask, tau_now), dim=-1)
                zh_w = F.normalize(wp_mean_lr(zh_pad, zh_mask, tau_now), dim=-1)
                logits = en_w @ zh_w.T / 0.07
                nce_acc = 100 * (logits.argmax(-1) == torch.arange(N, device=device)).float().mean().item()
                en_g = F.normalize(wp_mean_lr(g_en_pad, g_en_mask, tau_now), dim=-1)
                zh_g = F.normalize(wp_mean_lr(g_zh_pad, g_zh_mask, tau_now), dim=-1)
                gold_logits = en_g @ zh_g.T / 0.07
                gold_acc = 100 * (gold_logits.argmax(-1) == torch.arange(Ng, device=device)).float().mean().item()
                all_zh = F.normalize(wp_single_lr(torch.arange(V, device=device), tau_now), dim=-1)
                cos_o = en_g @ all_zh.T
                zh_gold_ids = torch.tensor([zl[0] for zl in zh_gold], device=device)
                open_acc = 100 * (cos_o.argmax(-1) == zh_gold_ids).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/500:.4f} tau={tau_now:.2f} nce={nce_acc:.1f}% gold={gold_acc:.1f}% open={open_acc:.1f}% "
                  f"cos_L0={cos0:.3f} {elapsed:.0f}s", flush=True)

    ckpt = {'L0': L0.state_dict(), 't_merge': t_merge.state_dict(), 'route_emb': route_emb.state_dict()}
    torch.save(ckpt, f'{save_dir}/anchor_tree_learn.pt')
    print(f"Saved: anchor_tree_learn.pt")

# ═══════════════════════════════════════════════
# PHASE J: Tree with learned routing (soft assignment per level)
# ═══════════════════════════════════════════════
elif phase == 'tree_lr':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    td = int(args.get('tree_depth', '5'))

    # Shared tree nodes
    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
    for tn in t_nodes:
        nn.init.normal_(tn.weight, 0, 0.1)
    t_merge = nn.Linear(d, d).to(device)
    nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

    # Learned routing: compact code per token → shared projection → per-level softmax
    route_dim = int(args.get('route_dim', '16'))
    route_emb = nn.Embedding(V, route_dim).to(device)
    nn.init.normal_(route_emb.weight, 0, 0.05)
    # Tie routing codes for anchor pairs: cat and 猫 share the same code
    with torch.no_grad():
        tied = 0
        for en_ids, zh_ids in ANCHORS:
            if ok(en_ids) and ok(zh_ids):
                route_emb.weight[zh_ids[0]].copy_(route_emb.weight[en_ids[0]])
                tied += 1
    print(f"  Tied {tied} anchor pair routing codes")
    route_proj = nn.ModuleList([nn.Linear(route_dim, 2 ** i).to(device) for i in range(td)])
    for rp in route_proj:
        nn.init.normal_(rp.weight, 0, 0.1); nn.init.zeros_(rp.bias)

    def tw_vec_lr(tok_ids):
        code = route_emb(tok_ids)  # [K, route_dim]
        w = torch.zeros(len(tok_ids), d, device=device)
        for level in range(td):
            n = 2 ** level
            a = F.softmax(route_proj[level](code), dim=-1)  # [K, n]
            nodes = t_nodes[level](torch.arange(n, device=device))  # [n, d]
            w = w + a @ nodes  # [K, n] @ [n, d] = [K, d]
        return w

    def tw_pos_lr(tok_ids):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(tw_vec_lr(tok_ids), dim=-1)
        tL, tR = t[..., :d // 2], t[..., d // 2:]
        wL, wR = w[..., :d // 2], w[..., d // 2:]
        return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

    opt_ce = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    all_tree_params = list(t_merge.parameters()) + list(route_emb.parameters())
    for tn in t_nodes: all_tree_params += list(tn.parameters())
    for rp in route_proj: all_tree_params += list(rp.parameters())
    opt_tree = torch.optim.Adam(all_tree_params, lr=lr)

    n_params = sum(p.numel() for p in all_tree_params)
    print(f"\n{'='*60}")
    print(f"PHASE TREE_LR: epochs={EPOCHS} depth={td} route_dim={route_dim} tree_params={n_params/1e3:.1f}K")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            do_anchor = (ti % 2 == 0)

            if do_anchor:
                opt_tree.zero_grad()
                L0.eval(); L1.eval()
                for m in [t_merge, route_emb, *t_nodes, *route_proj]: m.train()

                en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
                zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
                n_pairs = min(len(en_ids), len(zh_ids))
                if n_pairs > 1:
                    en_w = tw_pos_lr(en_ids[:n_pairs])
                    zh_w = tw_pos_lr(zh_ids[:n_pairs])
                    logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
                    loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_tree.param_groups[0]['params'], 2.0)
                    opt_tree.step()
                    tl += loss.item(); ti += 1
            else:
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt_ce.zero_grad()
                    L0.train(); L1.train()
                    for m in [t_merge, route_emb, *t_nodes, *route_proj]: m.eval()
                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T); ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt_ce.param_groups[0]['params'], 2.0)
                        opt_ce.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0

            en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
            zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
            n_pairs = min(len(en_ids), len(zh_ids))
            lr_acc = 0.0
            if n_pairs > 5:
                with torch.no_grad():
                    en_w = F.normalize(tw_pos_lr(en_ids[:n_pairs]), dim=-1)
                    zh_w = F.normalize(tw_pos_lr(zh_ids[:n_pairs]), dim=-1)
                    logits = en_w @ zh_w.T / 0.07
                    lr_acc = 100 * (logits.argmax(-1) == torch.arange(n_pairs, device=device)).float().mean().item()

            print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} repair_BLEU={rb:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} tree_acc={lr_acc:.1f}% {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            't_merge': t_merge.state_dict(), 'route_emb': route_emb.state_dict(),
            'epochs': EPOCHS, 'tree_depth': td}
    torch.save(ckpt, f'{save_dir}/anchor_tree_lr.pt')
    print(f"Saved: anchor_tree_lr.pt  tree_depth={td}")

# ═══════════════════════════════════════════════
# PHASE K: Recursive heap Multiply (true tree algebra)
# ═══════════════════════════════════════════════
elif phase == 'heap_rec':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    REC_DEPTH = int(args.get('rec_depth', '3'))

    # Merge matrices: one per internal level (depth-1 levels)
    merge_mats = nn.ModuleList()
    for i in range(REC_DEPTH - 1):
        dim_at_level = d // (2 ** i)  # 128, 64, 32, ...
        merge_mats.append(nn.Linear(dim_at_level, dim_at_level).to(device))
        nn.init.eye_(merge_mats[-1].weight); nn.init.zeros_(merge_mats[-1].bias)

    # Per-token world embedding (128D, sliced recursively like token emb)
    world_emb = nn.Embedding(V, d).to(device)
    nn.init.normal_(world_emb.weight, 0, 0.5)
    # Tie anchor pairs
    with torch.no_grad():
        tied = 0
        for en_ids, zh_ids in ANCHORS:
            if ok(en_ids) and ok(zh_ids):
                world_emb.weight[zh_ids[0]].copy_(world_emb.weight[en_ids[0]])
                tied += 1
    print(f"  Tied {tied} anchor pair world embeddings")

    def rec_multiply(t_vec, w_vec, level):
        """Recursive heap Multiply. t_vec, w_vec: [K, d_at_level]."""
        if level == REC_DEPTH:  # leaf: Hadamard
            return t_vec * w_vec
        sub = t_vec.shape[-1] // 2
        tL, tR = t_vec[..., :sub], t_vec[..., sub:]
        wL, wR = w_vec[..., :sub], w_vec[..., sub:]
        LL = rec_multiply(tL, wL, level + 1)
        RR = rec_multiply(tR, wR, level + 1)
        LR = rec_multiply(tL, wR, level + 1)
        RL = rec_multiply(tR, wL, level + 1)
        left = LL - RR
        right = LR + RL
        return merge_mats[level - 1](torch.cat([left, right], dim=-1))

    def heap_world_pos(tok_ids):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(world_emb(tok_ids), dim=-1)
        return rec_multiply(t, w, 1)  # start from level 1 (root children)

    opt_ce = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    all_wp = list(world_emb.parameters())
    for mm in merge_mats: all_wp += list(mm.parameters())
    opt_world = torch.optim.Adam(all_wp, lr=lr)

    n_params = sum(p.numel() for p in all_wp)
    print(f"\n{'='*60}")
    print(f"PHASE HEAP_REC: epochs={EPOCHS} depth={REC_DEPTH} world_params={n_params/1e3:.1f}K")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            do_anchor = (ti % 2 == 0)

            if do_anchor:
                opt_world.zero_grad()
                L0.eval(); L1.eval()
                world_emb.train()
                for mm in merge_mats: mm.train()

                en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
                zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
                n_pairs = min(len(en_ids), len(zh_ids))
                if n_pairs > 1:
                    en_w = heap_world_pos(en_ids[:n_pairs])
                    zh_w = heap_world_pos(zh_ids[:n_pairs])
                    logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
                    loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_world.param_groups[0]['params'], 2.0)
                    opt_world.step()
                    tl += loss.item(); ti += 1
            else:
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt_ce.zero_grad()
                    L0.train(); L1.train()
                    world_emb.eval()
                    for mm in merge_mats: mm.eval()
                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T); ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt_ce.param_groups[0]['params'], 2.0)
                        opt_ce.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0

            en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
            zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
            n_pairs = min(len(en_ids), len(zh_ids))
            rec_acc = 0.0
            if n_pairs > 5:
                with torch.no_grad():
                    en_w = F.normalize(heap_world_pos(en_ids[:n_pairs]), dim=-1)
                    zh_w = F.normalize(heap_world_pos(zh_ids[:n_pairs]), dim=-1)
                    logits = en_w @ zh_w.T / 0.07
                    rec_acc = 100 * (logits.argmax(-1) == torch.arange(n_pairs, device=device)).float().mean().item()

            print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} repair_BLEU={rb:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} rec_acc={rec_acc:.1f}% {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(),
            'world_emb': world_emb.state_dict(), 'epochs': EPOCHS, 'rec_depth': REC_DEPTH}
    torch.save(ckpt, f'{save_dir}/anchor_heap_rec.pt')
    print(f"Saved: anchor_heap_rec.pt  rec_depth={REC_DEPTH}")

# ═══════════════════════════════════════════════
# PHASE L: Full architecture — recursive Multiply + shared nodes + soft routing
# ═══════════════════════════════════════════════
elif phase == 'heap_full':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    FD = int(args.get('full_depth', '3'))

    merge_mats = nn.ModuleList()
    for i in range(FD - 1):
        dim_at_level = d // (2 ** i)
        merge_mats.append(nn.Linear(dim_at_level, dim_at_level).to(device))
        nn.init.eye_(merge_mats[-1].weight); nn.init.zeros_(merge_mats[-1].bias)

    br_nodes = []
    for lvl in range(1, FD):
        n_branches = 2 ** (lvl - 1)
        br_dim = d // (2 ** lvl)
        p = nn.Parameter(torch.randn(n_branches, 2, br_dim, device=device) * 0.1)
        br_nodes.append(p)

    def rec_multiply_full(t_vec, w_vec, level):
        if level == FD: return t_vec * w_vec
        sub = t_vec.shape[-1] // 2
        tL, tR = t_vec[..., :sub], t_vec[..., sub:]
        wL, wR = w_vec[..., :sub], w_vec[..., sub:]
        LL = rec_multiply_full(tL, wL, level + 1)
        RR = rec_multiply_full(tR, wR, level + 1)
        LR = rec_multiply_full(tL, wR, level + 1)
        RL = rec_multiply_full(tR, wL, level + 1)
        return merge_mats[level - 1](torch.cat([LL - RR, LR + RL], dim=-1))

    def build_world_vec(tok_ids):
        """Hardcoded routing: token determines path by binary decomposition of ID."""
        K = tok_ids.shape[0]; w = torch.zeros(K, d, device=device)
        for lvl in range(1, FD):
            n_br = 2 ** (lvl - 1); br_dim = d // (2 ** lvl)
            nodes = br_nodes[lvl - 1]  # [n_br, 2, br_dim]
            for br in range(n_br):
                # Simple deterministic: even token IDs → left (idx=0), odd → right (idx=1)
                sel = (tok_ids + br) % 2
                ps = br * (2 * br_dim)
                w[:, ps:ps + br_dim] = nodes[br, 0] * (1 - sel.float()).unsqueeze(-1) + nodes[br, 1] * sel.float().unsqueeze(-1)
        return w

    def heap_full_pos(tok_ids, tau):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(build_world_vec(t, tau), dim=-1)
        return rec_multiply_full(t, w, 1)

    opt_ce = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    all_w = list(merge_mats.parameters()) + list(br_nodes)
    opt_world = torch.optim.Adam(all_w, lr=lr)

    n_shared = sum(bn.numel() for bn in br_nodes)
    n_merge = sum(mm.weight.numel() for mm in merge_mats)
    print(f"\n{'='*60}")
    print(f"PHASE HEAP_FULL: epochs={EPOCHS} depth={FD} shared={n_shared} merge={n_merge}")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        tau_now = max(1.0, 10.0 * (1 - ep / 100))
        for bi in range(0, 5000, 16):
            do_anchor = (ti % 2 == 0)
            if do_anchor:
                opt_world.zero_grad()
                L0.eval(); L1.eval()
                for mm in merge_mats: mm.train()
                en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
                zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
                n_pairs = min(len(en_ids), len(zh_ids))
                if n_pairs > 1:
                    en_w = heap_full_pos(en_ids[:n_pairs], tau_now)
                    zh_w = heap_full_pos(zh_ids[:n_pairs], tau_now)
                    logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
                    loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_world.param_groups[0]['params'], 2.0)
                    opt_world.step()
                    tl += loss.item(); ti += 1
            else:
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt_ce.zero_grad()
                    L0.train(); L1.train()
                    for mm in merge_mats: mm.eval()
                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T); ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt_ce.param_groups[0]['params'], 2.0)
                        opt_ce.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0
            en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
            zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
            n_pairs = min(len(en_ids), len(zh_ids))
            full_acc = 0.0
            if n_pairs > 5:
                with torch.no_grad():
                    en_w = F.normalize(heap_full_pos(en_ids[:n_pairs], tau_now), dim=-1)
                    zh_w = F.normalize(heap_full_pos(zh_ids[:n_pairs], tau_now), dim=-1)
                    logits = en_w @ zh_w.T / 0.07
                    full_acc = 100 * (logits.argmax(-1) == torch.arange(n_pairs, device=device)).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} repair_BLEU={rb:.1f} tau={tau_now:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} full_acc={full_acc:.1f}% {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(), 'epochs': EPOCHS, 'full_depth': FD}
    torch.save(ckpt, f'{save_dir}/anchor_heap_full.pt')
    print(f"Saved: anchor_heap_full.pt  full_depth={FD}")

# ═══════════════════════════════════════════════
# PHASE M: Proper recursive heap Multiply — the real tree algebra
# ═══════════════════════════════════════════════
elif phase == 'heap_proper':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    HP_DEPTH = int(args.get('hp_depth', '3'))

    # Merge matrices: one per internal level (depth-1 levels, from leaf-up)
    merge_mats = nn.ModuleList()
    for i in range(HP_DEPTH - 1):
        dim_at_level = d // (2 ** i)
        merge_mats.append(nn.Linear(dim_at_level, dim_at_level).to(device))
        nn.init.eye_(merge_mats[-1].weight); nn.init.zeros_(merge_mats[-1].bias)

    # Shared tree nodes for routing (same as tree_alt)
    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(HP_DEPTH)])
    for tn in t_nodes:
        nn.init.normal_(tn.weight, 0, 0.1)

    def tw_vec(tok_ids):
        w = torch.zeros(len(tok_ids), d, device=device)
        for level in range(HP_DEPTH):
            nidx = torch.clamp(tok_ids // (V // (2 ** level)), 0, (2 ** level) - 1) if level > 0 else torch.zeros_like(tok_ids)
            w = w + t_nodes[level](nidx)
        return w

    def heap_multiply(t_vec, w_vec, level):
        """Recursive heap multiplication — the real tree algebra."""
        if level == HP_DEPTH:  # leaf: Hadamard
            return t_vec * w_vec
        sub = t_vec.shape[-1] // 2
        tL, tR = t_vec[..., :sub], t_vec[..., sub:]
        wL, wR = w_vec[..., :sub], w_vec[..., sub:]
        LL = heap_multiply(tL, wL, level + 1)
        RR = heap_multiply(tR, wR, level + 1)
        LR = heap_multiply(tL, wR, level + 1)
        RL = heap_multiply(tR, wL, level + 1)
        return merge_mats[level - 1](torch.cat([LL - RR, LR + RL], dim=-1))

    def hp_world_pos(tok_ids):
        t = F.normalize(L0.weight[tok_ids], dim=-1)
        w = F.normalize(tw_vec(tok_ids), dim=-1)
        return heap_multiply(t, w, 1)  # start from level 1 (root's children)

    opt_ce = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=lr)
    all_hp = list(merge_mats.parameters()) + [p for tn in t_nodes for p in tn.parameters()]
    opt_world = torch.optim.Adam(all_hp, lr=lr)

    n_hp = sum(p.numel() for p in all_hp)
    print(f"\n{'='*60}")
    print(f"PHASE HEAP_PROPER: epochs={EPOCHS} depth={HP_DEPTH} params={n_hp/1e3:.1f}K")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(all_auto)
        tl, ti = 0, 0
        for bi in range(0, 5000, 16):
            do_anchor = (ti % 2 == 0)
            if do_anchor:
                opt_world.zero_grad()
                L0.eval(); L1.eval()
                for mm in merge_mats: mm.train()
                for tn in t_nodes: tn.train()
                en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
                zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
                n_pairs = min(len(en_ids), len(zh_ids))
                if n_pairs > 1:
                    en_w = hp_world_pos(en_ids[:n_pairs])
                    zh_w = hp_world_pos(zh_ids[:n_pairs])
                    logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
                    loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_world.param_groups[0]['params'], 2.0)
                    opt_world.step()
                    tl += loss.item(); ti += 1
            else:
                batch = all_auto[bi:bi + 16]
                if batch:
                    opt_ce.zero_grad()
                    L0.train(); L1.train()
                    for mm in merge_mats: mm.eval()
                    for tn in t_nodes: tn.eval()
                    bl, ns = torch.tensor(0.0, device=device), 0
                    for ids in batch:
                        T = min(len(ids), MAX_LEN); ids = ids[:T]
                        ids_d = ids[:]; kept = 0
                        for i in range(len(ids_d)):
                            if random.random() > 0.25 or kept < 2: kept += 1
                            else: ids_d[i] = 1
                        ids_pad, _ = pad_to_heap(ids_d, T); ids_tgt, _ = pad_to_heap(ids, T)
                        with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
                        ctx = L1.fe(emb); dec = L1.fd(ctx)
                        loss = F.cross_entropy(dec.squeeze(0)[:T] @ L0.weight.T, ids_tgt[:T])
                        bl += loss; ns += 1
                    if ns > 0:
                        (bl / ns).backward()
                        torch.nn.utils.clip_grad_norm_(opt_ce.param_groups[0]['params'], 2.0)
                        opt_ce.step()
                        tl += (bl / ns).item(); ti += 1

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            rb = eval_repair_bleu()
            cos0, cos1 = eval_anchor_cos()
            elapsed = time.time() - t0
            en_ids = torch.tensor([e[0] for e, _ in ANCHORS if ok(e) and ok(_)], device=device)
            zh_ids = torch.tensor([z[0] for _, z in ANCHORS if ok(_) and ok(z)], device=device)
            n_pairs = min(len(en_ids), len(zh_ids))
            hp_acc, hp_open = 0.0, 0.0
            if n_pairs > 5:
                with torch.no_grad():
                    en_w = F.normalize(hp_world_pos(en_ids[:n_pairs]), dim=-1)
                    zh_w = F.normalize(hp_world_pos(zh_ids[:n_pairs]), dim=-1)
                    logits = en_w @ zh_w.T / 0.07
                    hp_acc = 100 * (logits.argmax(-1) == torch.arange(n_pairs, device=device)).float().mean().item()
                    # Open-set
                    all_zh = F.normalize(hp_world_pos(torch.arange(V, device=device)), dim=-1)
                    cos = en_w @ all_zh.T
                    hp_open = 100 * (cos.argmax(-1) == zh_ids[:n_pairs]).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f} repair_BLEU={rb:.1f} "
                  f"cos_L0={cos0:.3f} cos_L1={cos1:.3f} hp_acc={hp_acc:.1f}% open={hp_open:.1f}% {elapsed:.0f}s")

    ckpt = {'L0': L0.state_dict(), 'L1': L1.state_dict(), 'epochs': EPOCHS, 'hp_depth': HP_DEPTH}
    torch.save(ckpt, f'{save_dir}/anchor_hp.pt')
    print(f"Saved: anchor_hp.pt  hp_depth={HP_DEPTH}")

# ═══════════════════════════════════════════════
# PHASE P: Radix observation — per-level cosine + InfoNCE
# ═══════════════════════════════════════════════
elif phase == 'tree_radix':
    EPOCHS = int(args.get('epochs', '200'))
    lr = float(args.get('lr', '0.003'))
    RD = int(args.get('radix_depth', '3'))

    merge_mats = nn.ModuleList()
    for i in range(RD - 1):
        dim_at_level = d // (2 ** i)
        merge_mats.append(nn.Linear(dim_at_level, dim_at_level).to(device))
        nn.init.eye_(merge_mats[-1].weight); nn.init.zeros_(merge_mats[-1].bias)

    t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(RD)])
    for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
    rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
    for i in range(1, RD): nn.init.zeros_(t_nodes[i].weight)

    def tw_vec(tok_ids):
        w = torch.zeros(len(tok_ids), d, device=device)
        for l in range(RD):
            nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
            w = w + t_nodes[l](nidx)
        return w

    def radix_multiply(t_vec, w_vec, level):
        if level == RD: return t_vec * w_vec, [t_vec * w_vec]
        sub = t_vec.shape[-1] // 2
        tL, tR = t_vec[..., :sub], t_vec[..., sub:]
        wL, wR = w_vec[..., :sub], w_vec[..., sub:]
        LL, obs_LL = radix_multiply(tL, wL, level + 1)
        RR, obs_RR = radix_multiply(tR, wR, level + 1)
        LR, obs_LR = radix_multiply(tL, wR, level + 1)
        RL, obs_RL = radix_multiply(tR, wL, level + 1)
        merged = merge_mats[level - 1](torch.cat([LL - RR, LR + RL], dim=-1))
        return merged, [merged] + obs_LL

    def radix_world(t_vec, w_vec):
        _, obs = radix_multiply(t_vec, w_vec, 1)
        return obs

    def wp_mean_radix(padded, mask):
        N_b, M = padded.shape
        t_vec = F.normalize((L0.weight[padded] * mask.unsqueeze(-1)).sum(1), dim=-1)
        w_flat = tw_vec(padded.reshape(-1))
        w_3d = w_flat.reshape(N_b, M, d)
        w_vec = F.normalize((w_3d * mask.unsqueeze(-1)).sum(1), dim=-1)
        return radix_world(t_vec, w_vec)

    # Training anchors (LaBSE guided)
    en_ank_full = [e for e, _ in ANCHORS if ok(e) and ok(_)]
    zh_ank_full = [z for _, z in ANCHORS if ok(_) and ok(z)]
    N = min(len(en_ank_full), len(zh_ank_full))
    en_ank_full, zh_ank_full = en_ank_full[:N], zh_ank_full[:N]
    max_len = max(len(ids) for ids in en_ank_full + zh_ank_full)
    en_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    zh_pad = torch.zeros(N, max_len, dtype=torch.long, device=device)
    en_mask = torch.zeros(N, max_len, device=device)
    zh_mask = torch.zeros(N, max_len, device=device)
    for i, ids in enumerate(en_ank_full):
        L = len(ids); en_pad[i, :L] = torch.tensor(ids, device=device); en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_ank_full):
        L = len(ids); zh_pad[i, :L] = torch.tensor(ids, device=device); zh_mask[i, :L] = 1.0 / L

    # Gold anchors (manual only)
    en_gold, zh_gold = [], []
    for en_word, zh_word in ANCHOR_WORDS:
        ei = sp.encode_as_ids(en_word); zi = sp.encode_as_ids(zh_word)
        if ok(ei) and ok(zi): en_gold.append(ei); zh_gold.append(zi)
    Ng = min(len(en_gold), len(zh_gold))
    en_gold, zh_gold = en_gold[:Ng], zh_gold[:Ng]
    gmax = max(len(ids) for ids in en_gold + zh_gold)
    g_en_pad = torch.zeros(Ng, gmax, dtype=torch.long, device=device)
    g_zh_pad = torch.zeros(Ng, gmax, dtype=torch.long, device=device)
    g_en_mask = torch.zeros(Ng, gmax, device=device)
    g_zh_mask = torch.zeros(Ng, gmax, device=device)
    for i, ids in enumerate(en_gold):
        L = len(ids); g_en_pad[i, :L] = torch.tensor(ids, device=device); g_en_mask[i, :L] = 1.0 / L
    for i, ids in enumerate(zh_gold):
        L = len(ids); g_zh_pad[i, :L] = torch.tensor(ids, device=device); g_zh_mask[i, :L] = 1.0 / L

    all_params = list(L0.parameters()) + list(merge_mats.parameters())
    for tn in t_nodes: all_params += list(tn.parameters())
    opt = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-4)

    level_weights = [2.0, 1.0, 0.5, 0.2][:RD]

    print(f"\n{'='*60}")
    print(f"PHASE TREE_RADIX: epochs={EPOCHS} depth={RD} anchors={N} gold={Ng}")
    print(f"  level weights: {level_weights}")
    print("=" * 60)

    t0 = time.time()
    for ep in range(EPOCHS):
        L0.train()
        for mm in merge_mats: mm.train()
        for tn in t_nodes: tn.train()
        tl = 0.0
        random.shuffle(all_auto)
        for bi in range(0, 500, 1):
            opt.zero_grad()
            en_obs = wp_mean_radix(en_pad, en_mask)
            zh_obs = wp_mean_radix(zh_pad, zh_mask)
            loss = torch.tensor(0.0, device=device)
            for lv, (e_l, z_l, w) in enumerate(zip(en_obs, zh_obs, level_weights)):
                logits = F.normalize(e_l, dim=-1) @ F.normalize(z_l, dim=-1).T / 0.07
                loss = loss + w * F.cross_entropy(logits, torch.arange(N, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            tl += loss.item()

        if ep % max(EPOCHS // 5, 1) == 0 or ep == EPOCHS - 1:
            elapsed = time.time() - t0
            with torch.no_grad():
                en_g_obs = wp_mean_radix(g_en_pad, g_en_mask)
                zh_g_obs = wp_mean_radix(g_zh_pad, g_zh_mask)
                en_r = F.normalize(en_g_obs[0], dim=-1)
                zh_r = F.normalize(zh_g_obs[0], dim=-1)
                gold_acc = 100 * ((en_r @ zh_r.T / 0.07).argmax(-1) == torch.arange(Ng, device=device)).float().mean().item()
                n_show = min(Ng, 200)
                cos_str = ' '.join([f"lv{lv}={F.cosine_similarity(en_g_obs[lv][:n_show], zh_g_obs[lv][:n_show], dim=-1).mean():.3f}" for lv in range(RD)])
                en_tr = wp_mean_radix(en_pad, en_mask)
                zh_tr = wp_mean_radix(zh_pad, zh_mask)
                en_tr_r = F.normalize(en_tr[0], dim=-1)
                zh_tr_r = F.normalize(zh_tr[0], dim=-1)
                nce_acc = 100 * ((en_tr_r @ zh_tr_r.T / 0.07).argmax(-1) == torch.arange(N, device=device)).float().mean().item()
            print(f"  ep {ep:3d} loss={tl/500:.4f} nce={nce_acc:.1f}% gold={gold_acc:.1f}% cos=[{cos_str}] {elapsed:.0f}s", flush=True)

    ckpt = {'L0': L0.state_dict(), 'epochs': EPOCHS, 'radix_depth': RD}
    torch.save(ckpt, f'{save_dir}/anchor_tree_radix.pt')
    print(f"Saved: anchor_tree_radix.pt  radix_depth={RD}")
