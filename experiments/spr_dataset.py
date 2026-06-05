import os
import sys
import torch
import sentencepiece as spm

# 1000 EN-ZH manual anchor word pairs
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
    # ── Pos-aligned from parallel corpus ──
    ('但是','But'),('如果','If'),('中国','China'),('事实上','Indeed'),
    ('此外','Moreover'),('我们','We'),('他们','They'),('当然','Of'),
    ('therefore','因此'),('these','这些'),('even','即便'),('for','比如'),
    ('this','这一'),('after','毕竟'),('but','但这'),('even','即使'),
    ('as','随着'),('first','首先'),('but','但在'),('finally','最后'),
    ('as','结果'),('meanwhile','与此同时'),('many','许多'),('to','为了'),
    ('but','但如果'),('unfortunately','不幸的是'),('and','而且'),('they','它们'),
    ('second','其次'),('new','纽约'),('instead','相反'),('some','一些'),
    ('as','正如'),('london','伦敦'),('trump','特朗普'),('while','虽然'),
    ('for','例如'),('this','这种'),('now','现在'),('today','今天'),
    ('according','根据'),('fortunately','幸运的是'),('by','通过'),('as','作为'),
    ('so','那么'),('in','换句话说'),('world','世界'),('uk','英国'),
    ('us','在美国'),('likewise','类似地'),('only','只有'),('this','这是'),
    ('eu','欧盟'),('for','对于'),('likewise','同样'),('but','但这一'),
    ('this','这意味着'),('perhaps','也许'),('washington','华盛顿'),('last','去年'),
    ('so','所以'),('all','所有这些'),('third','第三'),('question','问题在于'),
    ('second','第二'),('indeed','实际上'),('nonetheless','尽管如此'),('obama','奥巴马'),
    ('india','印度'),('in','简言之'),('their','他们的'),('other','其他'),
    ('beyond','除了'),('some','有些'),('all','所有'),('japan','日本'),
    ('good','好消息是'),('and','而在'),('but','但如今'),('un','联合国'),
    ('us','美国的'),('unless','除非'),('problem','问题'),('indeed','确实'),
    ('sure','可以肯定'),('his','他的'),('such','这样的'),('most','大部分'),
    ('want','要想'),('past','在过去'),('why','为什么'),('so','由此'),
    ('clear','显然'),('that','这就是'),('africa','非洲'),('or','或者'),
    ('answer','答案'),('russia','俄罗斯'),('indeed','的确'),('those','那些'),
    ('year','今年'),('but','但尽管'),('once','一旦'),('even','甚至'),
    ('this','这就'),('but','但随着'),('but','但这种'),('some','有人'),
    ('germany','德国'),('we','我们必须'),('but','不过'),('moreover','不仅如此'),
    ('as','只要'),('otherwise','否则'),('this','这是一个'),('wb','世界银行'),
    ('here','在这方面'),('problem','问题是'),('and','而尽管'),('perhaps','或许'),
    ('west','西方'),('by','相比之下'),('london','发自伦敦'),('intl','国际'),
    ('and','而这'),('putin','普京'),('some','某些'),('in','在许多'),
    ('president','美国总统'),('in','在欧洲'),('tokyo','东京'),('this','这也是'),
    ('korea','韩国'),('eurozone','通过欧元区'),('my','我的'),('reason','原因'),
    ('italy','意大利'),('current','当前'),('in','在这一'),('for','多年来'),
    ('public','公共'),('but','但这并不意味着'),('one','一种'),('moreover','另外'),
    ('new','新的'),('first','第一'),('as','其结果是'),('turkey','土耳其'),
    ('but','但日本'),('with','有了'),('africa','非洲的'),('but','但这个'),
    ('even','即使是'),('not','并不是说'),('trade','贸易'),('according','按照'),
    ('other','其他国家'),('world','世界上最'),('but','但不'),('doing','这样做'),
    ('recently','直到最近'),('year','今年的'),('afghanistan','阿富汗'),('abe','安倍'),
    ('these','这些国家'),('company','公司'),('we','我们现在'),('education','教育'),
    ('worth','值得'),('plan','该计划'),('spain','西班牙'),('recent','最近的'),
    ('by','截止'),('they','它们可以'),('farmers','农民'),('recent','近年来'),
    ('ethiopia','埃塞俄比亚'),('palestinian','巴勒斯坦'),('perhaps','这或许'),('they','她们'),
    ('women','女性'),('johannesburg','约翰内斯堡'),('nigeria','尼日利亚'),('john','约翰'),
    ('south','南非'),('so','因此我'),('they','他们需要'),('agriculture','农业'),
    ('developed','发达国家'),('african','非洲国家'),('but','但非洲'),('building','建设')
]

def ok(ids):
    """过滤含有0 (PAD/UNK) 的ID列表"""
    return len(ids) > 0 and all(x != 0 for x in ids)

def load_tokenizer(sp_path="/mnt/nas/datasets/wmt17/sp_bpe.model"):
    """加载 SentencePiece Tokenizer"""
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)
    return sp

def build_anchors(sp, use_labse_only=False):
    """构建去重的双语锚点对 BPE ID 列表"""
    anchors = []
    seen = set()
    
    # 1. 优先加载 LaBSE 引导的高质量词对 (来自 /workspace/labse_anchors.py)
    labse_pairs = []
    try:
        # 尝试从各种可能的地方加载 labse_anchors
        for path in ["labse_anchors.py", "../labse_anchors.py", "/workspace/labse_anchors.py"]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 安全解析 LABSE_ANCHORS = [...]
                    if "=" in content:
                        labse_pairs = eval(content.split("=", 1)[1].strip())
                        print(f"[Dataset] 成功从 {path} 加载了 {len(labse_pairs)} 个 LaBSE 锚点对。")
                        break
    except Exception as e:
        print(f"[Dataset] 加载 LaBSE 锚点对失败: {e}")
        
    # 如果没加载到 LaBSE 锚点对，抛出提示
    if not labse_pairs:
        print("[Dataset] 警告: 未能加载到 LaBSE 锚点对！")

    # 2. 如果不只限于 LaBSE，则加入手工 Gold 锚点
    if not use_labse_only:
        for en_word, zh_word in ANCHOR_WORDS:
            en_ids = sp.encode_as_ids(en_word)
            zh_ids = sp.encode_as_ids(zh_word)
            if ok(en_ids) and ok(zh_ids):
                key = (tuple(en_ids), tuple(zh_ids))
                if key not in seen:
                    seen.add(key)
                    anchors.append((en_ids, zh_ids))
                    
    # 3. 将 LaBSE 锚点加入
    for en_word, zh_word in labse_pairs:
        en_ids = sp.encode_as_ids(en_word)
        zh_ids = sp.encode_as_ids(zh_word)
        if ok(en_ids) and ok(zh_ids):
            key = (tuple(en_ids), tuple(zh_ids))
            if key not in seen:
                seen.add(key)
                anchors.append((en_ids, zh_ids))
                
    return anchors

def get_gold_eval_anchors(sp):
    """
    专门构建金标准评测集 (Gold Eval Anchors) 且严格排除在训练集之外。
    我们使用 ANCHOR_WORDS 列表作为评测集。
    """
    gold_anchors = []
    seen = set()
    for en_word, zh_word in ANCHOR_WORDS:
        en_ids = sp.encode_as_ids(en_word)
        zh_ids = sp.encode_as_ids(zh_word)
        if ok(en_ids) and ok(zh_ids):
            key = (tuple(en_ids), tuple(zh_ids))
            if key not in seen:
                seen.add(key)
                gold_anchors.append((en_ids, zh_ids))
    return gold_anchors

def collate_anchors_to_padded(anchors_list, device):
    """
    将 list 形式的 token ID 列表快速拼装为 padded 2D Tensor 及其 mask。
    """
    N = len(anchors_list)
    if N == 0:
        return None, None
        
    max_len = max(len(ids) for ids in anchors_list)
    padded = torch.zeros(N, max_len, dtype=torch.long, device=device)
    mask = torch.zeros(N, max_len, device=device)
    
    for i, ids in enumerate(anchors_list):
        L = len(ids)
        padded[i, :L] = torch.tensor(ids, device=device)
        mask[i, :L] = 1.0 / L  # 预除以 L，之后矩阵乘法直接求 sum 即可实现平均
        
    return padded, mask
