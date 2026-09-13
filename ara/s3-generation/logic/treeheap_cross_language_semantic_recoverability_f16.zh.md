# TreeHeap 跨语言语义可恢复性审计

日期：2026-09-13
Claim：`S3-TREEHEAP-CROSS-LANGUAGE-SEMANTIC-RECOVERABILITY-F16`
状态：预注册，尚未观察结果。

## 1. 问题

F15 已证明，重复 `推` 的 source 计数在 root 与 READ-facing root 中都可被线性恢复。因此，严格的
“高层没有收到任何 `推` 信号”已被否证。但计数存在不等于跨语言语义存在：root 可能只保存了 source
词形或数量，却没有保存足以区分英文 target 是否应出现 `push*` 的上下文方向。

F16 在真实平行语料上冻结同一 TreeHeap-106M checkpoint，检测各层状态能否预测英文 target 中是否出现
`push / pushes / pushed / pushing`。实验不更新模型，也不以当前 Decoder 的成功或失败代替隐态可恢复性。

## 2. 数据合同

- 数据固定为 `NioClean-ZHEN-S098-7M-v2/pairs.tsv`；
- 只使用 source 含与“推”相关线索的六个互斥表达族：`physical`、`technical`、`abstract`、
  `press`、`force`、`other_tui`；
- 每族确定性抽取 40 个 target 含 `push*` 的正例和 40 个不含 `push*` 的负例，共 480 例；
- source 必须含至少四个汉字，正文 SentencePiece 长度在 2--30；加方向 token 与 EOS 后总长不超过
  底座固定 leaf 宽度 32；
- 每族正负样本按 source token 长度贪心配对，减少长度泄漏；
- 保存原始行号、双语文本、表达族、标签、token 长度和样本清单 SHA-256。

六族由固定正则按顺序互斥分配。数据标签只描述 target-side `push*` 是否出现，不把 score<0.98 或
未使用 `push` 的翻译称为错误。

## 3. 泛化协议

采用 leave-one-family-out：每次完整留出一个表达族的 80 个样本作为测试，其余五族作为训练。这样探针
不能靠记住同一中文词形族完成测试。对训练标签的负对照在每个训练族内部独立乱序，保持族别和样本数不变。

固定比较：

1. 原始 source SentencePiece 计数词袋 `input_bow`；
2. 原生 FOLD 的 `root / middle / leaf` masked-mean 状态；
3. Decoder convolution 后的 READ-facing `root / middle / leaf` 状态；
4. protocol depth 5、6、7；
5. 每个条件使用训练集标准化后的 mean-difference 线性方向，不反传到模型。

主指标是未见表达族上的 AUROC；另报告 AP、balanced accuracy、正负均值间隔和五次族内乱序标签对照。

## 4. 预注册门槛

- O0：六族均恰有 40 正例与 40 负例，六个留出折完整且无行号重叠；
- O1：全部特征、分数和指标有限；
- O2：checkpoint 前后逐 tensor 完全一致；
- C0：native READ-facing root 的五次乱序对照，在全部 depth 与留出族上的平均 AUROC 小于 0.60；
- P1：native FOLD root 在六族、三 depth 上的宏平均 AUROC 不低于 0.65；
- P2：native READ-facing root 的宏平均 AUROC 不低于 0.65；
- P3：若 leaf 宏平均 AUROC 不低于 0.70，而 root 比同阶段 leaf 低至少 0.10，才支持“跨语言目标信息随递归压缩显著损失”。

判读优先级：

- P1/P2 通过且与 leaf 差距小于 0.10：语义标签方向到达高层，后续优先修 READ/Decoder 对齐；
- leaf 清晰、root 显著落后：支持修订 FOLD/退火算子；
- leaf 也失败：当前 checkpoint 尚未形成可泛化的 target-side `push` 方向，不能归因于高层压缩；
- `input_bow` 明显优于所有隐态：模型变换损害了 source 中原本可用的线索；
- 隐态优于 `input_bow`：只说明模型形成了额外可线性读出的预测信息，不等于完整翻译能力。

## 5. 结论边界

这是冻结 checkpoint、单一英文词族、自动标签的观察实验。它检测的是“target-side `push*` 标签的跨表达族
线性可恢复性”，不是哲学意义上的语义，也不证明生成器能输出正确句子。任何结构替换必须在后续成对训练中
验证；F16 本身不修改退火公式。

## 6. Smoke 修订记录

taskd 416 在首个模型 batch、任何结果产生前失败：初版误把 32 个正文 token 与方向 token、EOS 一起
装入固定宽度 32 的底座，实际 tensor 宽度成为 34。失败现场保留，不作为阴性实验结果。修订版将正文
上限改为 30、输入 tensor 总宽度固定为 32；其他样本、分组、指标与门槛不变。
