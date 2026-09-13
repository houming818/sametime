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

## 7. Smoke 结果

taskd 418 在修订合同下完成。共抽取 480 个真实平行样本，六个表达族各 40 正例、40 负例；样本
manifest SHA-256 为 `940c7d11f201c7df45c9ac37f0482d567134d34090f5ced4a51cdf0f5eff61ed`。
运行输入代码与提交 `c5f1685` 的预结果逻辑文件哈希完全一致；当前逻辑文件追加结果后哈希自然变化。
checkpoint 前后逐 tensor 未变化。GPU 维持 270 W 限制，
采样最高温度 62 摄氏度、最高显存 878 MiB，未见 CUDA、Xid 或非有限值。

跨六个留出族、三个 protocol depth 的宏平均结果为：

| 特征 | AUROC |
|---|---:|
| source token 计数词袋 | 0.5327 |
| FOLD leaf | 0.6003 |
| FOLD root | 0.5974 |
| READ-facing leaf | 0.6003 |
| READ-facing root | 0.5842 |

FOLD root 相对 leaf 仅下降 `0.0029`，READ-facing root 相对 leaf 下降 `0.0161`，均远小于预注册的
0.10 压缩损失判据。各 depth 也没有随递归加深而单调恶化：FOLD root 均约 0.60，READ-facing root
约为 0.58、0.59、0.58。乱序标签对照的 root 宏平均约 0.45--0.47，没有伪造正向泛化。

READ-facing root 按留出表达族汇总后的 AUROC 约为：

| 留出族 | AUROC | 观察 |
|---|---:|---|
| physical | 0.53 | 接近随机 |
| technical | 0.46 | 低于随机，方向不能迁移 |
| abstract | 0.59 | 弱信号 |
| press | 0.65 | 局部可恢复 |
| force | 0.59 | 弱信号 |
| other_tui | 0.68 | 局部可恢复 |

O0--O2 与 C0 通过；P1、P2 未通过；两项 P3 压缩损失模式均未出现。

## 8. 当前结论

F16 不支持“跨语言 `push` 目标信息主要在 leaf 已形成、随后被高层递归退火抹除”。当前 checkpoint
从 leaf 开始就没有形成跨六类中文表达稳定泛化的 target-side `push*` 线性方向；root 与 leaf 几乎持平，
因此不能把这次失败隔离归因于 FOLD 压缩。

隐态约 0.60 的宏平均高于同一探针下的 source token 词袋 0.53，说明模型变换提供了一些额外预测信号，
但信号按语义族高度分裂：`press` 与 `other_tui` 可部分迁移，`technical` 与 `physical` 不可迁移。
这更符合“模型学到若干局部翻译协议，但尚未凝聚为统一、跨语境的 `push` 目标方向”。

因此下一步不应立即替换退火公式。更有因果针对性的阶梯是：先用多动词、多义项的目标存在性辅助任务，
使 leaf 出现稳定的跨族方向；随后在同一 checkpoint 上重跑 F16。只有 leaf 达标而 root 明显落后时，
才进入 FOLD 算子修订；若 root 同步达标而自由生成仍失败，则转向 READ/Decoder 对齐与目标读出训练。
