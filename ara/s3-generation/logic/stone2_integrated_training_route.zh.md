# STONE-2 统一训练航线

日期：2026-08-23

Claim：`S3-STONE2-INTEGRATED-C03`

状态：组合 smoke 已执行；8/9 门通过，S7 失败，正式长训练暂缓。

## 1. 靶训练目标

STONE-2 要得到一个可恢复、可继续训练、可通过 CLI 使用的 TreeHeap 基础
checkpoint。它必须先从自然文本形成私有概率协议，再在同一套参数和算子上学习
中英双向翻译与中文问答。最终输出必须因果依赖 TreeHeap 的 Butterfly 通信、
递归 FOLD 和多层 `H_state`，不能退化为 flat、root-only、leaf-only 或固定输出
捷径。

完整航线为：

```text
不可变数据 release
-> 自然文本 Pretrain
-> 匹配的 Task Train
-> 冻结 checkpoint
-> posterior / generation Proof
-> TreeHeap 结构干预
-> CLI、重载和 release 审计
```

## 2. 固定候选架构

```text
SentencePiece token + 独立 PAD/EOS/任务槽位
-> token WRITE
-> 动态宽度 XOR Butterfly
-> 零点参考系、有界、可逆 FOLD
-> H_state(root + addressed parent/detail levels)
-> 无 learned STOP 的强制多层 READ
-> recurrent Decoder
-> token probability container
-> autoregressive collapse
```

组合 smoke 使用 C13 `ref_zero` FOLD 与 C11 `READ-only`。第二个 bottom-up
`K_up` 不进入候选，因为历史 bypass 损失只有 `0.000340` NLL。K7 浅层 read
adapter、BLANK curriculum、root-exclusive Decoder、teacher distillation 和 learned
STOP 均不进入默认航线。

当前 runtime `H_state` 是 TreeHeap；共享 Kernel 的学习参数仍是张量集合，而不是
可扩展的参数 TreeHeap。STONE-2 不把参数内存拓扑重构列为通过条件。

## 3. 数据合同

### Stage A：自然文本 Pretrain

输入 release：`NioText-ZH-Integrity-2985K-v1`。完整性策略只排除明确的
`mojibake` 与 `extreme_repetition`；URL、低 CJK 比例等保留为 metadata，不在
物化时擅自删除。

训练目标为 next-span cross entropy，source 宽度覆盖
`4/8/16/32/64/128/256`，target 默认 32 pieces。训练流必须持续读取新窗口，
不得循环一个小型内存表。

### Stage B：中英双向翻译

输入 release：`NioClean-ZHEN-S098-7M-v2`，外部评价固定使用原始 WMT
held-out。训练采用 Native Butterfly，并把 C07 的 20% additive Identity replay
只作为任务阶段候选。必须保留同初始化、同 token 预算的 raw/selected 数据对照，
避免把额外更新误写成语料选择收益。

### Stage C：中文问答

在 `NioQA-ZH-S090-v1`、`S095`、`S098` 三个嵌套 view 上先做匹配阶梯，
再选择正式 task release。医疗问答属于独立领域续训，不作为通用 STONE-2
通过条件。

## 4. 组合 smoke 门

组合 smoke 只检查整条计算图，不支持产品 Claim：

```text
S0 数据、tokenizer、初始化与训练流 hash 完整
S1 FOLD/UNFOLD max_abs_error < 1e-4
S2 无 STOP 参数，无 raw leaf/string bypass
S3 root、至少两个 parent depth 与 leaf 都获得有限非零梯度
S4 每层 FOLD state 均获得梯度；Butterfly、READ 和 Decoder 参数获得有限非零梯度
S5 训练后 valid NLL 低于初始化
S6 source shuffle、runtime Identity、pair break 至少产生正 NLL 损伤
S7 leaf-only READ 比 native 更差，且至少两个非 leaf depth 消融有正损伤
S8 greedy 非空、无单一固定输出全面坍缩
S9 checkpoint 严格重载后固定 greedy token IDs 完全一致
```

任意 smoke 门失败，都必须停止，不允许靠延长训练掩盖架构错误。

## 5. 正式训练门

### Pretrain

- held-out continuation NLL 相对初始化改善至少 `0.20`；
- 模型到经验候选分布的 JS 比 unigram 至少低 `0.02`；
- 经验候选集合概率质量比 unigram 至少高 `0.02`；
- wrong-source 改变至少 25% 的 greedy collapse；
- 非空率 100%，严重重复率不高于 25%。

### 匹配任务迁移

- Pretrain 初始化比同一随机起点 scratch 低至少 `0.02` validation NLL；
- chrF 和标准 sacreBLEU 不得系统性下降；
- raw/selected 对照必须同初始化、同 batch stream、同 token/update 预算。

### 产品与结构

- 严重重复率不高于 10%；
- source shuffle damage、Butterfly Identity override、pair break 均为正且达到各自
  预注册阈值；
- leaf-only 至少恶化 `0.05` NLL；
- 至少两个非 leaf depth 消融各恶化 `0.01` NLL；
- 四个长度桶均保持 source dependence；
- checkpoint 重载和中断续训完全可复现；
- CLI 必须如实声明 translation 或 single-turn QA，不能互相改标签。

## 6. 执行与停止规则

执行顺序：

```text
物化并审核 release
-> 组合 smoke
-> 100M target-piece Pretrain pilot
-> 匹配 WMT PT/SC + raw/selected proof
-> 冻结结构和生成审计
-> 只有全部通过才进入全量 Pretrain 与 Task Train
```

出现 OOM、CUDA 故障、NaN/Inf、hash 不一致、重载失败、固定输出坍缩、结构损伤
归零或连续三个正式 wake 的原始 held-out 指标恶化时立即停止。GPU 保持现有 270W
限制，任务严格串行。每个 wake 记录 GPU 小时、训练 token、wall time、NLL/PPL、
生成样例、结构损伤和 checkpoint SHA-256，并通过 `sendme` 邮件通知。

## 7. 解释边界

通过最多说明：固定容量的 TreeHeap 形成了可迁移的 source-conditioned 私有协议，
并能在翻译或单轮问答中产生可测量、结构因果的输出。它不证明意识、完整世界
模型、参数本身已经 TreeHeap 化，也不证明优于其他架构。

## 8. 2026-08-23 执行结果

任务 `292` 完成了统一 smoke。训练、闭包、无 STOP、全层梯度、参数梯度、三项
结构干预、非固定输出和重载均通过；S7 未通过，因此没有启动 100M-piece 正式
Pretrain。

任务 `293` 对冻结 PT checkpoint 做了 coarse/middle/fine 全组合诊断。三组的
Shapley 贡献均为正，合计 Test NLL 收益 `0.1311`；两两交互均为负。当前最严谨
结论是：多分辨率 READ 已产生分布式贡献，但共享 READ kernel 在不同深度之间
存在冗余或能量干扰。下一项结构实验应预注册深度能量归一化或交互约束，不能
靠延长训练补签 S7。

任务 `294` 把五个数据 view 固化为不可变 release，并完成独立 SHA-256 复核：

- `NioText-ZH-Integrity-2985K-v1`：2,972,976 行；
- `NioClean-ZHEN-S098-7M-v2`：7,304,358 对；
- `NioQA-ZH-S090/S095/S098-v1`：4,767,788 / 4,180,947 / 3,375,921 行。

统一数据根哈希为
`75caafdc24058eb96a957fd680b41789843eb3726e4febb4a110b7c96b38be29`。
数据管线已经闭合，正式训练只等待结构 successor smoke 通过。
