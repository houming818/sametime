# D10：清洗全集上的结构协议训练管道

日期：2026-08-29

Claim：`S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10`

状态：端到端 smoke 已通过；正式长训练运行中。

## 1. 为什么不是把 D09 直接放大

D09 使用原始 WMT 的 20 万行验证相邻递归 subheap ownership 能继续学习。它不是
STONE-2 的数据管道，也没有执行自然文本 Pretrain。清洗完成后，正式训练必须读取已经
固化并校验哈希的 release，而不是重新把原始 1417 万对称作“全量”。

本次固定顺序为：

```text
NioText-ZH-Integrity-2985K-v1
-> Stage A：中文自然文本 next-span Pretrain
-> NioClean-ZHEN-S098-7M-v2
-> Stage B：中英双向翻译 Task Train
-> 原始 WMT held-out + TreeHeap 结构干预 Proof
-> checkpoint/reload/CLI 审计
```

中文 QA 不混入本轮。翻译 checkpoint 通过后，再以独立 Claim 续训 QA，避免无法定位
退化来自 Pretrain、翻译还是 QA。

## 2. 数据合同

| Stage | release | 物理行数 | SHA-256 |
|---|---|---:|---|
| A | `NioText-ZH-Integrity-2985K-v1` | 2,972,976 | `04a90d88b51755561645d0fec962fc8bd5e642d099423348417de9318e22c94e` |
| B | `NioClean-ZHEN-S098-7M-v2` | 7,304,358 | `299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8` |

Stage A 对每个满足长度的文档确定性抽取一个窗口，source 宽度在 `8/16/32` 中按文档
哈希选择，target 固定为随后 32 pieces。这样每个文档最多贡献一次，预计约 9500 万
target pieces，不循环小型内存表。

Stage B 每个清洗语对恰好选择一个确定性翻译方向。source 截断到 32 内容 pieces，target
截断到 63 内容 pieces 后添加 EOS；必须记录截断比例。外部 valid/test 从未清洗的原始
WMT massive 固定哈希分区产生，其语对从 Stage B 训练流中排除。

## 3. 模型合同

```text
固定观察底座：C12 formal seed10101 READ checkpoint
初始化协议：D09 best step 25000
通信：底座已有 XOR Butterfly + 递归 FOLD/H_state
输入协议：D08R1/D09 adjacent recursive subheap ownership
协议容量：32 slots
递归压力：depth 5/6/7 均衡暴露
训练参数：compressor、protocol FOLD、reconstructor READ/K_up、bounded gain、语言 Decoder
冻结参数：C12 observation/source model
loss：目标 token cross entropy
```

D09 为了隔离机制冻结了语言 Decoder；D10 是产品训练，因此解冻 reconstructor 的
embedding/query/cell/output/branch/depth embedding。观察底座仍冻结并在每个阶段核对
SHA-256。该变化属于新 Claim，不能反写成 D09 的结果。

## 4. 流式执行与恢复

- 不把 297 万文档或 730 万语对装入列表；按物理行流式读取；
- `checkpoint_latest.pt` 保存 stage、下一物理行、step、optimizer、最佳指标和 RNG 合同；
- `checkpoint_best.pt` 按固定 external valid mean NLL 选择；
- Stage B 只从 Stage A 的 best checkpoint 开始；
- 每次 wake 保存 NLL/PPL、生成、重复率、route/slot 指标、数据 cursor、处理 token、
  wall time 和 state SHA-256；
- 断点恢复必须重现同一物理行和深度序列。

长任务保持 RTX 3090 的 270W 限制。出现 OOM、CUDA 错误、NaN/Inf、数据哈希变化、
冻结底座哈希变化、重载不一致、固定输出坍缩，或达到最小训练步数后连续三个 wake 没有
至少 `0.005` NLL 改善时停止。

## 5. Smoke 门

端到端 smoke 使用每阶段 4096 物理行，必须满足：

1. 两个 release、tokenizer、source 和 warm-start 身份一致；
2. Stage A best 可被 Stage B 严格加载；
3. 两阶段 loss/gradient 有限，全部预期参数至少一次获得梯度；
4. source SHA-256 不变，语言 Decoder SHA-256 发生变化；
5. subheap owner/argmax coverage 为 `1.0`，route overlap 小于 `0.05`；
6. latest/best checkpoint 可以重载，固定评估 NLL 差小于 `1e-9`；
7. 不要求 4096 行产生产品 BLEU，smoke 只授权执行。

## 6. 正式观察门

Stage A：

- best held-out next-span mean NLL 相对初始化改善至少 `0.10`；
- wrong-sample shuffle 与 zero state 至少两个深度各增加 `0.10` NLL；
- 非空生成率为 `1.0`，严重相邻重复率不高于 `0.25`。

Stage B：

- external WMT best mean NLL 相对 Stage B 初始化改善至少 `0.10`；
- 固定生成探针 BLEU4 中位数相对 Stage B 初始化提高至少 `0.50`；
- shuffle/zero 因果门、ownership 覆盖、route overlap 和 reload 门全部通过；
- 非空率 `1.0`，相邻重复率不高于 `0.10`。

这些门支持“清洗全集管道继续生长”，不等同于商业翻译质量。正式报告必须同时给出
错误样例，固定探针 BLEU 不与公开模型的标准测试 BLEU混用。

## 7. 资源预计

单次完整 release 暴露预计：

```text
Stage A：约 8--14 GPU 小时
Stage B：约 28--45 GPU 小时
Proof/重载：约 1 GPU 小时
总计：约 2--3 天墙钟时间
```

Evidence：

```text
ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10/
```

## 8. 执行记录

- taskd `#339`：首次 smoke 在模型装载阶段失败。原因是历史 C12 loader 仍读取
  `args.wmt_data`，D10 只声明了新的 `eval_wmt_data`；尚未进入训练。
- taskd `#340`：补齐只用于冻结底座加载的兼容参数后，4096 行双阶段 smoke 通过。
  Stage A best NLL 为 `6.8238`，Stage B best NLL 为 `5.5852`，Stage B 固定探针
  BLEU4 中位数为 `2.8433`。这些数值只证明管道可执行，不作为产品质量结论。
- taskd `#341`：增加分阶段 contract 和 A→B 硬门后，从 smoke checkpoint 重载复验
  通过；若 Stage A 未达到 `stage_supported`，Stage B 不会启动。
- taskd `#342`：2026-08-29 启动正式任务。两份 release 的全文件 SHA-256 校验通过，
  当前正在 Stage A。正式结果必须等待任务完成或触发预注册停止门后再登记。
