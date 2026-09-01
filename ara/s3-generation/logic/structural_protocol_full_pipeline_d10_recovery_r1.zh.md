# D10-R1：CUDA allocator 异常后的可审计恢复

Claim：`S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10-RECOVERY-R1`

状态：预注册，等待 canary。

## 1. 问题

D10 Stage B 在 step `299000` 触发 PyTorch `CUDACachingAllocator` 内部断言。
最近的持久 checkpoint 位于 step `275000`、数据 cursor `4,400,986`。原任务在退出前
没有 OOM、NaN/Inf 或 NVIDIA Xid 证据，因此本轮不修改模型、数据、损失函数或优化器，
只验证能否从持久边界恢复。

## 2. 恢复不等于重训

恢复必须严格加载：

- `trainable_state_dict`；
- AdamW `optimizer_state_dict`；
- step 与数据 cursor；
- checkpoint 中保存的 best NLL、best step 与 stale wake；
- 与原 D10 相同的 source checkpoint、Stage A best、tokenizer、数据 release 和 seeds。

原 evidence 保持只读。恢复运行复制 latest/best 到新的 evidence 目录，避免覆盖事故现场。

## 3. Canary

从 step `275000` 运行到 `277000`，每 500 step 保存并评估一次。

通过条件：

1. latest/best 文件 SHA-256、claim、stage、step、cursor 可记录；
2. checkpoint 内全部浮点张量有限；
3. 2,000 step 内没有 OOM、CUDA allocator 断言、NaN/Inf 或 GPU Xid；
4. 训练结束生成 step `277000` 的可重载 latest checkpoint；
5. 新 checkpoint 的训练状态哈希与文件哈希可复算；
6. native 结构统计保持有限，固定评估 mean NLL 相对 step `275000` 不恶化超过 `0.30`。

第 6 条只用于拦截明显损坏，不把 2,000 step 波动解释为模型能力结论。

## 4. 正式恢复

Canary 全部通过后，从 canary latest 继续完成剩余 Stage B。每个进程最多再训练
`25,000` step，然后退出并从持久 checkpoint 启动下一段。这样不改变训练的样本顺序、
梯度或优化器状态，只缩短单个 CUDA allocator 进程的生命周期。

每段必须保留：起止 step/cursor、checkpoint SHA-256、wake、trace、GPU 状态和退出码。
任一段出现 OOM、CUDA/GPU 错误、非有限值、checkpoint 不可重载或数据/hash 合同变化，
立即停止后续分段。

## 5. 边界

- step `275000--299000` 的旧 trace 不是可恢复参数，必须从 275000 重新计算；
- 分段恢复是运行时可靠性措施，不是新的 TreeHeap 算法收益；
- 只有完整跑完 Stage B 及最终 proof/reload，D10 主 Claim 才能重新判定。
