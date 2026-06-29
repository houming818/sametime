# claims（中文读者版）

对应英文文件：`ara/s2-translation/logic/claims.md`  
所属主题：S2 Translation（折叠、图构建与翻译前结构）

> 说明：这是递归生成并人工组织过的中文读者版。它保留原文件的 ARA 用途、claim/evidence 边界和 reviewer 读法；命令、ID、路径、指标名通常保留英文，方便和原文及实验输出对照。

## 文件用途

本文件是 claim registry：每一行都应说明主张、状态、证据指针和可证伪条件。

## 阅读要点

- S2 的目标是从 S1 表示进入 fold stack、graph builder、probability container，再接近翻译结构。
- 历史 checkpoint 与过强 syntax-energy claim 已经被降级；当前重点是 graph assembly bottleneck 和概率容器。
- 任何 WMT 或 Transformer 替代 claim 都必须等待新证据，不允许靠旧 checkpoint 背书。

## Reviewer 检查方式

- 不要只看 claim 文本，要同时检查 Status、Evidence、Falsification 三列。
- 如果证据不完整，claim 只能是 supported/open，不能升级为 verified。

## 与英文原文的关系

- 英文原文保留为 canonical source，中文版本用于中文读者快速理解和审阅。
- 涉及实验数字、命令、文件路径、claim ID 时，应以英文原文和 evidence 目录中的机器输出为最终校验来源。
- 如果后续英文原文有 claim 状态更新，本文件也应同步更新。
