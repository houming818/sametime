# SPR 实验记录

## 核心结论 (截至 2026-06-03)

### 已经确认的工作

1. **独立 128D path 向量** → ms=97% (词义对齐), cos=0.14 (路径分离), gold=6.4% (单义词无损)
   - 3191 条 path, 408K 参数, 无共享

2. **BLI P@1** → 13.4% (20K ECDICT 训练 + LaBSE 预训练)
   - 128D 空间对齐容量饱和在 ~5K 词
   - 训练规模翻倍 (10K, 20K, 50K) 无效
   - 维度翻倍 (256D) 无效 (冷启动零)

3. **堆树共享方案全部失败**:
   - AttnPaths (31D softmax over 31 nodes): ms=33%, cos=0.99 (均匀软注意力)
   - DiffTree (有序邻域掩码): 保住了节点正交性 (L4cos=-0.07→0.02) 但注意力仍然坍缩
   - 温度参数: 未测试 (可能解决 softmax 均匀问题)
   - 有序梯度 (差分树): 未充分测试

4. **ECDICT 词典** → 98K EN→ZH 对, 词级 oracle precision=45.2%
   - 字典覆盖率高但多义项选择需要 context

5. **CMul 可分化** → 证明: 同一 base 乘两条独立 path 可以产生分离输出 (cos=0.53, 非 1.0)

### 瓶颈

- **31 共享节点**: 信息传递通路未通
- **Word-level BLI**: 128D 容量天花板 ~13%
- **Context routing**: 缺少简单有效的 context selector

### 下一步候选方向

1. AttnPaths + 低温度 (T=0.1) — 让 softmax 尖锐, 注意力做选择
2. 独立 path 做翻译 BLEU — 先验证功能价值
3. 树的有序性 / 模运算设计 — 理论重构
4. 共现表多头 — 简单 context selector
