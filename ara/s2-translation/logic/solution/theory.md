# S2 数学基础：TreeHeap 几何运算

> TreeHeap 的规范化数学操作：路由、外积、距离、能量、进制位类比。
> 所有运算均为纯数学，不依赖词性标签或外部解析器。

---

## 1. TreeHeap 架构

### 1.1 参数

| 符号 | 值 | 说明 |
|------|------|------|
| `V` | 32,000 | 词汇量 (SentencePiece BPE) |
| `d` | 128 | 向量维度 |
| `D` | 5 | 树深度 (层数) |
| `K` | 2 | 分支因子 (二叉树) |

### 1.2 节点结构

每层 `l` 有 `K^l` 个节点，共 `Σ_{l=0}^{D-1} 2^l = 31` 个共享节点。每节点存储一个 `d` 维向量。

```
Level 0: [n0]                    ← 根节点 (1个)
Level 1: [n1, n2]                ← 2个
Level 2: [n3,n4,n5,n6]           ← 4个
Level 3: [n7..n14]               ← 8个
Level 4: [n15..n30]              ← 16个
```

### 1.3 路由规则 (Path Routing)

Token ID `x` 在层级 `l` 的节点索引：

$$i_l(x) = \begin{cases} 0 & l = 0 \\ \lfloor x / \lfloor V / 2^l \rfloor \rfloor \bmod 2^l & l > 0 \end{cases}$$

**性质**：
- 路由是确定性的：每个 token ID 对应唯一条路径
- 路径不编码语法角色 — 仅反映 SP 词汇表顺序
- 路径相似度 = 两个 token 在 SP 词汇表中 ID 的相对接近程度

---

## 2. 向量空间

### 2.1 L0 嵌入

$$h_0(x) = \text{L0}[x] \in \mathbb{R}^{d}$$

| 属性 | 值 |
|------|------|
| L0 维度 | (32000, 128) |
| 向量范数 | ≈ 11.5 |
| 训练方式 | InfoNCE 损失 |

L0 编码了 token 的语义信息（"cat" 和 "fish" 可能接近，因为都是名词/动物）。

### 2.2 树路径向量

$$w(x) = \sum_{l=0}^{D-1} \text{tree\_nodes}[l][i_l(x)] \in \mathbb{R}^{d}$$

每一层的共享节点向量求和得到 128D 路径向量。这个向量编码了 token 在树空间中的"位置指纹"。

### 2.3 最终 128D 向量 (TreeHeap)

$$\bar{h}_0 = \frac{h_0}{\|h_0\|}, \quad \bar{w} = \frac{w}{\|w\|}$$

$$\text{CMul}(\bar{h}_0, \bar{w}) = \begin{pmatrix} \bar{h}_{0,L} \odot \bar{w}_L - \bar{h}_{0,R} \odot \bar{w}_R \\ \bar{h}_{0,L} \odot \bar{w}_R + \bar{h}_{0,R} \odot \bar{w}_L \end{pmatrix}$$

$$h(x) = \text{TreeMerge}(\text{CMul}(\bar{h}_0(x), \bar{w}(x))) \in \mathbb{R}^{d}$$

| 属性 | 值 |
|------|------|
| 范数 | ≈ 0.59 (归一化) |
| 均值 | ≈ 0.001 |
| 语义区分 | 弱 — 同 cluster 内所有 token 范数近似 |

**当前限制**：3 epoch 训练后，`h(x)` 向量的句法角色区分度低。所有 token 向量的范数和方向高度相似（cosine 0.98+），导致 K-means 角色聚类失败。

### 2.4 距离计算

**欧几里得距离**：
$$d_E(x, y) = \|h(x) - h(y)\|_2$$

**余弦相似度**：
$$\text{sim}(x, y) = \frac{h(x) \cdot h(y)}{\|h(x)\| \cdot \|h(y)\|}$$

**路径汉明距离**：
$$d_H(x, y) = \sum_{l=0}^{D-1} [i_l(x) \neq i_l(y)]$$

**路径共享节点数**：
$$S(x, y) = \sum_{l=0}^{D-1} [i_l(x) = i_l(y)]$$

---

## 3. 张量运算 (Structure Tensors)

### 3.1 外积 (Outer Product)

Token 向量 `h_i` 与路径嵌入 `p_i` 的张量组合：

$$T_i = h_i \otimes p_i \in \mathbb{R}^{d \times 31}$$

`p_i` 是 31 维 one-hot 向量（各层节点索引展开）。

### 3.2 求和外积 (Commutative)

$$T_{\text{sum}} = \sum_{i=1}^{n} h_i \otimes p_i$$

**性质**：交换律成立 → 排列不变。Token 集合相同 → `T_sum` 相同。

### 3.3 拼接外积 (Non-Commutative)

$$T_{\text{cat}} = \text{concat}(w_1 \cdot (h_1 \otimes p_1), \; w_2 \cdot (h_2 \otimes p_2), \; ..., \; w_n \cdot (h_n \otimes p_n))$$

**性质**：排列敏感（非交换）。排列改变 → 所有权重对应的 token 改变 → `T_cat` 不同。

**位置权重方案**：

| 方案 | 公式 | 特性 |
|------|------|------|
| 线性递减 | `w_i = 1/(i+1)` | 首位权重最大 |
| 中心增强 | `w_i = (n - |i - c|)/n` | 中间权重最大 (c = 中心) |
| 固定常数 | `w_i = 1` | 等价于无权重 (退化) |

### 3.4 n 阶张量积 (Full Tensor Product)

$$T_n = h_1 \otimes h_2 \otimes ... \otimes h_n \in \mathbb{R}^{d^n}$$

- n=3: 128^3 = 2,097,152 维
- n=4: 128^4 = 268,435,456 维

**限制**：维度随 `n` 指数增长。实际中 n>3 需要随机投影 (`T_proj = P(T_n)`) 或成对近似。

---

## 4. 能量函数

### 4.1 Frobenius 范数能量

$$E(T) = -\|T\|_F^2 = -\sum_{i,j} T_{ij}^2$$

- 推导自：更高的 `||T||` 意味着向量更"集中"、更有序
- 在排列实验中：不同的排列由于位置权重不同产生不同的能量
- 但当前向量质量下，能量差异太小 (≈0.001) 无法有效排序

### 4.2 邻接加权能量

$$E_{\text{adj}}(T) = -\sum_{i=1}^{n-1} w_i \cdot \|h_i \otimes h_{i+1}\|_F^2$$

- 只考虑相邻 token pair 的外积
- 不同排列 → 相邻对重组 → 不同能量
- 当前结果：正确 SVO 不是最低（金级 rank #4~22/24）

### 4.3 参考张量距离

$$E_{\text{ref}}(T) = \|T - T_{\text{ref}}\|_F$$

- `T_ref` 可以是预计算模板或所有排列平均值
- 已用于 D1/D2a 的跨语言结构预测中

---

## 5. 进制位类比 (Positional Algebra)

### 5.1 类比映射

| 十进制 | TreeHeap |
|--------|----------|
| 数字 `{1,2,3}` | Token 128D 向量 `{h_cat, h_look, h_fish}` |
| 百位/十位/个位 `{10², 10¹, 10⁰}` | 树层级 `{level0, level1, ..., level4}` |
| 数值 `3×100 + 2×10 + 1` | 路径位置 `h_token ⊗ p_path` |
| 比较 `321 ≠ 123` | 不同排列 → 不同外积拼接 |

### 5.2 当前未解决

- 树层级不天然编码"位权"语义（只是 SP 词汇表排序的分割）
- 需要将路径信息映射到可解释的位置嵌入，使"靠近根的token"自然成为"高位"

---

## 6. 当前数学限制

| 问题 | 原因 | 影响 |
|------|------|------|
| 128D 向量句法区分度低 | 3 epoch 训练仅编码语义相似度 | 能量最小化无法区分 SVO/OVS |
| 路径 = SP 词汇表 ID 分位 | 路径按词频排序，非按语法功能 | 路径距离 ≠ 语法关系 |
| 外积和不可交换但 Frobenius 不敏感 | norm 对拼接顺序不变 | 排列区分度仅来自位置权重 |
| n 阶张量积内存爆炸 | d^n 指数增长 | n>3 无法直接计算 |

## 7. 开放方向

1. **更多 epoch TreeHeap 训练** → 改善 128D 向量句法区分度
2. **学习路径嵌入** → 用数据训练真正的位置编码 (替代 one-hot)
3. **张量近似** → CP 分解 / Tensor Train 压缩 n 阶张量积
4. **几何 Hash** → 保留张量结构的同时压缩为固定维指纹
