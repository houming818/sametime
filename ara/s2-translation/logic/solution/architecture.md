# S2 TreeHeap Architecture Diagram

```
                    ┌──────────────────────┐
                    │    Token ID (x)       │
                    │   SP BPE 词汇表索引    │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌──────────┐  ┌──────────┐
        │ L0(x)   │  │ Path(x)  │  │ Path(x)  │
        │ 128D    │  │ Tree sum │  │ 31D one- │
        │ token   │  │ 128D     │  │ hot      │
        │ embed   │  │          │  │          │
        └────┬────┘  └────┬─────┘  └────┬─────┘
             │            │            │
             │    CMul(L0, Path)       │
             ▼            ▼            │
        ┌─────────────────────────┐    │
        │    CMul(L0̄, Path̄)       │    │
        │    复平面乘法            │    │
        │    h(x) = TreeMerge()   │    │
        │    128D 归一化: ||≈0.59 │    │
        └───────────┬─────────────┘    │
                    │                  │
                    ▼                  ▼
              ┌──────────┐     ┌───────────────┐
              │ 语义向量  │     │  路径嵌入      │
              │ h(x)     │     │  p(x)          │
              │ 128D     │     │  31D one-hot   │
              └────┬─────┘     └───────┬───────┘
                   │                   │
                   └──── 外积 T = h⊗p ──┘
                             │
                             ▼
                    ┌────────────────┐
                    │ 句子张量        │
                    │ 排列→能量→排序  │
                    └────────────────┘

    核心运算:
    CMul(a,b) = [a_L·b_L - a_R·b_R, a_L·b_R + a_R·b_L]
    外积: T = Σ w_i · (h_i ⊗ p_i)
    能量: E(T) = -||T||²
```

## Token Path Example

```yaml
cat  (id=6361):  [0, 0, 0, 1, 3]
look (id=806):   [0, 0, 0, 0, 0]
fish (id=8897):  [0, 0, 1, 2, 4]
the  (id=12):    [0, 0, 0, 0, 0]   # same as look!

note: "the" and "look" share the identical 5-level path —
      they are routed to the same 5 node indices at every level.
      This proves paths encode SP vocabulary proximity,
      not syntactic role.
```
