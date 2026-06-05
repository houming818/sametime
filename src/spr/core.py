"""
SPR Core — 堆树 × 世界嵌入 × 可学习路径
=============================================
数据结构:
  TreeNodes      — 31 共享节点, 加法/乘积两种路径组合
  HeapWorld      — L0 embedding × tree path → 世界坐标
  Paths          — 每条词义的独立可学习 128D 向量
  AnchorIndex    — 锚点对索引 + InfoNCE 矩阵
  Trainer        — InfoNCE 训练循环 (单义 + 多义)
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

device_lut = {'cuda': 'cuda:0', 'cpu': 'cpu'}
device = device_lut.get('cuda' if torch.cuda.is_available() else 'cpu', 'cpu')

# ═══════════════════════════════════════════════
# TreeNodes — 树节点
# ═══════════════════════════════════════════════
class TreeNodes(nn.Module):
    """
    堆树节点: depth 层, 每层 2^L 个 128D 节点, 总共 Σ 2^L 个共享节点.

    路由:  token_id // (vocab // 2^L) % 2^L   (硬模基)
    组合模式:
      "add"  → path = Σ node[L][idx]          (加法, root 压倒 child)
      "mul"  → path = Π CMul(node[L][idx])    (乘积, 每层独立旋转)

    初始化模式:
      "rand"  → N(0, 0.1)
      "unit"  → 2^L 次单位根 (纯旋转, 每对 a_r² + a_i² = 1)
      "zero"  → root rand, children zeros
    """
    def __init__(self, dim: int = 128, depth: int = 5,
                 init: str = "unit", combine: str = "mul",
                 vocab: int = 16000, branching: int = 2):
        super().__init__()
        self.dim = dim; self.depth = depth; self.vocab = vocab
        self.combine = combine; self.branching = branching
        node_counts = [branching ** l for l in range(depth)]
        total = sum(node_counts)
        self.node_counts = node_counts
        self.total_nodes = total
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, dim) for n in node_counts
        ])
        self._init_weights(init)

    def _init_weights(self, init: str):
        for l in range(self.depth):
            n = self.branching ** l
            if init == "unit":
                w = torch.zeros(n, self.dim)
                for k in range(n):
                    angle = 2 * math.pi * k / n
                    for p in range(self.dim // 2):
                        w[k, 2 * p]     = math.cos(angle)
                        w[k, 2 * p + 1] = math.sin(angle)
                self.embeddings[l].weight.data = w
            elif init == "zero":
                if l == 0:
                    rv = torch.randn(1, self.dim)
                    self.embeddings[0].weight.data = rv / rv.norm()
                else:
                    nn.init.zeros_(self.embeddings[l].weight)
            else:  # "rand"
                nn.init.normal_(self.embeddings[l].weight, 0, 0.1)

    def route(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """token_ids [B] → 每层的节点索引 [B]"""
        indices = []
        K = self.branching
        for l in range(self.depth):
            if l == 0:
                idx = torch.zeros_like(token_ids)
            else:
                stride = self.vocab // (K ** l) if self.vocab >= K ** l else 1
                idx = torch.clamp(token_ids // stride, 0, (K ** l) - 1)
            indices.append(idx)
        return indices

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """→ path 向量 [B, dim]"""
        indices = self.route(token_ids)
        if self.combine == "add":
            w = torch.zeros(len(token_ids), self.dim, device=token_ids.device)
            for l in range(self.depth):
                w = w + self.embeddings[l](indices[l])
            return w
        else:  # "mul" — recursive CMul
            result = F.normalize(self.embeddings[0](indices[0]), dim=-1)
            for l in range(1, self.depth):
                nxt = F.normalize(self.embeddings[l](indices[l]), dim=-1)
                result = cmul(result, nxt)
            return result

    def soft_path(self, weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Context routing: weights[l] 是 [B, 2^L] 的 softmax 权重
        → 加权求和每个节点的向量 → soft path
        """
        dev = weights[0].device
        if self.combine == "add":
            w = torch.zeros(weights[0].shape[0], self.dim, device=dev)
            for l in range(self.depth):
                nodes = self.embeddings[l].weight.to(dev)  # [2^L, dim]
                w = w + weights[l].to(dev) @ nodes
            return w
        else:
            result = weights[0].to(dev) @ F.normalize(self.embeddings[0].weight.to(dev), dim=-1)
            for l in range(1, self.depth):
                nodes = F.normalize(self.embeddings[l].weight.to(dev), dim=-1)
                nxt = weights[l].to(dev) @ nodes
                result = cmul(result, nxt)
            return result

    def __repr__(self):
        return f"TreeNodes(dim={self.dim}, depth={self.depth}, nodes={self.total_nodes}, combine={self.combine})"

    # ── 有序邻域 ──
    def path_distance(self, path_a: List[int], path_b: List[int]) -> int:
        """
        两条路径的有序距离 (层级加权).
        L1 分歧 > L2 > L3...
        """
        d = 0
        for l in range(self.depth):
            w = 2 ** (self.depth - l)
            d += w * abs(path_a[l] - path_b[l])
        return d

    def nodes_near_path(self, path: List[int], radius: int = 2) -> List[int]:
        """
        返回离目标路径 radius 步内的所有节点全局索引 [int].
        节点全局索引: 前面各层节点数之和 + 该层内偏移
        """
        nearby = []
        offset = 0
        for l in range(self.depth):
            n = self.node_counts[l]
            center = path[l]
            for k in range(max(0, center - radius), min(n, center + radius + 1)):
                nearby.append(offset + k)
            offset += n
        return nearby

    def gradient_mask(self, target_paths: List[List[int]],
                      radius: int = 2) -> torch.Tensor:
        """
        返回 [total_nodes] 的 bool mask.
        True = 该节点收到梯度; False = 梯度清零.
        target_paths: 批次中所有 target path 的 union.
        """
        mask = torch.zeros(self.total_nodes, dtype=torch.bool)
        for p in target_paths:
            for idx in self.nodes_near_path(p, radius):
                mask[idx] = True
        return mask

    def apply_mask(self, target_paths: List[List[int]], radius: int = 2):
        """
        对树节点的梯度应用邻域掩码——掩码外的节点梯度清零.
        在 loss.backward() 之后, optimizer.step() 之前调用.
        """
        mask = self.gradient_mask(target_paths, radius)
        offset = 0
        for l in range(self.depth):
            n = self.node_counts[l]
            l_mask = mask[offset:offset + n]
            for k in range(n):
                if not l_mask[k]:
                    grad = self.embeddings[l].weight.grad
                    if grad is not None:
                        grad[k] = 0
            offset += n


# ═══════════════════════════════════════════════
# HeapWorld — 世界嵌入 (L0 × tree)
# ═══════════════════════════════════════════════
class HeapWorld(nn.Module):
    """
    L0 embedding × tree path → 世界坐标.
    heap_world(tok) = CMul(L0[tok], tree_path(tok)).
    """
    def __init__(self, vocab: int, dim: int, tree: TreeNodes):
        super().__init__()
        self.vocab = vocab; self.dim = dim
        self.embedding = nn.Embedding(vocab, dim)   # L0
        self.tree = tree
        self.merge = nn.Linear(dim, dim)
        # 初始化为恒等 + 零偏置
        nn.init.eye_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)

    def forward(self, token_ids: torch.Tensor, use_path: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids [B, L] → heap_world [B, L, dim]
        use_path: 如果不为 None, 用此 path 替代 tree_path(token_ids)
        """
        t = F.normalize(self.embedding(token_ids), dim=-1)
        if use_path is not None:
            w = F.normalize(use_path, dim=-1)
        else:
            w = F.normalize(self.tree(token_ids.view(-1)).view(*token_ids.shape, -1), dim=-1)
        return self.merge(cmul(t, w))

    def mean(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [L] → mean heap_world over all BPE tokens → [dim]"""
        return self.forward(token_ids.unsqueeze(0)).squeeze(0).mean(dim=0)


# ═══════════════════════════════════════════════
# Paths — 多义路径集
# ═══════════════════════════════════════════════
class Paths(nn.Module):
    """
    每条词义一个 learnable 128D 向量.
    path[word][sense_k] ← learnable parameter.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.path_dict = nn.ParameterDict()
        self.meta: Dict[str, Tuple[str, str, int]] = {}
        # Device tracking: register a dummy buffer
        self.register_buffer('_device_tracker', torch.zeros(1))

    def add(self, en_word: str, zh_word: str, anchor_idx: int, sense_k: int):
        """添加一条 path"""
        name = f"{en_word}_{sense_k}"
        if name not in self.path_dict:
            dev = self._device_tracker.device
            self.path_dict[name] = nn.Parameter(torch.randn(self.dim, device=dev) * 0.001)
        self.meta[name] = (en_word, zh_word, anchor_idx)
        return name

    def get(self, name: str) -> torch.Tensor:
        return self.path_dict[name]

    def world_pos(self, name: str, l0_embedding: nn.Embedding,
                  en_token_id: int) -> torch.Tensor:
        """CMul(L0[en_word], path) → 世界位置"""
        base = F.normalize(l0_embedding.weight[en_token_id], dim=-1)
        p = self.path_dict[name].to(base.device)
        return cmul(base.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

    def pairwise_cos(self):
        """所有同一 EN 词的各 path 之间的 pairwise cos"""
        groups = defaultdict(list)
        for name in self.path_dict:
            en = self.meta[name][0]
            groups[en].append(name)
        cosines = []
        for names in groups.values():
            if len(names) < 2: continue
            p0 = F.normalize(self.path_dict[names[0]], dim=-1)
            p1 = F.normalize(self.path_dict[names[1]], dim=-1)
            cosines.append((p0 * p1).sum().item())
        return cosines

    @property
    def num_paths(self):
        return len(self.path_dict)

    def names(self):
        return list(self.path_dict.keys())

    def __repr__(self):
        return f"Paths(word_groups={len(set(m[0] for m in self.meta.values()))}, total={len(self.path_dict)})"


# ═══════════════════════════════════════════════
# AttnPaths — 注意力路径集 (31D 权重 → t_nodes)
# ═══════════════════════════════════════════════
class AttnPaths(nn.Module):
    """
    每条 sense 存 31D softmax 权重 → soft_path = Σ attn[i] × t_node[i]
    → 通过同一份 t_nodes，概念自然共享。

    world_pos = CMul(L0[tok], soft_path)
    """
    def __init__(self, tree: TreeNodes):
        super().__init__()
        self.tree = tree
        self.n_nodes = tree.total_nodes  # 31
        self.attn_dict = nn.ParameterDict()
        self.meta: Dict[str, Tuple[str, str, int]] = {}
        self.register_buffer('_device_tracker', torch.zeros(1))

    def add(self, en_word: str, zh_word: str, anchor_idx: int, sense_k: int):
        name = f"{en_word}_{sense_k}"
        if name not in self.attn_dict:
            dev = self._device_tracker.device
            self.attn_dict[name] = nn.Parameter(torch.randn(self.n_nodes, device=dev) * 0.01)
        self.meta[name] = (en_word, zh_word, anchor_idx)
        return name

    def soft_path(self, name: str) -> torch.Tensor:
        """31D → softmax → weighted sum of t_nodes → 128D"""
        logits = self.attn_dict[name]  # [31]
        weights = []
        dev = logits.device
        start = 0
        for n in self.tree.node_counts:
            end = start + n
            w = F.softmax(logits[start:end], dim=-1).unsqueeze(0).to(dev)  # [1, n]
            weights.append(w)
            start = end
        return self.tree.soft_path(weights).squeeze(0)  # [dim]

    def world_pos(self, name: str, l0_embedding: nn.Embedding,
                  en_token_id: int) -> torch.Tensor:
        base = F.normalize(l0_embedding.weight[en_token_id], dim=-1)
        sp = self.soft_path(name).to(base.device)
        return cmul(base.unsqueeze(0), sp.unsqueeze(0)).squeeze(0)

    def get_attn_vector(self, name: str) -> torch.Tensor:
        """返回归一化后的 31 维注意力向量 [用于概念重叠分析]"""
        logits = self.attn_dict[name]
        weights = []
        start = 0
        for n in self.tree.node_counts:
            end = start + n
            weights.append(F.softmax(logits[start:end], dim=-1))
            start = end
        return torch.cat(weights)  # [31]

    def pairwise_cos(self):
        """同一 EN 词的各 soft_path 之间的 pairwise cos"""
        from collections import defaultdict
        groups = defaultdict(list)
        for name in self.attn_dict:
            en = self.meta[name][0]
            groups[en].append(name)
        cosines = []
        for names in groups.values():
            if len(names) < 2: continue
            sp0 = F.normalize(self.soft_path(names[0]), dim=-1)
            sp1 = F.normalize(self.soft_path(names[1]), dim=-1)
            cosines.append((sp0 * sp1).sum().item())
        return cosines

    def concept_overlap(self, words: List[str]) -> Dict[str, float]:
        """词对之间的节点激活模式 cos → 概念共享度"""
        attns = {}
        for name in self.attn_dict:
            en = self.meta[name][0]
            if en in words:
                attns.setdefault(en, []).append(self.get_attn_vector(name).detach())
        overlaps = {}
        wlist = list(attns.keys())
        for i in range(len(wlist)):
            for j in range(i + 1, len(wlist)):
                a = torch.stack(attns[wlist[i]]).mean(dim=0)
                b = torch.stack(attns[wlist[j]]).mean(dim=0)
                overlaps[(wlist[i], wlist[j])] = F.cosine_similarity(
                    a.unsqueeze(0), b.unsqueeze(0)).item()
        return overlaps

    @property
    def num_paths(self):
        return len(self.attn_dict)

    def names(self):
        return list(self.attn_dict.keys())

    def __repr__(self):
        return f"AttnPaths(words={len(set(m[0] for m in self.meta.values()))}, paths={len(self.attn_dict)}, nodes={self.n_nodes})"


# ═══════════════════════════════════════════════
# AnchorIndex — 锚点对索引
# ═══════════════════════════════════════════════
class AnchorIndex:
    """
    锚点对管理:
      anchor_list[i] = (en_word, zh_word, en_ids, zh_ids)
      zh_world_mat[i] = heap_world(zh_ids).mean()  ← 预计算

    查询:
      find(en_word) → [(anchor_index, zh_word), ...]
    单义: 一个 EN word 只有一个 anchor_index
    多义: 一个 EN word 有多个 anchor_index
    """
    def __init__(self):
        self.entries: List[Tuple[str, str, List[int], List[int]]] = []
        self.zh_world: Optional[torch.Tensor] = None
        self._en_to_ai: Dict[str, List[int]] = defaultdict(list)

    def add(self, en_word: str, zh_word: str,
            en_ids: List[int], zh_ids: List[int]):
        ai = len(self.entries)
        self.entries.append((en_word, zh_word, en_ids, zh_ids))
        self._en_to_ai[en_word].append(ai)

    def build_zh_world(self, heap_world_fn) -> torch.Tensor:
        """调用 heap_world_fn 预计算所有 ZH 世界坐标"""
        N = len(self.entries)
        zh_mat = torch.zeros(N, heap_world_fn(self.entries[0][3]).shape[-1])
        with torch.no_grad():
            for ai, (_, _, _, zi) in enumerate(self.entries):
                zi_t = torch.tensor(zi, device=device)
                zh_mat[ai] = F.normalize(
                    heap_world_fn(zi_t).mean(dim=0), dim=-1
                )
        self.zh_world = zh_mat.to(device)
        return self.zh_world

    def find(self, en_word: str) -> List[int]:
        return self._en_to_ai.get(en_word, [])

    def single_sense_indices(self, multi_en_words: set) -> List[int]:
        """返回不是多义词的所有锚点索引"""
        return [ai for ai, (en, _, _, _) in enumerate(self.entries)
                if en not in multi_en_words and len(self.find(en)) == 1]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


# ═══════════════════════════════════════════════
# Trainer — InfoNCE 训练
# ═══════════════════════════════════════════════
class Trainer:
    """
    InfoNCE 训练循环:
      单义: heap_world(en_ids) → zh_world[ai]
      多义: CMul(L0[en_tok], path_k) → zh_world[ai]

    参数:
      heap_world: HeapWorld 实例 (L0 + tree + merge)
      paths:      Paths 实例 (可选, 多义训练需要)
      anchors:    AnchorIndex 实例
      tau:        InfoNCE 温度
    """
    def __init__(self, heap_world: HeapWorld,
                 anchors: AnchorIndex,
                 paths: Optional[Paths] = None,
                 tau: float = 0.07,
                 lr: float = 0.003):
        self.hw = heap_world
        self.anchors = anchors
        self.paths = paths
        self.tau = tau
        self.lr = lr
        self.optimizer = None

    def setup_optimizer(self, trainable_params: List, lr_mult: float = 1.0):
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr * lr_mult)
        return self.optimizer

    def infonce_loss(self, query: torch.Tensor,
                     target_idx: torch.Tensor) -> torch.Tensor:
        """
        query [B, dim] → similarity with all ZH anchors → CE
        """
        logits = (query @ self.anchors.zh_world.T) / self.tau
        return F.cross_entropy(logits, target_idx)

    def train_step_single(self, anchor_indices: List[int]):
        """单义锚点 InfoNCE — 直接 heap_world"""
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device); n = 0
        for ai in anchor_indices:
            en_ids = torch.tensor(self.anchors[ai][2], device=device)
            hw = F.normalize(self.hw.mean(en_ids), dim=-1)
            loss += self.infonce_loss(hw.unsqueeze(0),
                                       torch.tensor([ai], device=device))
            n += 1
        if n > 0:
            (loss / n).backward()
            self.optimizer.step()
        return loss.item() / max(n, 1)

    def train_step_multi(self, path_batch: List[Tuple[str, int]]):
        """多义路径 InfoNCE"""
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device); n = 0
        for pname, ai in path_batch:
            en_word = self.paths.meta[pname][0]
            en_id = self.anchors[ai][2][0]  # first BPE token
            cw = self.paths.world_pos(pname, self.hw.embedding, en_id)
            loss += self.infonce_loss(F.normalize(cw, dim=-1).unsqueeze(0),
                                       torch.tensor([ai], device=device))
            n += 1
        if n > 0:
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.paths.parameters()), 1.0
            )
            self.optimizer.step()
        return loss.item() / max(n, 1)

    def train_epoch(self, single_indices: List[int],
                    path_names: List[str],
                    single_ratio: int = 4):
        """单 epoch: 混合 single-sense + multi-sense batch"""
        import random
        random.shuffle(single_indices)
        B = 32
        for bi in range(0, len(single_indices), B):
            s_batch = single_indices[bi:bi + B]
            loss_s = self.train_step_single(s_batch)

            m_keys = random.sample(path_names,
                                   min(max(1, len(path_names)), 4))
            m_batch = [(n, self.paths.meta[n][2]) for n in m_keys]
            loss_m = self.train_step_multi(m_batch)

        return loss_s, loss_m


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """复数乘法: 64 个 2D 旋转块的逐元素乘积"""
    a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
    aL, aR = a_n[..., :a_n.shape[-1] // 2], a_n[..., a_n.shape[-1] // 2:]
    bL, bR = b_n[..., :b_n.shape[-1] // 2], b_n[..., b_n.shape[-1] // 2:]
    return torch.cat([aL * bL - aR * bR, aL * bR + aR * bL], -1)


# ═══════════════════════════════════════════════
# DiffTree — 树上差分梯度
# ═══════════════════════════════════════════════
class DiffTree:
    """
    树上差分: 两条路径在 LCA (最低公共祖先) 处分叉。
    叶子节点收完整梯度, LCA 及以上节点取消冲突分量。

    算法:
      对同一 EN 词的每对 path (path_A, path_B):
        LCA 深度 = 两条路径最先分歧的层 - 1
        LCA 以上的节点 (共享) → 梯度抵消
        LCA 以下的节点 (独有) → 保留梯度
    """
    def __init__(self, tree: 'TreeNodes'):
        self.tree = tree

    def find_lca_depth(self, path_a: List[int], path_b: List[int]) -> int:
        for l in range(self.tree.depth):
            if path_a[l] != path_b[l]:
                return l - 1  # 上一级是最后共享的层级
        return self.tree.depth - 1  # 完全相同的路径

    def apply(self, path_pairs: List[Tuple[List[int], List[int]]],
              scale: float = 1.0):
        """
        loss.backward() 后, optimizer.step() 前调用。
        对共享节点 (LCA 及以上) 的梯度按冲突程度缩放。
        scale=0.0 → 完全取消共享节点梯度; scale=0.5 → 减半。
        """
        if not path_pairs:
            return

        # 统计每个节点被多少对路径共享
        conflict = [torch.zeros(n, device=self.tree.embeddings[0].weight.device)
                    for n in self.tree.node_counts]

        for pa, pb in path_pairs:
            lca = self.find_lca_depth(pa, pb)
            for l in range(lca + 1):  # LCA 及以上 (共享层级)
                if l < len(pa):
                    conflict[l][pa[l]] += 1
                if l < len(pb):
                    conflict[l][pb[l]] += 1

        # 对每个共享节点, 梯度按冲突度缩放
        offset = 0
        for l in range(self.tree.depth):
            n = self.tree.node_counts[l]
            for k in range(n):
                if conflict[l][k] >= 1:
                    grad = self.tree.embeddings[l].weight.grad
                    if grad is not None:
                        grad[k] *= scale / max(conflict[l][k], 1.0)
            offset += n

    def get_path_for_token(self, token_id: int) -> List[int]:
        """获取 token_id 在 tree 上的硬路径"""
        tid_t = torch.tensor([token_id], device=next(self.tree.parameters()).device)
        indices = self.tree.route(tid_t)
        return [indices[l].item() for l in range(self.tree.depth)]
