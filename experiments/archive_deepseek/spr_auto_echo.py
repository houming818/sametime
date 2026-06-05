"""
SPR Auto Echo — IWSLT14 真实数据
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os

# 从 base.dataset 加载 IWSLT14
def load_data(vocab_size=100):
    """加载 IWSLT14 验证集英语，生成简单 token 表"""
    data_dir = "/data/datasets/iwslt14"
    split = "validation"
    cache_path = os.path.join(data_dir, f"iwslt14.{'valid' if split == 'validation' else split}.de-en")
    
    if not os.path.exists(cache_path):
        print(f"IWSLT14 not found at {cache_path}, using random sentences")
        # 生成一些假的 "句子"
        import random; random.seed(42)
        word2id = {f"word_{i}": i for i in range(50)}
        sentences = [random.sample(list(word2id.keys())[:20], random.randint(4, 10)) for _ in range(32)]
    else:
        with open(cache_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split("\t", 1)[1] for l in f if "\t" in l][:128]
        word2id = {}
        sentences = []
        for line in lines:
            words = line.split()
            for w in words:
                if w not in word2id:
                    word2id[w] = len(word2id)
            sentences.append(words)
    
    print(f"loaded {len(sentences)} sentences, {len(word2id)} unique words")
    
    # 取前 N 句，token id → 8维 embedding
    N = min(32, len(sentences))
    all_tokens = []
    for s in sentences[:N]:
        vec = torch.tensor([word2id[w] / len(word2id) for w in s], dtype=torch.float32)
        k = 1
        while k < len(vec):
            k *= 2
        padded = torch.zeros(k, 8)
        for i in range(min(len(vec), k)):
            v = vec[i]
            padded[i, 0] = v
            padded[i, 1:] = torch.sin(v * torch.arange(1, 8) * 0.5)
        all_tokens.append(padded)
    
    return all_tokens, sentences[:N], word2id

all_tokens, sentences, word2id = load_data()

# ── 模型 ──
depth, d = 4, 8
n_leaves = 2**depth

def route(tokens, weights, depth):
    if depth == 1:
        w = weights[0]
        probs_r = torch.sigmoid(tokens @ w)
        assign = torch.zeros(len(tokens), 2)
        assign[:, 0] = 1 - probs_r
        assign[:, 1] = probs_r
        return assign
    w = weights[0]
    probs_r = torch.sigmoid(tokens @ w)
    probs_l = 1 - probs_r
    child = route(tokens, weights[1:], depth-1)
    L = child.shape[1]
    assign = torch.zeros(len(tokens), L*2)
    assign[:, :L] = probs_l.unsqueeze(1) * child
    assign[:, L:] = probs_r.unsqueeze(1) * child
    return assign

# 使用第一句话训练（echo: 同句→同路由）
tokens = all_tokens[0]
print(f"\nfirst sentence: \"{' '.join(sentences[0][:8])}...\" ({len(tokens)} tokens padded)")

weights = nn.ParameterList([nn.Parameter(torch.randn(d)*0.05) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.03)

print("training auto-echo...")
for step in range(100):
    opt.zero_grad()
    assign = route(tokens, weights, depth)
    leaf_sum = assign.sum(dim=0) + 1e-8
    leaf_ctr = assign.T @ tokens / leaf_sum.unsqueeze(1)
    dists = torch.cdist(tokens, leaf_ctr)
    with torch.no_grad():
        best = dists.argmin(dim=1)
        target = torch.zeros_like(assign)
        target[range(len(tokens)), best] = 1.0
    loss = F.mse_loss(assign, target)
    loss.backward()
    opt.step()

    if step % 25 == 0:
        with torch.no_grad():
            a = route(tokens, [w.detach() for w in weights], depth)
            cur = a.argmax(dim=1)
            ls = a.sum(dim=0) + 1e-8
            lc = a.T @ tokens / ls.unsqueeze(1)
            d = torch.cdist(tokens, lc)
            b = d.argmin(dim=1)
            m = (cur == b).sum().item()
            af2 = route(tokens, [w.detach() for w in weights], depth)
            s = (cur == af2.argmax(dim=1)).sum().item()
        print(f"  step {step:3d}: loss={loss.item():.4f}  matched={m}/{len(tokens)}  deterministic={s}/{len(tokens)}")

# ── 跨句验证 ──
print(f"\n=== Cross-sentence test ===")
same_word_leaves = []
for i in range(min(5, len(all_tokens))):
    with torch.no_grad():
        a = route(all_tokens[i], [w.detach() for w in weights], depth)
        lf = a.argmax(dim=1)
        af2 = route(all_tokens[i], [w.detach() for w in weights], depth)
        s = (lf == af2.argmax(dim=1)).sum().item()
    sent_str = ' '.join(sentences[i][:5])
    print(f"  '{sent_str}...': deterministic={s}/{len(all_tokens[i])}")

print("AUTO ECHO TEST COMPLETE")

