"""
SPR-008 Twin Tree v2 — Trainable Decoder
Encoder: cyclic shift (0 params, deterministic)
Decoder: per-depth W_split (d×d, layer-shared)
Training: source → encode → decode → MSE(target_tree)
"""
import torch, torch.nn as nn

torch.manual_seed(42)
d = 16
depth = 2  # 3 tokens → depth 2
n_leaves = 1 << depth

# ── Mock embeddings: 10 tokens ──
E = torch.randn(10, d) * 0.5

# Sample training data
src_sents = [
    [0, 1, 2],  # 我 打 你
    [3, 4, 5],  # 他 吃 鱼
    [6, 0, 7],  # 她 我 看
    [1, 3, 8],  # 打 他 球
]
tgt_sents = [
    [5, 6, 7],  # I hit you
    [8, 9, 2],  # He eat fish
    [7, 5, 0],  # She I see
    [9, 8, 6],  # Hit him ball
]

def sign_alt(x):
    mask = torch.tensor([1., -1.] * (d//2 + 1))[:d]
    return x * mask

# ── Encoder (deterministic, 0 params) ──
def encode(tokens, depth=0):
    """Cyclic shift hash — bottom-up merge"""
    if len(tokens) <= 1:
        return E[tokens[0]] if len(tokens) == 1 else torch.zeros(d)
    mid = len(tokens) // 2
    HL = encode(tokens[:mid], depth + 1)
    HR = encode(tokens[mid:], depth + 1)
    return HL + sign_alt(torch.roll(HR, shifts=depth + 1, dims=-1))

# ── Decoder (trainable W_split per depth) ──
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_split = nn.ParameterList([
            nn.Parameter(torch.randn(d, d) * 0.1) for _ in range(depth)
        ])
    
    def split(self, H, level):
        """Predict H_left from parent hash H"""
        H_left_pred = H @ self.W_split[level]
        # H_right computed deterministically from encoding rule
        # H = HL + sign_alt(roll(HR, level+1))
        # → HR = reverse_roll(sign_alt(H - HL_pred))
        residual = H - H_left_pred
        H_right_pred = sign_alt(residual)  # undo sign_alt
        H_right_pred = torch.roll(H_right_pred, shifts=-(level + 1), dims=-1)  # un-roll
        return H_left_pred, H_right_pred
    
    def forward(self, H, remaining_depth, level=0):
        """Recursively split to leaves"""
        if remaining_depth == 0:
            # Leaf: return hash vector
            return [H]
        HL, HR = self.split(H, level)
        left_tokens = self.forward(HL, remaining_depth - 1, level + 1)
        right_tokens = self.forward(HR, remaining_depth - 1, level + 1)
        return left_tokens + right_tokens

decoder = Decoder()
opt = torch.optim.Adam(decoder.parameters(), lr=0.03)

# ── Train ──
print("Training decoder with per-depth W_split...")
for step in range(300):
    opt.zero_grad()
    loss = torch.tensor(0.0)
    
    for src_ids, tgt_ids in zip(src_sents, tgt_sents):
        # Encode source → root hash
        H_src = encode(src_ids)
        
        # Decode → predicted leaf hashes
        leaf_preds = decoder.forward(H_src, depth)
        
        # Ground truth: encode target to get target leaf hashes
        tgt_emb = E[tgt_ids]
        # Each leaf should match its corresponding target token embedding
        for i, (leaf_h, tgt_h) in enumerate(zip(leaf_preds, tgt_emb)):
            loss += ((leaf_h - tgt_h) ** 2).mean()
    
    loss = loss / (len(src_sents) * 3)  # avg per token
    loss.backward()
    opt.step()
    
    if step % 60 == 0:
        with torch.no_grad():
            # Test: decode "我打你" → should predict leaf tokens
            H_test = encode([0, 1, 2])
            leaf_test = decoder.forward(H_test, depth)
            pred_tokens = []
            for lh in leaf_test:
                scores = lh @ E.T
                pred_tokens.append(scores.argmax().item())
            acc = sum(1 for a, b in zip(pred_tokens, [5, 6, 7]) if a == b)
        print(f"  step {step:3d}: loss={loss.item():.4f}  test acc={acc}/3  pred={pred_tokens}")

# ── Final test ──
print(f"\n=== Final Results ===")
for i, (src_ids, tgt_ids) in enumerate(zip(src_sents, tgt_sents)):
    H_src = encode(src_ids)
    with torch.no_grad():
        leaf_preds = decoder.forward(H_src, depth)
        pred_tokens = [lh @ E.T for lh in leaf_preds]
        pred_tokens = [p.argmax().item() for p in pred_tokens]
    acc = sum(1 for a, b in zip(pred_tokens, tgt_ids) if a == b)
    print(f"  sent {i}: src={src_ids} → pred={pred_tokens} tgt={tgt_ids} acc={acc}/3")

print(f"\nW_split norms:")
for level, w in enumerate(decoder.W_split):
    print(f"  L{level}: {w.detach().norm().item():.3f}")
