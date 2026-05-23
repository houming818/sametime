"""
SPR Translation — Twin Tree: Encoder + Bridge + Decoder
Encoder: cyclic shift hash (source → root)
Bridge: Linear(d,d) cross-lingual mapping
Decoder: reverse shift split → leaf tokens → nearest neighbor
"""
import torch, torch.nn as nn
import numpy as np

torch.manual_seed(42)
d = 16

# ── Mock bilingual embeddings ──
E_cn = torch.randn(10, d) * 0.5  # 10 Chinese tokens
E_en = torch.randn(10, d) * 0.5  # 10 English tokens

# Sample sentence: 我(0) 打(1) 你(2) → I(5) hit(6) you(7)
src_tokens = [E_cn[0], E_cn[1], E_cn[2]]  # 我打你
tgt_tokens = [E_en[5], E_en[6], E_en[7]]  # I hit you

# ── Encoder: cyclic shift hash ──
def sign_alt(x):
    mask = torch.tensor([1., -1.] * (d//2 + 1))[:d]
    return x * mask

def encoder(tokens, depth=0):
    if len(tokens) <= 1:
        return tokens[0] if len(tokens) == 1 else torch.zeros(d)
    mid = len(tokens) // 2
    HL = encoder(tokens[:mid], depth + 1)
    HR = encoder(tokens[mid:], depth + 1)
    return HL + sign_alt(torch.roll(HR, shifts=depth + 1, dims=-1))

H_src = encoder(src_tokens)  # Chinese root hash
H_tgt = encoder(tgt_tokens)  # English root hash (ground truth)

print(f"Encoder: 我打你 → H_src={H_src[:4].round(decimals=2).tolist()}")
print(f"Encoder: I hit you → H_tgt={H_tgt[:4].round(decimals=2).tolist()}")

# ── Bridge: learn to map H_src → H_tgt ──
bridge = nn.Linear(d, d)
opt = torch.optim.SGD(bridge.parameters(), lr=0.1)

for step in range(100):
    opt.zero_grad()
    H_pred = bridge(H_src)
    loss = ((H_pred - H_tgt) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    H_mapped = bridge(H_src)
    print(f"\nBridge: H_src → H_mapped={H_mapped[:4].round(decimals=2).tolist()}")
    print(f"        target={H_tgt[:4].round(decimals=2).tolist()}")
    print(f"        MSE={((H_mapped - H_tgt)**2).mean().item():.4f}")

# ── Decoder: reverse the hash split ──
def decoder(H, depth, emb_matrix):
    """Reverse cyclic shift: predict left/right child hashes, recurse to leaves"""
    if depth == 0:
        # Leaf: nearest neighbor in target vocabulary
        scores = H @ emb_matrix.T  # (V,)
        return [scores.argmax().item()]
    
    # Predict left and right child hashes from parent
    # Simple split: use half of H as left, other half as right (learned weights would be better)
    HL_pred = H[:d//2].repeat(2)  # naive: expand half to full dim
    HR_pred = H[d//2:].repeat(2)
    
    # Un-shift the right child
    HR_unshifted = sign_alt(torch.roll(HR_pred, shifts=-(depth), dims=-1))
    
    left_tokens = decoder(HL_pred, depth - 1, emb_matrix)
    right_tokens = decoder(HR_unshifted, depth - 1, emb_matrix)
    return left_tokens + right_tokens

# ── Test: decode the mapped hash ──
pred_tokens = decoder(H_mapped, depth=2, emb_matrix=E_en)
print(f"\nDecoder: mapped hash → predicted tokens = {pred_tokens}")
print(f"         expected (I hit you) = [5, 6, 7]")
print(f"         match: {pred_tokens == [5, 6, 7]}")

# Also test: perfect echo (decode source hash in source space)
echo_tokens = decoder(H_src, depth=2, emb_matrix=E_cn)
print(f"\nEcho test: src hash → source tokens = {echo_tokens}")
print(f"           expected (我打你)   = [0, 1, 2]")
print(f"           match: {echo_tokens == [0, 1, 2]}")
