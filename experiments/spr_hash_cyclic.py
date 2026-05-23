"""
SPR Hash v2 — Cyclic Shift (torch.roll) as zero-cost positional encoding
Depth-aware: deeper = more shift = higher base exponent
Left child: no shift. Right child: shift = depth+1
"""
import torch, numpy as np

def cyclic_hash(tokens, depth=0):
    """Recursive position-split tree with cyclic shift merge.
    Each level deeper → shift +1 → encodes position as base exponent.
    Left subtree: no shift (original position preserved).
    Right subtree: rolled right by depth+1 (encodes 'later in sequence').
    """
    if len(tokens) <= 1:
        return tokens[0] if len(tokens) == 1 else torch.zeros(tokens[0].shape[0]) if tokens else None
    mid = len(tokens) // 2
    HL = cyclic_hash(tokens[:mid], depth + 1)  # left child
    HR = cyclic_hash(tokens[mid:], depth + 1)   # right child
    # right child shifted — deeper subtree = more shift
    return HL + torch.roll(HR, shifts=depth + 1, dims=-1)

def cyclic_hash_chunked(embeddings, depth=0):
    """Same as cyclic_hash but takes an embedding tensor (N, d) and depth."""
    if len(embeddings) <= 1:
        return embeddings[0] if len(embeddings) == 1 else torch.zeros(embeddings.shape[1])
    mid = len(embeddings) // 2
    HL = cyclic_hash_chunked(embeddings[:mid], depth + 1)
    HR = cyclic_hash_chunked(embeddings[mid:], depth + 1)
    return HL + torch.roll(HR, shifts=depth + 1, dims=-1)

# ── test ──
if __name__ == "__main__":
    torch.manual_seed(42)
    E = torch.randn(10, 16) * 0.5
    sent = [E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7]]  # 8 tokens
    
    H_fwd = cyclic_hash(sent)
    H_rev = cyclic_hash(sent[::-1])
    
    print(f"cyclic shift hash (8 tokens, depth 0-3)")
    print(f"  fwd: {H_fwd[:6].numpy().round(2)}")
    print(f"  rev: {H_rev[:6].numpy().round(2)}")
    print(f"  order-aware: {not torch.allclose(H_fwd, H_rev, atol=1e-3)}")
    print(f"  deterministic: {torch.allclose(cyclic_hash(sent), cyclic_hash(sent), atol=1e-3)}")
    
    # Chunked version
    H_fwd2 = cyclic_hash_chunked(torch.stack(sent))
    print(f"\n  chunked fwd: {H_fwd2[:6].numpy().round(2)}")
    print(f"  chunked == list: {torch.allclose(H_fwd, H_fwd2, atol=1e-3)}")
