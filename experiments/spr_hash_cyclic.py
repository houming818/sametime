"""
SPR Hash v2.1 — Cyclic Shift + Sign Break (零成本)
Fix: pure cyclic shift has collision — "A打B" == "B打A" when dims align
Solution: multiply right child by [1, -1, 1, -1...] after roll → breaks symmetry
"""
import torch

torch.manual_seed(42)
E_me  = torch.tensor([1.,2.,3.,4.])   # me
E_hit = torch.tensor([0.,1.,0.,1.])   # hit
E_you = torch.tensor([5.,6.,7.,8.])   # you

def sign_alt(x):
    mask = torch.tensor([1., -1.] * (x.shape[-1] // 2 + 1))[:x.shape[-1]]
    return x * mask

def hash_ordered(left, right, depth=0):
    return left + sign_alt(torch.roll(right, shifts=depth+1, dims=-1))

# Bug: pure cyclic shift
H_fwd_bug = E_me + torch.roll(E_you, shifts=1)
H_rev_bug = E_you + torch.roll(E_me, shifts=1)
print(f"BUG  pure roll:")
print(f"  me-hit-you: {H_fwd_bug.tolist()}")
print(f"  you-hit-me: {H_rev_bug.tolist()}")
print(f"  collision:  {torch.allclose(H_fwd_bug, H_rev_bug)}")

# Fixed: roll + sign alternation
H_fwd = hash_ordered(E_me, E_you)
H_rev = hash_ordered(E_you, E_me)
print(f"\nFIX  roll + sign alt [1,-1,1,-1]:")
print(f"  me-hit-you: {H_fwd.tolist()}")
print(f"  you-hit-me: {H_rev.tolist()}")
print(f"  separated:  {not torch.allclose(H_fwd, H_rev)}")
print(f"  deterministic: {torch.allclose(hash_ordered(E_me, E_you), hash_ordered(E_me, E_you))}")

# Full sentence test
s1 = [E_me, E_hit, E_you]
s2 = [E_you, E_hit, E_me]
def tree_hash(tokens, depth=0):
    if len(tokens) <= 1:
        return tokens[0] if tokens else torch.zeros(4)
    mid = len(tokens) // 2
    left = tree_hash(tokens[:mid], depth+1)
    right = tree_hash(tokens[mid:], depth+1)
    return hash_ordered(left, right, depth)

H_s1 = tree_hash(s1)
H_s2 = tree_hash(s2)
print(f"\nFull tree (3 tokens):")
print(f"  me-hit-you: {H_s1[:4].round(decimals=2).tolist()}")
print(f"  you-hit-me: {H_s2[:4].round(decimals=2).tolist()}")
print(f"  separated:  {not torch.allclose(H_s1, H_s2, atol=1e-3)}")
