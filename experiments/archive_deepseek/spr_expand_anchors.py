"""
Expand anchor pairs using embedding-space nearest neighbors.
Load a trained tree_alt checkpoint, find the top-K nearest EN/ZH tokens
to existing anchor words, and expand the anchor list.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128
print(f"vocab={V}")

def ok(ids): return all(x != 0 for x in ids)

# ─── Load existing anchors ───
# (Import from spr_anchor_bridge — just hardcode relevant pairs here)
import sys, os
sys.path.insert(0, '/workspace' if os.path.exists('/workspace') else '.')
try:
    from spr_anchor_bridge import ANCHORS as existing_anchors
except:
    # Fallback: load from spr_anchor_bridge.py
    exec(open('/workspace/spr_anchor_bridge.py').read().split("ANCHOR_WORDS = [")[1].split("]")[0])
    _tmp = []
    for line in ANCHOR_WORDS:
        pass
    # Hmm, this is fragile. Let me just use the anchors we know.

# Simple: scan the file for ANCHOR_WORDS
en_pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    content = f.read()
    idx = content.find("ANCHOR_WORDS = [")
    idx2 = content.find("]", idx)
    block = content[idx:idx2+1]
    # Parse lines like ('one','一'),
    import re
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", block):
        en_pairs.append((m.group(1), m.group(2)))

print(f"Loaded {len(en_pairs)} manual anchor pairs")

# ─── Load model ───
class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
    def fd(self, x): return self.dp(self.dec(x)[0])

L0 = nn.Embedding(V,d).to(device)
L1 = BiGRU(d).to(device)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_alt.pt'
ckpt = torch.load(ckpt_path, map_location=device)
L0.load_state_dict(ckpt['L0'])
print(f"Loaded {ckpt_path}")

K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
print(f"Expanding anchors: top-{K} nearest neighbors per anchor word\n")

# For each EN anchor word, find nearest K EN neighbors and nearest K ZH neighbors
new_pairs = set()
for en_word, zh_word in en_pairs:
    en_ids = sp.encode_as_ids(en_word)
    zh_ids = sp.encode_as_ids(zh_word)
    if not ok(en_ids) or not ok(zh_ids): continue

    # Get anchor embeddings
    with torch.no_grad():
        e_en = L0.weight[torch.tensor(en_ids, device=device)].mean(dim=0)
        e_zh = L0.weight[torch.tensor(zh_ids, device=device)].mean(dim=0)

    # Find nearest K EN neighbors (excluding self)
    en_cos = F.cosine_similarity(e_en.unsqueeze(0), L0.weight, dim=-1)
    en_cos[en_ids[0]] = -2  # exclude self
    top_en = en_cos.topk(K).indices.cpu().tolist()

    # Find nearest K ZH neighbors (excluding self)
    zh_cos = F.cosine_similarity(e_zh.unsqueeze(0), L0.weight, dim=-1)
    zh_cos[zh_ids[0]] = -2
    top_zh = zh_cos.topk(K).indices.cpu().tolist()

    # Create new anchor pairs: nearest EN → nearest ZH
    for i in range(K):
        en_tok = top_en[i]
        zh_tok = top_zh[i]
        en_piece = sp.id_to_piece(en_tok)
        zh_piece = sp.id_to_piece(zh_tok)
        en_cos_val = en_cos[en_tok].item()
        zh_cos_val = zh_cos[zh_tok].item()

        # Filter: don't add pairs with very low cos (meaningless neighbors)
        if en_cos_val < 0.3 and zh_cos_val < 0.3: continue

        # Filter: skip pure subword/punctuation tokens
        if en_piece.startswith('▁') and len(en_piece) < 3: continue
        if zh_piece.startswith('▁') and len(zh_piece) < 3: continue

        en_id_list = [en_tok]
        zh_id_list = [zh_tok]
        new_pairs.add((en_piece, zh_piece, en_id_list, zh_id_list, en_cos_val, zh_cos_val))

print(f"Expanded anchor pairs: {len(new_pairs)}")

# Print as Python list for ANCHOR_WORDS
print(f"\n# === Expanded anchors ({len(new_pairs)} pairs) ===")
print("EXPANDED_ANCHORS = [")
for en_p, zh_p, e_ids, z_ids, ec, zc in sorted(new_pairs, key=lambda x: -(x[4]+x[5])):
    print(f"    ('{en_p}', '{zh_p}'),  # en_cos={ec:.2f} zh_cos={zc:.2f}")
print("]")

# Also print the IDs for direct use in ANCHORS list
print(f"\n# === BPE-ID format for spr_anchor_bridge.py ===")
print(f"# Replace ANCHORS assignment in build_anchors() with:")
already = set()
for en_p, zh_p, e_ids, z_ids, ec, zc in sorted(new_pairs, key=lambda x: -(x[4]+x[5])):
    if en_p not in already:
        already.add(en_p)
        print(f"    ({e_ids}, {z_ids}),  # {en_p} → {zh_p}")
