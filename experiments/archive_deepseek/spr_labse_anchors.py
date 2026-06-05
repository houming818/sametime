"""Generate LaBSE-guided anchor pairs: EN token → nearest ZH token by cosine."""
import torch, torch.nn.functional as F, sentencepiece as spm, re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

teacher = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_teacher.pt', map_location=device)['teacher_emb']
teacher = F.normalize(teacher, dim=-1)
print(f"Teacher: {teacher.shape}")

# Get manual anchor IDs to exclude
manual_ids = set()
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    idx = txt.find("ANCHOR_WORDS = [")
    if idx < 0:
        print("WARNING: ANCHOR_WORDS not found!")
    else:
        idx2 = txt.find("]", idx)
        block = txt[idx:idx2+1]
        for m in re.finditer(r"\('([^']+)','([^']+)'\)", block):
            ei = sp.encode_as_ids(m.group(1)); zi = sp.encode_as_ids(m.group(2))
            if ok(ei) and ok(zi):
                manual_ids.add((ei[0], zi[0]))
print(f"Manual anchors (to exclude): {len(manual_ids)}")

# Language detection: EN tokens contain ASCII letters, ZH tokens contain CJK chars
def is_en_piece(p):
    return any(ord(c) < 128 and c.isalpha() for c in p.replace('\u2581',''))
def is_zh_piece(p):
    return any('\u4E00' <= c <= '\u9FFF' for c in p)

pairs = []
CHUNK = 2000
print("Computing cosine matrix...")
for i_start in range(0, V, CHUNK):
    i_end = min(i_start + CHUNK, V)
    cos_block = teacher[i_start:i_end] @ teacher.T  # [CHUNK, V]
    # For each EN token in this chunk, find top ZH tokens
    # Mask self
    cos_block[torch.arange(min(CHUNK, i_end-i_start)), i_start:i_end] = -2
    top_vals, top_idx = cos_block.topk(20, dim=-1)  # [CHUNK, 20]
    for k in range(i_end - i_start):
        ei = i_start + k
        ep_piece = sp.id_to_piece(ei)
        en_side = is_en_piece(ep_piece)
        for m in range(20):
            zj = top_idx[k, m].item()
            c = top_vals[k, m].item()
            zp_piece = sp.id_to_piece(zj)
            zh_side = is_zh_piece(zp_piece)
            # EN→ZH only
            if c > 0.35 and en_side and zh_side:  # lowered threshold — more anchors
                key = (ei, zj)
                if key not in manual_ids:
                    if len(ep_piece) >= 2 and len(zp_piece) >= 2:
                        pairs.append((ep_piece.replace('\u2581',''), zp_piece.replace('\u2581',''), c))
    if i_start % 4000 == 0:
        print(f"  {i_start}/{V}  pairs: {len(pairs)}")

# Also sweep ZH→EN
print("Sweep ZH→EN...")
for i_start in range(0, V, CHUNK):
    i_end = min(i_start + CHUNK, V)
    cos_block = teacher[i_start:i_end] @ teacher.T
    for k in range(i_end - i_start):
        ei = i_start + k
        ep_piece = sp.id_to_piece(ei)
        if not is_zh_piece(ep_piece): continue
        top_vals, top_idx = cos_block[k:k+1].topk(10, dim=-1)
        for m in range(10):
            zj = top_idx[0, m].item()
            c = top_vals[0, m].item()
            zp_piece = sp.id_to_piece(zj)
            if c > 0.35 and is_en_piece(zp_piece):
                key = (zj, ei)
                if key not in manual_ids:
                    if len(ep_piece) >= 2 and len(zp_piece) >= 2:
                        pairs.append((zp_piece.replace('\u2581',''), ep_piece.replace('\u2581',''), c))

print(f"Total pairs (cos>0.4, cross-lang): {len(pairs)}")
pairs.sort(key=lambda x: -x[2])
top_n = min(len(pairs), 8000)
pairs = pairs[:top_n]

# Deduplicate by EN token
seen_en = set()
dedup = []
for ep, zp, c in pairs:
    en_ids = sp.encode_as_ids(ep)
    if ok(en_ids) and en_ids[0] not in seen_en:
        seen_en.add(en_ids[0])
        dedup.append((ep, zp, c))
print(f"  After dedup: {len(dedup)}")

# Write
with open('/workspace/labse_anchors.py', 'w') as f:
    f.write(f"# LaBSE-guided anchors ({len(dedup)} pairs, cos>0.5)\n")
    f.write("LABSE_ANCHORS = [\n")
    for ep, zp, c in dedup:
        f.write(f"    ({repr(ep)}, {repr(zp)}),  # cos={c:.3f}\n")
    f.write("]\n")
print("Saved: /workspace/labse_anchors.py")

print("\nSample:")
for ep, zp, c in dedup[:15]:
    print(f"  {ep:20s} → {zp:20s}  cos={c:.3f}")
