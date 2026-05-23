"""
SPR Decoder v5 — trained Transformer embeddings
TW3 encoder/decoder embed → bilingual lookup
"""
import torch, numpy as np, math, os
from collections import Counter

# Load TW3 embeddings
ck = torch.load("checkpoints/tw3_d256_6l.pt", map_location="cpu")
sd = ck['model']
enc_embed = sd['encoder.embed.weight']  # (32000, 256) shared BPE vocab
dec_embed = sd.get('decoder.embed.weight', enc_embed)  # might be same

V, d = len(enc_embed), enc_embed.shape[1]
print(f"embed: {V} tokens × {d} dims")

# Load BPE tokenizer
import sentencepiece as spm
sp_de = spm.SentencePieceProcessor(); sp_de.load("checkpoints/wmt14_spm_de.model")
sp_en = spm.SentencePieceProcessor(); sp_en.load("checkpoints/wmt14_spm_en.model")

print(f"de vocab: {sp_de.vocab_size()}  en vocab: {sp_en.vocab_size()}")

# Load test sentences
tab = "/data/datasets/wmt14/wmt14.validation.de-en"
val_pairs = []
with open(tab) as f:
    for i, l in enumerate(f):
        if i >= 200: break
        if "\t" in l:
            de, en = l.strip().split("\t", 1)
            val_pairs.append((de, en))

print(f"test sentences: {len(val_pairs)}")

# Test: en word → nearest de word via trained embeddings
# We need word-level, but embeddings are BPE. Use simple approach:
# Encode each word with BPE, average BPE embeddings → word embedding → nearest de word similarly

def embed_word(word, sp, emb):
    ids = sp.encode(word, out_type=int)  # already strips BOS/EOS
    valid = [i for i in ids if i < len(emb)]
    if not valid: return None
    return emb[valid].mean(dim=0)

# Build DE word embeddings from short test sentences
de_word_emb = {}
for de, _ in val_pairs:
    for w in de.split():
        if w not in de_word_emb:
            emb = embed_word(w, sp_de, enc_embed)
            if emb is not None:
                de_word_emb[w] = emb

print(f"de words with embeddings: {len(de_word_emb)}")

# Nearest neighbor: en word → nearest de word
def translate_word(w):
    en_emb = embed_word(w, sp_en, enc_embed)
    if en_emb is None: return "?"
    best_w, best_s = "?", -1e9
    for dw, de_emb in de_word_emb.items():
        s = float(torch.cosine_similarity(en_emb.unsqueeze(0), de_emb.unsqueeze(0)).item())
        if s > best_s:
            best_s, best_w = s, dw
    return best_w

# BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
refs, hyps = [], []
for de, en in val_pairs[:50]:
    de_words = de.split()
    en_words = en.split()
    hyp = [translate_word(w) for w in en_words[:len(de_words)]]
    refs.append(de_words)
    hyps.append(hyp)

def bleu(rf, hp):
    C=Counter; ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(rf,hp):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1))
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

b = bleu(refs, hyps)
print(f"\nBLEU-4 = {b:.2f}")

# samples
for i in [0, 2, 5]:
    de_words = val_pairs[i][1].split()[:6]  # en
    ref_words = val_pairs[i][0].split()[:6]  # de
    hyp_words = [translate_word(w) for w in de_words]
    print(f"\n  src: {' '.join(de_words)}")
    print(f"  ref: {' '.join(ref_words)}")
    print(f"  hyp: {' '.join(hyp_words)}")
