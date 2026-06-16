"""
Owner: Nio Log Squad
Writer: Codex Review Engineer
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Reproduce SPR S1 smoke results for order sensitivity and echo capacity.

This script intentionally mirrors the prior S1 experiments:
- spr_hash_cyclic.py: pure cyclic roll collision, sign alternation separation.
- spr_echo_proof.py: decomposed TreeHeap routing on WMT14 English tokens.
"""

import argparse
import json
import math
from collections import Counter, defaultdict

import torch


def sign_alt(x):
    mask = torch.tensor([1.0, -1.0] * (x.shape[-1] // 2 + 1), device=x.device)
    return x * mask[: x.shape[-1]]


def hash_ordered(left, right, depth=0):
    return left + sign_alt(torch.roll(right, shifts=depth + 1, dims=-1))


def reproduce_order_hash():
    e_me = torch.tensor([1.0, 2.0, 3.0, 4.0])
    e_hit = torch.tensor([0.0, 1.0, 0.0, 1.0])
    e_you = torch.tensor([5.0, 6.0, 7.0, 8.0])

    h_fwd_bug = e_me + torch.roll(e_you, shifts=1)
    h_rev_bug = e_you + torch.roll(e_me, shifts=1)
    h_fwd = hash_ordered(e_me, e_you)
    h_rev = hash_ordered(e_you, e_me)

    def tree_hash(tokens, depth=0):
        if len(tokens) <= 1:
            return tokens[0] if tokens else torch.zeros(4)
        mid = len(tokens) // 2
        left = tree_hash(tokens[:mid], depth + 1)
        right = tree_hash(tokens[mid:], depth + 1)
        return hash_ordered(left, right, depth)

    h_s1 = tree_hash([e_me, e_hit, e_you])
    h_s2 = tree_hash([e_you, e_hit, e_me])
    return {
        "pure_roll_collision": bool(torch.allclose(h_fwd_bug, h_rev_bug)),
        "sign_alt_separated": bool(not torch.allclose(h_fwd, h_rev)),
        "full_tree_separated": bool(not torch.allclose(h_s1, h_s2, atol=1e-3)),
        "pure_roll_forward": h_fwd_bug.tolist(),
        "pure_roll_reverse": h_rev_bug.tolist(),
        "sign_alt_forward": h_fwd.tolist(),
        "sign_alt_reverse": h_rev.tolist(),
    }


def load_english_sents(path, limit):
    sents = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if "\t" in line:
                sents.append(line.split("\t", 1)[1].strip().lower().split())
    return sents


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu4(refs, hyps):
    precisions = []
    for n in range(1, 5):
        matched, total = 0, 0
        for ref, hyp in zip(refs, hyps):
            ref_counts = Counter(ngrams(ref, n))
            hyp_counts = Counter(ngrams(hyp, n))
            total += sum(hyp_counts.values())
            matched += sum(min(hyp_counts[k], ref_counts.get(k, 0)) for k in hyp_counts)
        precisions.append(matched / max(total, 1) if total > 0 else 1.0)
    bp_values = [1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    brevity = min(1.0, math.exp(max(bp_values) if bp_values else 0))
    return brevity * math.exp(sum(math.log(max(p, 1e-10)) for p in precisions) / 4) * 100


def reproduce_echo(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)

    train_sents = load_english_sents(args.train_file, args.train_limit)
    val_sents = load_english_sents(args.val_file, args.val_limit)

    word2id = {"<pad>": 0, "<unk>": 1}
    for sent in train_sents + val_sents:
        for word in sent:
            if word not in word2id:
                word2id[word] = len(word2id)

    vocab_size = len(word2id)
    dim = args.dim
    chunks = args.chunks
    chunk_dim = dim // chunks
    chunk_depth = args.chunk_depth
    chunk_leaves = 1 << chunk_depth
    total_leaves = chunk_leaves**chunks

    # Mirror spr_echo_proof.py exactly: generate on CPU, then move to CUDA.
    # CPU and CUDA RNG streams differ even with the same manual seed.
    emb = torch.randn(vocab_size, dim).to(device)
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    sign_mask = torch.tensor([1.0, -1.0] * (chunk_dim // 2 + 1), device=device)[:chunk_dim]

    def route_chunk(chunk_emb):
        idx = torch.zeros(len(chunk_emb), dtype=torch.long, device=device)
        current = chunk_emb.clone()
        for depth in range(chunk_depth):
            current = torch.roll(current, shifts=depth + 1, dims=-1) * sign_mask
            scores = (chunk_emb * current).sum(dim=-1)
            idx = idx * 2 + 1
            idx[scores > 0] += 1
        return idx - (chunk_leaves - 1)

    leaf_chunks = torch.zeros(vocab_size, chunks, dtype=torch.long, device=device)
    chunk_solo = []
    for chunk_id in range(chunks):
        chunk_emb = emb[:, chunk_id * chunk_dim : (chunk_id + 1) * chunk_dim]
        leaf_chunks[:, chunk_id] = route_chunk(chunk_emb)
        counts = leaf_chunks[:, chunk_id].unique(return_counts=True)[1]
        chunk_solo.append(int((counts == 1).sum().item()))

    leaf_combined = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for chunk_id in range(chunks - 1, -1, -1):
        leaf_combined = leaf_combined * chunk_leaves + leaf_chunks[:, chunk_id]

    leaf_words = defaultdict(list)
    for word_id in range(vocab_size):
        leaf_words[int(leaf_combined[word_id].item())].append(word_id)

    solo = sum(1 for words in leaf_words.values() if len(words) == 1)
    multi = sum(1 for words in leaf_words.values() if len(words) > 1)
    leaf_top = {leaf: Counter(words).most_common(1)[0][0] for leaf, words in leaf_words.items()}

    refs, hyps = [], []
    for sent in val_sents[: args.eval_limit]:
        ids = [word2id.get(word, 1) for word in sent]
        if len(ids) < 4:
            continue
        refs.append(ids)
        hyps.append([leaf_top[int(leaf_combined[word_id].item())] for word_id in ids])

    return {
        "device": device,
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "eval_sentences": len(refs),
        "vocab": vocab_size,
        "dim": dim,
        "chunks": chunks,
        "chunk_dim": chunk_dim,
        "chunk_depth": chunk_depth,
        "chunk_leaves": chunk_leaves,
        "total_effective_leaves": total_leaves,
        "words_per_leaf": vocab_size / total_leaves,
        "chunk_solo": chunk_solo,
        "combined_solo": solo,
        "combined_multi": multi,
        "active_leaves": len(leaf_words),
        "solo_percent": 100 * solo / vocab_size,
        "bleu4": bleu4(refs, hyps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="/data/datasets/wmt14/wmt14.train.de-en")
    parser.add_argument("--val-file", default="/data/datasets/wmt14/wmt14.validation.de-en")
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-depth", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    result = {
        "order_hash": reproduce_order_hash(),
        "echo": reproduce_echo(args),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    echo = result["echo"]
    print(
        "SUMMARY "
        f"collision={result['order_hash']['pure_roll_collision']} "
        f"sign_alt_separated={result['order_hash']['sign_alt_separated']} "
        f"solo={echo['combined_solo']}/{echo['vocab']} "
        f"solo_percent={echo['solo_percent']:.2f} "
        f"bleu4={echo['bleu4']:.2f}"
    )


if __name__ == "__main__":
    main()
