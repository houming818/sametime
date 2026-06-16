"""
Owner: Nio Log Squad
Writer: Codex Review Engineer
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Controlled proof that SPR needs context-conditioned routing for polysemy.

This is not a corpus benchmark. It proves a narrower mechanism:
- token-only routing gives the same path for the same token, so it cannot separate senses;
- route(token + context) can separate senses when the context vector carries sense signal;
- shuffled labels collapse the context-conditioned advantage.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import torch


POLYSEMY = {
    "light": {
        "illumination": [
            "lamp bright window glow",
            "camera photo exposure lamp",
            "street glow night window",
            "morning sun room bright",
            "candle glow shadow room",
            "flash photo bright camera",
            "window sun beam morning",
            "lamp switch room glow",
        ],
        "weight": [
            "suitcase carry fabric pack",
            "jacket thin travel carry",
            "meal small training diet",
            "box empty carry easy",
            "fabric thin summer jacket",
            "tool small easy carry",
            "package empty light carry",
            "travel bag easy pack",
        ],
    },
    "bank": {
        "finance": [
            "loan account teller money",
            "savings account interest payment",
            "credit card teller branch",
            "deposit money account branch",
            "loan approval interest credit",
            "payment transfer account money",
            "branch customer teller deposit",
            "card payment credit account",
        ],
        "river": [
            "river water mud grass",
            "stream boat grass shore",
            "flood water mud river",
            "tree grass stream shore",
            "boat river shore water",
            "muddy grass path stream",
            "river bend water tree",
            "shore grass boat path",
        ],
    },
    "charge": {
        "money": [
            "fee bill hotel payment",
            "service fee invoice payment",
            "repair bill cost service",
            "hotel invoice cost fee",
            "payment cost extra bill",
            "service charge invoice hotel",
            "fee payment repair bill",
            "extra cost service invoice",
        ],
        "electric": [
            "battery plug cable phone",
            "charger battery laptop cable",
            "solar panel battery station",
            "phone plug charger overnight",
            "electric car station battery",
            "laptop cable plug power",
            "battery power charger device",
            "station cable electric car",
        ],
        "legal": [
            "court judge prosecutor case",
            "police court criminal case",
            "lawyer judge penalty court",
            "prosecutor criminal trial police",
            "case penalty lawyer judge",
            "trial court criminal charge",
            "police prosecutor case charge",
            "judge dismissed legal case",
        ],
    },
}


def sign_alt(x):
    mask = torch.tensor([1.0, -1.0] * (x.shape[-1] // 2 + 1), device=x.device)
    return x * mask[: x.shape[-1]]


def route_bits(vec, chunks=4, depth=7):
    dim = vec.numel()
    chunk_dim = dim // chunks
    bits = []
    for chunk_id in range(chunks):
        chunk = vec[chunk_id * chunk_dim : (chunk_id + 1) * chunk_dim]
        current = chunk.clone()
        for level in range(depth):
            current = sign_alt(torch.roll(current, shifts=level + 1, dims=-1))
            bits.append(1 if torch.dot(chunk, current).item() > 0 else 0)
    return torch.tensor(bits, dtype=torch.float32)


def hamming(a, b):
    return int((a != b).sum().item())


def make_rows():
    rows = []
    for target, senses in POLYSEMY.items():
        for sense, contexts in senses.items():
            for context in contexts:
                rows.append(
                    {
                        "target": target,
                        "sense": sense,
                        "context_words": context.split(),
                    }
                )
    return rows


def build_embeddings(rows, dim, seed, noise):
    gen = torch.Generator().manual_seed(seed)
    token_emb = {}
    sense_anchor = {}
    word_emb = {}

    for row in rows:
        token_emb.setdefault(row["target"], torch.randn(dim, generator=gen))
        key = (row["target"], row["sense"])
        sense_anchor.setdefault(key, torch.randn(dim, generator=gen))

    for row in rows:
        anchor = sense_anchor[(row["target"], row["sense"])]
        for word in row["context_words"]:
            if word not in word_emb:
                word_emb[word] = anchor + noise * torch.randn(dim, generator=gen)

    token_emb = {k: v / (v.norm() + 1e-8) for k, v in token_emb.items()}
    word_emb = {k: v / (v.norm() + 1e-8) for k, v in word_emb.items()}
    return token_emb, word_emb


def context_vec(row, token_emb, word_emb, alpha):
    ctx = torch.stack([word_emb[w] for w in row["context_words"]]).mean(dim=0)
    vec = token_emb[row["target"]] + alpha * ctx
    return vec / (vec.norm() + 1e-8)


def token_vec(row, token_emb):
    return token_emb[row["target"]]


def shuffle_labels(rows, seed):
    rng = random.Random(seed)
    copied = [dict(r) for r in rows]
    for target in sorted({r["target"] for r in copied}):
        idxs = [i for i, r in enumerate(copied) if r["target"] == target]
        labels = [copied[i]["sense"] for i in idxs]
        rng.shuffle(labels)
        for i, label in zip(idxs, labels):
            copied[i]["sense"] = label
    return copied


def loo_1nn(rows, features):
    correct = 0
    total = 0
    for i, row in enumerate(rows):
        candidates = [j for j, other in enumerate(rows) if j != i and other["target"] == row["target"]]
        nearest = min(candidates, key=lambda j: hamming(features[i], features[j]))
        pred = rows[nearest]["sense"]
        correct += int(pred == row["sense"])
        total += 1
    return correct / total


def path_purity(rows, features):
    buckets = defaultdict(list)
    for row, feat in zip(rows, features):
        buckets[(row["target"], tuple(int(x) for x in feat.tolist()))].append(row["sense"])
    pure = 0
    total = 0
    collisions = 0
    for labels in buckets.values():
        total += len(labels)
        pure += Counter(labels).most_common(1)[0][1]
        if len(set(labels)) > 1:
            collisions += 1
    return {
        "bucket_count": len(buckets),
        "mixed_buckets": collisions,
        "purity": pure / total,
    }


def run(args):
    rows = make_rows()
    shuffled = shuffle_labels(rows, args.seed)
    token_emb, word_emb = build_embeddings(rows, args.dim, args.seed, args.noise)

    token_features = [route_bits(token_vec(row, token_emb), args.chunks, args.depth) for row in rows]
    context_features = [
        route_bits(context_vec(row, token_emb, word_emb, args.alpha), args.chunks, args.depth)
        for row in rows
    ]
    shuffled_context_features = [
        route_bits(context_vec(row, token_emb, word_emb, args.alpha), args.chunks, args.depth)
        for row in shuffled
    ]

    return {
        "setup": {
            "examples": len(rows),
            "targets": sorted({r["target"] for r in rows}),
            "dim": args.dim,
            "chunks": args.chunks,
            "depth": args.depth,
            "alpha": args.alpha,
            "noise": args.noise,
            "seed": args.seed,
        },
        "metrics": {
            "token_only_loo_acc": loo_1nn(rows, token_features),
            "context_route_loo_acc": loo_1nn(rows, context_features),
            "context_route_shuffled_label_acc": loo_1nn(shuffled, shuffled_context_features),
            "token_path_purity": path_purity(rows, token_features),
            "context_path_purity": path_purity(rows, context_features),
        },
        "claim": {
            "proves": "Context-conditioned routing can separate controlled polysemy when context vectors carry sense signal.",
            "does_not_prove": "It does not prove real-corpus semantic routing or translation quality.",
            "architecture_decision": "S1b must be route(token, context), not route(token).",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--noise", type=float, default=0.08)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    metrics = result["metrics"]
    print(
        "SUMMARY "
        f"token_acc={metrics['token_only_loo_acc']:.3f} "
        f"context_acc={metrics['context_route_loo_acc']:.3f} "
        f"shuffled_acc={metrics['context_route_shuffled_label_acc']:.3f} "
        f"context_purity={metrics['context_path_purity']['purity']:.3f} "
        f"mixed_context_buckets={metrics['context_path_purity']['mixed_buckets']}"
    )


if __name__ == "__main__":
    main()
