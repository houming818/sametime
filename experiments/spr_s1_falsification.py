"""
Owner: Nio Log Squad
Writer: Codex Review Engineer
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Re-run SPR S1 evidence and falsify the over-strong "path equals semantics" claim.

The script has three sections:
1. Order hash: pure roll collision vs roll+sign_alt separation.
2. Echo capacity: decomposed routing on WMT14 English tokens.
3. Polysemy falsification: current token-only S1 routing cannot distinguish senses.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import torch


POLYSEMY = {
    "light": {
        "illumination": [
            "the light from the lamp filled the room",
            "a bright light shone through the window",
            "the camera needs more light for the photo",
            "street light reflected on the wet road",
            "she turned on the light before reading",
            "morning light entered the kitchen",
        ],
        "weight": [
            "the suitcase was light enough to carry",
            "this fabric is light and easy to pack",
            "he ordered a light meal before training",
            "the box felt light after we emptied it",
            "a light jacket is enough today",
            "the tool is light but strong",
        ],
    },
    "bank": {
        "finance": [
            "the bank approved the loan yesterday",
            "she opened a savings account at the bank",
            "the bank raised interest rates again",
            "customers waited inside the bank lobby",
            "the bank transferred the payment",
            "he called the bank about his card",
        ],
        "river": [
            "the boat stopped near the river bank",
            "children played on the muddy bank",
            "trees grew along the bank of the stream",
            "the flood covered the lower bank",
            "we sat on the grassy bank",
            "the path followed the river bank",
        ],
    },
    "charge": {
        "money": [
            "the hotel added an extra charge",
            "there is no charge for delivery",
            "the repair shop waived the charge",
            "a small charge appears on the bill",
            "they disputed the service charge",
            "the bank reversed the charge",
        ],
        "electric": [
            "the battery needs more charge",
            "plug in the phone to charge overnight",
            "the device can charge quickly",
            "solar panels charge the battery",
            "the charger failed to charge the laptop",
            "the car will charge at the station",
        ],
        "legal": [
            "the prosecutor filed a criminal charge",
            "he denied the charge in court",
            "police added another charge",
            "the judge dismissed the charge",
            "the charge carried a heavy penalty",
            "lawyers challenged the charge",
        ],
    },
}


KEYWORDS = {
    "light": {
        "illumination": {"lamp", "bright", "shone", "window", "camera", "photo", "street", "reflected", "turned", "reading", "morning", "kitchen"},
        "weight": {"suitcase", "carry", "fabric", "pack", "meal", "training", "box", "emptied", "jacket", "tool", "strong"},
    },
    "bank": {
        "finance": {"approved", "loan", "savings", "account", "raised", "interest", "customers", "lobby", "transferred", "payment", "card"},
        "river": {"boat", "river", "muddy", "trees", "stream", "flood", "lower", "grassy", "path"},
    },
    "charge": {
        "money": {"hotel", "extra", "delivery", "repair", "shop", "waived", "bill", "service", "reversed"},
        "electric": {"battery", "plug", "phone", "overnight", "device", "quickly", "solar", "panels", "charger", "laptop", "car", "station"},
        "legal": {"prosecutor", "filed", "criminal", "denied", "court", "police", "judge", "dismissed", "penalty", "lawyers", "challenged"},
    },
}


def sign_alt(x):
    mask = torch.tensor([1.0, -1.0] * (x.shape[-1] // 2 + 1), device=x.device)
    return x * mask[: x.shape[-1]]


def order_hash_result():
    e_me = torch.tensor([1.0, 2.0, 3.0, 4.0])
    e_you = torch.tensor([5.0, 6.0, 7.0, 8.0])
    pure_a = e_me + torch.roll(e_you, shifts=1)
    pure_b = e_you + torch.roll(e_me, shifts=1)
    fixed_a = e_me + sign_alt(torch.roll(e_you, shifts=1))
    fixed_b = e_you + sign_alt(torch.roll(e_me, shifts=1))
    return {
        "pure_roll_collision": bool(torch.allclose(pure_a, pure_b)),
        "sign_alt_separated": bool(not torch.allclose(fixed_a, fixed_b)),
        "pure_a": pure_a.tolist(),
        "pure_b": pure_b.tolist(),
        "fixed_a": fixed_a.tolist(),
        "fixed_b": fixed_b.tolist(),
    }


def load_english(path, limit):
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
            rc = Counter(ngrams(ref, n))
            hc = Counter(ngrams(hyp, n))
            total += sum(hc.values())
            matched += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        precisions.append(matched / max(total, 1) if total else 1.0)
    bp = min(1.0, math.exp(max((1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if h), default=0)))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in precisions) / 4) * 100


def route_chunks(emb, chunks=4, depth=7):
    vocab_size, dim = emb.shape
    chunk_dim = dim // chunks
    leaves = 1 << depth
    sign_mask = torch.tensor([1.0, -1.0] * (chunk_dim // 2 + 1), device=emb.device)[:chunk_dim]
    leaf_chunks = torch.zeros(vocab_size, chunks, dtype=torch.long, device=emb.device)
    for k in range(chunks):
        chunk = emb[:, k * chunk_dim : (k + 1) * chunk_dim]
        idx = torch.zeros(vocab_size, dtype=torch.long, device=emb.device)
        current = chunk.clone()
        for dp in range(depth):
            current = torch.roll(current, shifts=dp + 1, dims=-1) * sign_mask
            idx = idx * 2 + 1
            idx[(chunk * current).sum(dim=-1) > 0] += 1
        leaf_chunks[:, k] = idx - (leaves - 1)
    combined = torch.zeros(vocab_size, dtype=torch.long, device=emb.device)
    for k in range(chunks - 1, -1, -1):
        combined = combined * leaves + leaf_chunks[:, k]
    return combined


def echo_capacity(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    train = load_english(args.train_file, args.train_limit)
    val = load_english(args.val_file, args.val_limit)
    word2id = {"<pad>": 0, "<unk>": 1}
    for sent in train + val:
        for word in sent:
            if word not in word2id:
                word2id[word] = len(word2id)
    # Match the original experiment: generate on CPU, then move to CUDA.
    emb = torch.randn(len(word2id), args.dim).to(device)
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    combined = route_chunks(emb, args.chunks, args.depth)
    leaf_words = defaultdict(list)
    for wid in range(len(word2id)):
        leaf_words[int(combined[wid].item())].append(wid)
    leaf_top = {leaf: Counter(words).most_common(1)[0][0] for leaf, words in leaf_words.items()}
    refs, hyps = [], []
    for sent in val[: args.eval_limit]:
        ids = [word2id.get(w, 1) for w in sent]
        if len(ids) >= 4:
            refs.append(ids)
            hyps.append([leaf_top[int(combined[wid].item())] for wid in ids])
    solo = sum(1 for words in leaf_words.values() if len(words) == 1)
    return {
        "device": device,
        "vocab": len(word2id),
        "active_leaves": len(leaf_words),
        "solo": solo,
        "solo_percent": 100 * solo / len(word2id),
        "bleu4": bleu4(refs, hyps),
    }


def make_polysemy_rows(shuffle_labels=False, seed=42):
    rows = []
    for target, senses in POLYSEMY.items():
        for sense, examples in senses.items():
            for text in examples:
                rows.append({"target": target, "sense": sense, "tokens": text.split()})
    if shuffle_labels:
        rng = random.Random(seed)
        for target in {r["target"] for r in rows}:
            idxs = [i for i, r in enumerate(rows) if r["target"] == target]
            labels = [rows[i]["sense"] for i in idxs]
            rng.shuffle(labels)
            for i, label in zip(idxs, labels):
                rows[i]["sense"] = label
    return rows


def token_only_polysemy(rows):
    # Current S1 token route has one path per token. Same target => same feature.
    total, correct = 0, 0
    by_target = {}
    for target in {r["target"] for r in rows}:
        labels = [r["sense"] for r in rows if r["target"] == target]
        pred = Counter(labels).most_common(1)[0][0]
        by_target[target] = pred
    for row in rows:
        total += 1
        correct += int(by_target[row["target"]] == row["sense"])
    return correct / total


def keyword_polysemy(rows):
    total, correct = 0, 0
    for row in rows:
        scores = {}
        context = set(w.strip(".,;:!?").lower() for w in row["tokens"] if w != row["target"])
        for sense, keys in KEYWORDS[row["target"]].items():
            scores[sense] = len(context & keys)
        pred = max(scores, key=scores.get)
        total += 1
        correct += int(pred == row["sense"])
    return correct / total


def polysemy_falsification(seed):
    real_rows = make_polysemy_rows(False, seed)
    shuffled_rows = make_polysemy_rows(True, seed)
    return {
        "examples": len(real_rows),
        "targets": sorted({r["target"] for r in real_rows}),
        "token_only_real_acc": token_only_polysemy(real_rows),
        "token_only_shuffled_acc": token_only_polysemy(shuffled_rows),
        "keyword_real_acc": keyword_polysemy(real_rows),
        "keyword_shuffled_acc": keyword_polysemy(shuffled_rows),
        "interpretation": "Current S1 token-only routing is invariant across senses of the same word; it proves identity capacity, not contextual semantics.",
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
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result = {
        "order_hash": order_hash_result(),
        "echo_capacity": echo_capacity(args),
        "polysemy_falsification": polysemy_falsification(args.seed),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(
        "SUMMARY "
        f"collision={result['order_hash']['pure_roll_collision']} "
        f"sign_alt={result['order_hash']['sign_alt_separated']} "
        f"solo={result['echo_capacity']['solo']}/{result['echo_capacity']['vocab']} "
        f"bleu4={result['echo_capacity']['bleu4']:.2f} "
        f"token_polysemy={result['polysemy_falsification']['token_only_real_acc']:.2f} "
        f"keyword_polysemy={result['polysemy_falsification']['keyword_real_acc']:.2f}"
    )


if __name__ == "__main__":
    main()
