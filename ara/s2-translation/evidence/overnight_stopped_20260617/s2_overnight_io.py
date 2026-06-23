"""
s2_overnight_io.py - 8 hour overnight audit for TreeHeap S2 strategy.

The task is deliberately evidence-first.  It does not assume TreeHeap works;
it tests where the signal actually lives.

Hypotheses
----------
H1. The old ep3 TreeHeap vector collapse may be caused by the post-merge
    TreeHeap output.  Raw L0/path/CMul variants may preserve more signal.
H2. Role-slot structure is more learnable from semantic+context features than
    from token-only TreeHeap output.
H3. Tensor energy only becomes meaningful if the vector/role signal is strong;
    raw non-commutativity alone is insufficient.
H4. Parent probability containers are valuable only if top-k gold coverage is
    stable across sample slices, not just in one small run.

Outputs are written incrementally so the run can be inspected while active:
  overnight_summary.json
  overnight_report.md
  role_classifier_results.csv
  tensor_ranking_results.csv
  container_stability_results.csv
  checkpoint_geometry.csv
"""

import argparse
import csv
import glob
import json
import itertools
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_DATA = "/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv"
DEFAULT_BPE = "/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model"
DEFAULT_CKPT_GLOB = "/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep*.pt"

ROLES = [
    "SUBJ",
    "ROOT",
    "OBJ",
    "IOBJ",
    "ADVMOD",
    "AMOD",
    "DET",
    "PREP",
    "POBJ",
    "COMP",
    "NEG",
    "AUX",
    "OTHER",
]
ROLE_TO_ID = {r: i for i, r in enumerate(ROLES)}

FOLD_CHILD_DEPS = {
    "NP": {"det", "amod", "nmod", "nummod", "poss", "case", "appos", "compound", "conj", "cc"},
    "VP": {"aux", "auxpass", "advmod", "prt", "neg", "oprd", "xcomp", "ccomp", "dobj", "obj", "nsubj"},
    "PP": {"pobj", "pcomp"},
    "ADJP": {"advmod", "neg", "amod", "cc", "conj"},
}
FOLD_HEADS = {
    "NP": (("NOUN", "PROPN", "PRON"), {"case", "det", "amod", "compound", "nummod"}),
    "VP": (("VERB", "AUX"), set()),
    "PP": (("ADP",), set()),
    "ADJP": (("ADJ",), {"amod"}),
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def dep_to_role(dep: str) -> str:
    if dep in {"nsubj", "nsubjpass", "csubj"}:
        return "SUBJ"
    if dep == "ROOT":
        return "ROOT"
    if dep in {"dobj", "obj"}:
        return "OBJ"
    if dep == "iobj":
        return "IOBJ"
    if dep == "advmod":
        return "ADVMOD"
    if dep == "amod":
        return "AMOD"
    if dep == "det":
        return "DET"
    if dep in {"prep", "case"}:
        return "PREP"
    if dep in {"pobj", "pcomp"}:
        return "POBJ"
    if dep in {"xcomp", "ccomp", "acomp", "attr", "oprd"}:
        return "COMP"
    if dep == "neg":
        return "NEG"
    if dep in {"aux", "auxpass"}:
        return "AUX"
    return "OTHER"


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def topk_acc(logits, y, k):
    top = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean([int(int(label) in top[i]) for i, label in enumerate(y)]))


def macro_f1(y_true, y_pred, n_classes):
    vals = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        vals.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def normalize_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-12)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class TreeHeapCkpt:
    def __init__(self, ckpt_path, bpe_path, device):
        import sentencepiece as spm

        self.ckpt_path = ckpt_path
        self.name = Path(ckpt_path).stem
        self.device = torch.device(device)
        self.d = 128
        self.td = 5
        self.vocab = 32000
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.l0 = ckpt["L0"]["weight"].float().to(self.device)
        self.t_nodes = [ckpt["t_nodes"][i]["weight"].float().to(self.device) for i in range(self.td)]
        self.tm_w = ckpt["t_merge"]["weight"].float().to(self.device)
        self.tm_b = ckpt["t_merge"]["bias"].float().to(self.device)

    def encode(self, word):
        ids = self.sp.encode(word.lower())
        return int(ids[0]) if ids else 0

    def ids(self, words):
        return torch.tensor([self.encode(w) for w in words], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def path(self, ids):
        w = torch.zeros((ids.numel(), self.d), device=self.device)
        for level in range(self.td):
            stride = max(self.vocab // (2**level), 1)
            if level == 0:
                node_idx = torch.zeros_like(ids)
            else:
                node_idx = torch.clamp(ids // stride, 0, (2**level) - 1)
            w = w + self.t_nodes[level][node_idx]
        return F.normalize(w, dim=-1)

    @torch.no_grad()
    def cmul_pre(self, ids):
        t = F.normalize(self.l0[ids], dim=-1)
        w = self.path(ids)
        left = t[:, :64] * w[:, :64] - t[:, 64:] * w[:, 64:]
        right = t[:, :64] * w[:, 64:] + t[:, 64:] * w[:, :64]
        return torch.cat([left, right], dim=-1)

    @torch.no_grad()
    def vector_batch(self, words, mode):
        ids = self.ids(words)
        if mode == "l0":
            return F.normalize(self.l0[ids], dim=-1).cpu().numpy()
        if mode == "path":
            return self.path(ids).cpu().numpy()
        if mode == "cmul":
            return self.cmul_pre(ids).cpu().numpy()
        if mode == "tree":
            return (self.cmul_pre(ids) @ self.tm_w.T + self.tm_b).cpu().numpy()
        if mode == "l0_path":
            a = F.normalize(self.l0[ids], dim=-1)
            b = self.path(ids)
            return torch.cat([a, b], dim=-1).cpu().numpy()
        raise ValueError(mode)


def load_texts(path, limit):
    texts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                texts.append(parts[1].strip())
                if len(texts) >= limit:
                    break
    return texts


def parse_docs(texts, n_jobs):
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return list(nlp.pipe(texts, batch_size=1024, n_process=n_jobs))


def extract_role_examples(docs, max_examples, ctx_window):
    examples = []
    for doc in docs:
        toks = [t for t in doc if not t.is_space and not t.is_punct and t.is_alpha]
        words = [t.text.lower() for t in toks]
        for i, t in enumerate(toks):
            role = dep_to_role(t.dep_)
            left = words[max(0, i - ctx_window) : i]
            right = words[i + 1 : i + 1 + ctx_window]
            examples.append(
                {
                    "word": t.text.lower(),
                    "role": role,
                    "label": ROLE_TO_ID[role],
                    "ctx": left + right,
                    "dep": t.dep_,
                    "pos": t.pos_,
                    "sent": doc.text[:180],
                }
            )
            if len(examples) >= max_examples:
                return examples
    return examples


def extract_svo_cases(docs, max_cases, with_adv=False):
    cases = []
    seen = set()
    for doc in docs:
        toks = [t for t in doc if not t.is_space and not t.is_punct and t.is_alpha]
        for root in toks:
            if root.dep_ != "ROOT" or root.pos_ not in {"VERB", "AUX"}:
                continue
            subjs = [c for c in root.children if c.is_alpha and dep_to_role(c.dep_) == "SUBJ"]
            objs = [c for c in root.children if c.is_alpha and dep_to_role(c.dep_) == "OBJ"]
            advs = [c for c in root.children if c.is_alpha and dep_to_role(c.dep_) == "ADVMOD"]
            if not (subjs and objs):
                continue
            if with_adv and not advs:
                continue
            if with_adv:
                toks_case = [subjs[0].text.lower(), root.text.lower(), objs[0].text.lower(), advs[0].text.lower()]
                roles = ["SUBJ", "ROOT", "OBJ", "ADVMOD"]
                kind = "SVOA4"
            else:
                toks_case = [subjs[0].text.lower(), root.text.lower(), objs[0].text.lower()]
                roles = ["SUBJ", "ROOT", "OBJ"]
                kind = "SVO3"
            key = tuple(toks_case)
            if len(set(key)) != len(key) or key in seen:
                continue
            seen.add(key)
            cases.append({"kind": kind, "tokens": toks_case, "roles": roles, "text": doc.text[:180]})
            if len(cases) >= max_cases:
                return cases
    return cases


class LinearProbe(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.net = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.net(x)


def build_features(ckpt, examples, mode, ctx_mode):
    words = [e["word"] for e in examples]
    base = ckpt.vector_batch(words, mode)
    if ctx_mode == "none":
        return base.astype(np.float32)
    ctx_vec_mode = mode if mode != "l0_path" else "l0"
    ctx_dim = ckpt.vector_batch([words[0]], ctx_vec_mode).shape[1] if words else base.shape[1]
    ctx_vecs = []
    for e in examples:
        if e["ctx"]:
            ctx = ckpt.vector_batch(e["ctx"], ctx_vec_mode)
            ctx_vecs.append(ctx.mean(axis=0))
        else:
            ctx_vecs.append(np.zeros(ctx_dim, dtype=np.float32))
    ctx = np.stack(ctx_vecs).astype(np.float32)
    if ctx_mode == "ctx_only":
        return ctx
    if ctx_mode == "token_ctx":
        return np.concatenate([base, ctx], axis=1).astype(np.float32)
    if ctx_mode == "token_minus_ctx":
        if base.shape[1] != ctx.shape[1]:
            return np.concatenate([base, ctx], axis=1).astype(np.float32)
        return np.concatenate([base, ctx, base - ctx], axis=1).astype(np.float32)
    raise ValueError(ctx_mode)


def train_probe(X, y, seed, device, epochs):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    mean = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mean) / std

    x_train = torch.tensor(Xn[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    x_test = torch.tensor(Xn[test_idx], dtype=torch.float32, device=device)
    model = LinearProbe(X.shape[1], len(ROLES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    bs = 1024
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0], device=device)
        for st in range(0, len(perm), bs):
            b = perm[st : st + bs]
            loss = F.cross_entropy(model(x_train[b]), y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = model(x_test).detach().cpu().numpy()
    yt = y[test_idx]
    pred = logits.argmax(axis=1)
    probs = softmax_np(logits)
    entropy = float(np.mean(-(probs * np.log(probs + 1e-12)).sum(axis=1)))
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "top1": float(np.mean(pred == yt)),
        "top3": topk_acc(logits, yt, 3),
        "macro_f1": macro_f1(yt, pred, len(ROLES)),
        "entropy": entropy,
    }


def geometry_audit(ckpt, words, modes):
    rows = []
    for mode in modes:
        X = normalize_np(ckpt.vector_batch(words, mode))
        sims = X @ X.T
        off = sims[~np.eye(len(words), dtype=bool)]
        rows.append(
            {
                "checkpoint": ckpt.name,
                "mode": mode,
                "n_words": len(words),
                "cos_mean": float(off.mean()),
                "cos_std": float(off.std()),
                "cos_p05": float(np.quantile(off, 0.05)),
                "cos_p95": float(np.quantile(off, 0.95)),
            }
        )
    return rows


def role_basis(dim, seed=42):
    rng = np.random.RandomState(seed)
    m = rng.randn(max(dim, len(ROLES)), max(dim, len(ROLES))).astype(np.float32)
    q, _ = np.linalg.qr(m)
    return {r: q[i, :dim].astype(np.float32) for i, r in enumerate(ROLES)}


def sent_tensor(ckpt, tokens, roles, mode, basis):
    X = ckpt.vector_batch(list(tokens), mode)
    out = np.zeros((X.shape[1], len(next(iter(basis.values())))), dtype=np.float32)
    for x, r in zip(X, roles):
        out += np.outer(x, basis[r])
    return out.reshape(-1)


def tensor_ranking(ckpt, cases, mode, seed):
    if len(cases) < 20:
        return []
    rng = random.Random(seed)
    cases = list(cases)
    rng.shuffle(cases)
    n_train = int(len(cases) * 0.65)
    train, test = cases[:n_train], cases[n_train : n_train + 80]
    basis = role_basis(32, seed)
    templates = {}
    for kind in sorted({c["kind"] for c in train}):
        arr = [sent_tensor(ckpt, c["tokens"], c["roles"], mode, basis) for c in train if c["kind"] == kind]
        if arr:
            templates[kind] = normalize_np(np.mean(np.stack(arr), axis=0))
    rows = []
    for ci, c in enumerate(test):
        if c["kind"] not in templates:
            continue
        scores = []
        words = tuple(c["tokens"])
        for perm in itertools.permutations(words):
            t = normalize_np(sent_tensor(ckpt, perm, c["roles"], mode, basis))
            scores.append((perm, cosine(t, templates[c["kind"]])))
        scores.sort(key=lambda x: -x[1])
        rank = next(i + 1 for i, (p, _) in enumerate(scores) if p == words)
        gold = next(s for p, s in scores if p == words)
        best_wrong = max(s for p, s in scores if p != words)
        rows.append(
            {
                "checkpoint": ckpt.name,
                "mode": mode,
                "kind": c["kind"],
                "case_id": ci,
                "gold_rank": rank,
                "top1": int(rank == 1),
                "top3": int(rank <= 3),
                "margin": float(gold - best_wrong),
                "tokens": " ".join(words),
            }
        )
    return rows


def extract_fold_graph(doc):
    tok = [t for t in doc if not t.is_space]
    if len(tok) < 2:
        return None
    children = {t.i: [] for t in tok}
    root = None
    for t in tok:
        if t.head.i == t.i:
            root = t.i
        elif t.head.i in children:
            children[t.head.i].append(t.i)
    if root is None:
        return None
    order = []
    stack = [(root, 0)]
    while stack:
        i, state = stack.pop()
        if state == 0:
            stack.append((i, 1))
            for c in sorted(children.get(i, []), reverse=True):
                stack.append((c, 0))
        else:
            order.append(i)
    nodes, folds, folded = {}, [], set()
    for i in order:
        t = doc[i]
        for ft, (head_pos, excluded) in FOLD_HEADS.items():
            if t.pos_ not in head_pos or t.dep_ in excluded:
                continue
            cl = [c for c in children.get(i, []) if doc[c].dep_ != "punct" and doc[c].dep_ in FOLD_CHILD_DEPS[ft]]
            if not cl:
                continue
            nodes[i] = {"type": ft, "dep": t.dep_, "pos": t.pos_}
            folds.append((i, cl, ft))
            folded.update(cl)
    edges = []
    for i in order:
        for c in children.get(i, []):
            if c in folded or c not in children or doc[c].dep_ == "punct":
                continue
            if c in nodes:
                edges.append((i, c))
    return {"n": len(tok), "nodes": nodes, "folds": folds, "edges": edges}


def container_slice(docs, seed, max_nodes):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        return {"error": str(exc)}
    graphs = []
    for doc in docs:
        g = extract_fold_graph(doc)
        if g and 2 <= len(g["nodes"]) <= max_nodes:
            graphs.append(g)
    pair_rows = []
    for gi, g in enumerate(graphs):
        nids = sorted(g["nodes"])
        gold = set(g["edges"])
        for ci in nids:
            nearest = min((p for p in nids if p != ci), key=lambda p: abs(p - ci), default=None)
            nd = abs(nearest - ci) if nearest is not None else 999
            for pi in nids:
                if pi == ci:
                    continue
                feat = np.array(
                    [
                        abs(pi - ci),
                        ci - pi,
                        1 if pi < ci else 0,
                        nd,
                        1 if pi == nearest else 0,
                        {"NP": 0, "VP": 1, "PP": 2, "ADJP": 3}.get(g["nodes"][ci]["type"], 4),
                        {"NP": 0, "VP": 1, "PP": 2, "ADJP": 3}.get(g["nodes"][pi]["type"], 4),
                    ],
                    dtype=np.float32,
                )
                pair_rows.append((feat, int((pi, ci) in gold), gi, ci, pi))
    if len(pair_rows) < 100 or sum(r[1] for r in pair_rows) < 10:
        return {"n_graphs": len(graphs), "error": "not enough pairs"}
    y = np.array([r[1] for r in pair_rows])
    idx = np.arange(len(pair_rows))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=2, n_jobs=4, random_state=seed)
    clf.fit(np.stack([pair_rows[i][0] for i in tr]), y[tr])
    by = defaultdict(list)
    for i in te:
        feat, label, gi, ci, pi = pair_rows[i]
        prob = float(clf.predict_proba(feat.reshape(1, -1))[0, 1])
        by[(gi, ci)].append((pi, prob, label))
    hits = Counter()
    total = 0
    ent = []
    for _, cands in by.items():
        cands.sort(key=lambda x: -x[1])
        if not any(x[2] for x in cands):
            continue
        total += 1
        probs = np.array([max(x[1], 1e-9) for x in cands])
        probs = probs / probs.sum()
        ent.append(float(-(probs * np.log(probs)).sum()))
        for k in [1, 2, 3, 5]:
            hits[k] += int(any(x[2] for x in cands[:k]))
    return {
        "n_graphs": len(graphs),
        "n_pairs": len(pair_rows),
        "n_eval_child_sets": total,
        "top1": hits[1] / max(total, 1),
        "top2": hits[2] / max(total, 1),
        "top3": hits[3] / max(total, 1),
        "top5": hits[5] / max(total, 1),
        "entropy": float(np.mean(ent)) if ent else None,
    }


def save_report(out_dir, summary):
    lines = []
    lines.append("# S2 Overnight Audit Report")
    lines.append("")
    lines.append(f"Created: {now()}")
    lines.append("")
    lines.append("## Hypotheses")
    lines.append("")
    for h in summary["hypotheses"]:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("## Current Decisions")
    lines.append("")
    for d in summary.get("decision_notes", []):
        lines.append(f"- {d}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in [
        "overnight_summary.json",
        "checkpoint_geometry.csv",
        "role_classifier_results.csv",
        "tensor_ranking_results.csv",
        "container_stability_results.csv",
    ]:
        lines.append(f"- `{name}`")
    Path(out_dir, "overnight_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--bpe", default=DEFAULT_BPE)
    p.add_argument("--ckpt-glob", default=DEFAULT_CKPT_GLOB)
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--sample-lines", type=int, default=80000)
    p.add_argument("--max-role-examples", type=int, default=160000)
    p.add_argument("--ctx-window", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--probe-epochs", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deadline = time.time() + args.hours * 3600

    summary = {
        "started_at": now(),
        "config": vars(args),
        "hypotheses": [
            "H1: ep3 collapse may be a post-merge vector issue; raw L0/path/CMul may preserve signal.",
            "H2: context features should improve role-slot prediction over token-only TreeHeap.",
            "H3: tensor energy needs syntax-bearing vectors; non-commutativity alone is insufficient.",
            "H4: probability containers are justified only if top-k coverage is stable across slices.",
        ],
        "events": [],
    }

    def event(msg):
        print(f"[{now()}] {msg}", flush=True)
        summary["events"].append({"time": now(), "msg": msg})
        Path(out, "overnight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    event("loading texts")
    texts = load_texts(args.data, args.sample_lines)
    event(f"loaded texts={len(texts)}")

    event("parsing docs")
    docs = parse_docs(texts, args.n_jobs)
    event(f"parsed docs={len(docs)}")

    role_examples = extract_role_examples(docs, args.max_role_examples, args.ctx_window)
    svo_cases = extract_svo_cases(docs, 500, with_adv=False) + extract_svo_cases(docs, 160, with_adv=True)
    summary["data"] = {
        "n_texts": len(texts),
        "n_docs": len(docs),
        "n_role_examples": len(role_examples),
        "role_counts": dict(Counter(e["role"] for e in role_examples)),
        "n_svo_cases": len(svo_cases),
    }
    event(f"extracted role_examples={len(role_examples)} svo_cases={len(svo_cases)}")

    ckpts = sorted(glob.glob(args.ckpt_glob))
    summary["checkpoints"] = ckpts
    event(f"found checkpoints={ckpts}")

    geometry_rows, role_rows, tensor_rows, container_rows = [], [], [], []
    vector_modes = ["l0", "path", "cmul", "tree", "l0_path"]
    ctx_modes = ["none", "ctx_only", "token_ctx", "token_minus_ctx"]

    unique_words = sorted({e["word"] for e in role_examples})[:1000]
    y = np.array([e["label"] for e in role_examples], dtype=np.int64)

    for ckpt_path in ckpts:
        if time.time() > deadline:
            event("deadline reached before next checkpoint")
            break
        event(f"loading checkpoint {ckpt_path}")
        ckpt = TreeHeapCkpt(ckpt_path, args.bpe, args.device)

        event(f"geometry audit {ckpt.name}")
        geometry_rows.extend(geometry_audit(ckpt, unique_words, vector_modes))
        write_csv(out / "checkpoint_geometry.csv", geometry_rows)

        for mode in vector_modes:
            for ctx_mode in ctx_modes:
                if mode == "l0_path" and ctx_mode in {"ctx_only", "token_minus_ctx"}:
                    continue
                if time.time() > deadline:
                    event("deadline reached during role probe")
                    break
                event(f"role probe ckpt={ckpt.name} mode={mode} ctx={ctx_mode}")
                X = build_features(ckpt, role_examples, mode, ctx_mode)
                res = train_probe(X, y, args.seed, torch.device(args.device), args.probe_epochs)
                res.update({"checkpoint": ckpt.name, "mode": mode, "ctx_mode": ctx_mode, "dim": X.shape[1]})
                role_rows.append(res)
                write_csv(out / "role_classifier_results.csv", role_rows)

        for mode in ["l0", "cmul", "tree", "l0_path"]:
            if time.time() > deadline:
                event("deadline reached during tensor ranking")
                break
            event(f"tensor ranking ckpt={ckpt.name} mode={mode}")
            tensor_rows.extend(tensor_ranking(ckpt, svo_cases, mode, args.seed))
            write_csv(out / "tensor_ranking_results.csv", tensor_rows)

    for sample in [5000, 12000, 30000, len(docs)]:
        if time.time() > deadline:
            event("deadline reached during container stability")
            break
        sample_docs = docs[: min(sample, len(docs))]
        event(f"container stability sample={len(sample_docs)}")
        res = container_slice(sample_docs, args.seed, max_nodes=10)
        res["sample_docs"] = len(sample_docs)
        container_rows.append(res)
        write_csv(out / "container_stability_results.csv", container_rows)

    summary["finished_at"] = now()
    summary["runtime_seconds"] = round(time.time() - (deadline - args.hours * 3600), 3)
    summary["result_counts"] = {
        "geometry_rows": len(geometry_rows),
        "role_rows": len(role_rows),
        "tensor_rows": len(tensor_rows),
        "container_rows": len(container_rows),
    }

    decision_notes = []
    if role_rows:
        best_role = max(role_rows, key=lambda r: r.get("top3", 0))
        decision_notes.append(
            f"Best role probe: {best_role['checkpoint']} {best_role['mode']} {best_role['ctx_mode']} "
            f"top1={best_role['top1']:.3f} top3={best_role['top3']:.3f} macro_f1={best_role['macro_f1']:.3f}."
        )
    tree_geom = [r for r in geometry_rows if r["mode"] == "tree"]
    if tree_geom:
        worst = max(tree_geom, key=lambda r: r["cos_mean"])
        decision_notes.append(
            f"Tree output geometry remains {'collapsed' if worst['cos_mean'] > 0.9 else 'spread'}; "
            f"max tree offdiag cosine mean={worst['cos_mean']:.4f}."
        )
    if container_rows:
        last = container_rows[-1]
        if "top3" in last:
            decision_notes.append(
                f"Container stability latest sample={last['sample_docs']} top1={last['top1']:.3f} "
                f"top3={last['top3']:.3f} top5={last['top5']:.3f}."
            )
    summary["decision_notes"] = decision_notes
    Path(out, "overnight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report(out, summary)
    event("done")


if __name__ == "__main__":
    main()
