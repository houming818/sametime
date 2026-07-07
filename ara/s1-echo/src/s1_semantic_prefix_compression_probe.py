#!/usr/bin/env python3
"""SPR-047 semantic prefix compression proof.

Question:
  Can a TreeHeap-like semantic prefix encode a leaf token through class
  ancestors so that a rule learned at an internal semantic node transfers to an
  unseen leaf?

This is a controlled toy proof, not a language understanding claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def dot(w: list[float], xs: list[int]) -> float:
    return sum(w[i] for i in xs)


class SparseLogistic:
    def __init__(self, n_features: int, *, lr: float = 0.2, l2: float = 1e-4, epochs: int = 600) -> None:
        self.w = [0.0 for _ in range(n_features)]
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs

    def fit(self, rows: list[tuple[list[int], int]]) -> list[dict[str, float]]:
        trace: list[dict[str, float]] = []
        for epoch in range(1, self.epochs + 1):
            loss = 0.0
            correct = 0
            for xs, y in rows:
                p = sigmoid(self.b + dot(self.w, xs))
                loss += -(y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12))
                g = p - y
                self.b -= self.lr * g
                for i in xs:
                    self.w[i] -= self.lr * (g + self.l2 * self.w[i])
                correct += int((p >= 0.5) == bool(y))
            if epoch in {1, 2, 5, 10, 50, 100, self.epochs}:
                trace.append(
                    {
                        "epoch": epoch,
                        "loss": loss / max(1, len(rows)),
                        "acc": correct / max(1, len(rows)),
                    }
                )
        return trace

    def predict_proba(self, xs: list[int]) -> float:
        return sigmoid(self.b + dot(self.w, xs))


def build_world() -> dict:
    # Object leaves are encoded as semantic prefix paths. In a TreeHeap, these
    # paths would be internal semantic nodes above token leaves.
    objects = {
        "rice": ["entity", "consumable", "food"],
        "noodle": ["entity", "consumable", "food"],
        "apple": ["entity", "consumable", "food"],
        "amoxicillin": ["entity", "consumable", "medicine"],
        "ibuprofen": ["entity", "consumable", "medicine"],
        "water": ["entity", "drinkable", "beverage"],
        "milk": ["entity", "drinkable", "beverage"],
        "shirt": ["entity", "wearable", "clothing"],
        "hoodie": ["entity", "wearable", "clothing"],
        "car": ["entity", "drivable", "vehicle"],
        "truck": ["entity", "drivable", "vehicle"],
        "paris": ["entity", "visitable", "location"],
        "museum": ["entity", "visitable", "location"],
    }
    verbs = {
        "eat": {"consumable"},
        "take": {"medicine"},
        "drink": {"drinkable"},
        "wear": {"wearable"},
        "drive": {"drivable"},
        "visit": {"visitable"},
    }
    heldout = {
        ("eat", "amoxicillin"),
        ("eat", "ibuprofen"),
        ("drink", "milk"),
        ("wear", "hoodie"),
        ("drive", "truck"),
        ("visit", "museum"),
    }
    return {"objects": objects, "verbs": verbs, "heldout": heldout}


def label(world: dict, verb: str, obj: str) -> int:
    accepted = world["verbs"][verb]
    path = set(world["objects"][obj])
    return int(bool(accepted & path))


def make_rows(world: dict) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    rows: list[tuple[str, str, int]] = []
    test: list[tuple[str, str, int]] = []
    for verb in world["verbs"]:
        for obj in world["objects"]:
            y = label(world, verb, obj)
            item = (verb, obj, y)
            if (verb, obj) in world["heldout"]:
                test.append(item)
            else:
                rows.append(item)
    # Add several diagnostic negative OOD pairs that share surface familiarity
    # but not the correct semantic prefix.
    for item in [
        ("eat", "water", label(world, "eat", "water")),
        ("drink", "rice", label(world, "drink", "rice")),
        ("wear", "car", label(world, "wear", "car")),
        ("drive", "shirt", label(world, "drive", "shirt")),
    ]:
        if item not in test:
            test.append(item)
    return rows, test


class Featurizer:
    def __init__(self, world: dict, mode: str, train_rows: list[tuple[str, str, int]]) -> None:
        self.world = world
        self.mode = mode
        keys: list[tuple[str, str]] = []
        if mode == "pair":
            keys = [("pair", f"{v}|{o}") for v, o, _ in train_rows]
        elif mode == "token_additive":
            keys = [("verb", v) for v in world["verbs"]] + [("obj", o) for o in world["objects"]]
        elif mode == "semantic_prefix":
            classes = sorted({c for path in world["objects"].values() for c in path})
            keys = [("verb_class", f"{v}|{c}") for v in world["verbs"] for c in classes]
            keys += [("bias_verb", v) for v in world["verbs"]]
        else:
            raise ValueError(mode)
        self.index = {k: i for i, k in enumerate(keys)}

    def encode(self, verb: str, obj: str) -> list[int]:
        out: list[int] = []
        if self.mode == "pair":
            key = ("pair", f"{verb}|{obj}")
            if key in self.index:
                out.append(self.index[key])
        elif self.mode == "token_additive":
            out.append(self.index[("verb", verb)])
            out.append(self.index[("obj", obj)])
        else:
            out.append(self.index[("bias_verb", verb)])
            for cls in self.world["objects"][obj]:
                out.append(self.index[("verb_class", f"{verb}|{cls}")])
        return out


def eval_model(model: SparseLogistic, feat: Featurizer, rows: list[tuple[str, str, int]]) -> dict:
    examples = []
    correct = 0
    for verb, obj, y in rows:
        p = model.predict_proba(feat.encode(verb, obj))
        pred = int(p >= 0.5)
        correct += int(pred == y)
        examples.append(
            {
                "verb": verb,
                "object": obj,
                "path": feat.world["objects"][obj],
                "gold": y,
                "prob": p,
                "pred": pred,
            }
        )
    return {"acc": correct / max(1, len(rows)), "examples": examples}


def eval_pair_memory(train_rows: list[tuple[str, str, int]], rows: list[tuple[str, str, int]]) -> dict:
    positives = {(v, o) for v, o, y in train_rows if y == 1}
    examples = []
    correct = 0
    for verb, obj, y in rows:
        pred = int((verb, obj) in positives)
        correct += int(pred == y)
        examples.append({"verb": verb, "object": obj, "gold": y, "pred": pred})
    return {"acc": correct / max(1, len(rows)), "examples": examples}


def jsonable_world(world: dict) -> dict:
    return {
        "objects": world["objects"],
        "verbs": {verb: sorted(classes) for verb, classes in world["verbs"].items()},
        "heldout": [list(item) for item in sorted(world["heldout"])],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_semantic_prefix_compression_probe")
    args = ap.parse_args()

    start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    world = build_world()
    train_rows, test_rows = make_rows(world)

    results = {
        "pair_memory": eval_pair_memory(train_rows, test_rows),
        "models": {},
    }
    traces = {}
    for mode in ["pair", "token_additive", "semantic_prefix"]:
        feat = Featurizer(world, mode, train_rows)
        model = SparseLogistic(len(feat.index), lr=0.15, epochs=800)
        encoded_train = [(feat.encode(v, o), y) for v, o, y in train_rows]
        traces[mode] = model.fit(encoded_train)
        results["models"][mode] = {
            "train": eval_model(model, feat, train_rows),
            "test": eval_model(model, feat, test_rows),
            "n_features": len(feat.index),
        }

    semantic_examples = results["models"]["semantic_prefix"]["test"]["examples"]
    target = next(e for e in semantic_examples if e["verb"] == "eat" and e["object"] == "amoxicillin")
    pair_target = next(e for e in results["pair_memory"]["examples"] if e["verb"] == "eat" and e["object"] == "amoxicillin")

    summary = {
        "claim": "S1-SEMANTIC-PREFIX-C01",
        "predict": "P-S1-SEMANTIC-PREFIX01",
        "purpose": "test whether semantic prefix compression supports deductive transfer to unseen verb-object leaves",
        "world": jsonable_world(world),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "metrics": {
            "pair_memory_test_acc": results["pair_memory"]["acc"],
            "pair_logistic_test_acc": results["models"]["pair"]["test"]["acc"],
            "token_additive_test_acc": results["models"]["token_additive"]["test"]["acc"],
            "semantic_prefix_test_acc": results["models"]["semantic_prefix"]["test"]["acc"],
            "semantic_prefix_train_acc": results["models"]["semantic_prefix"]["train"]["acc"],
        },
        "key_case": {
            "question": "Can eat + amoxicillin be accepted without that pair in training?",
            "pair_memory": pair_target,
            "semantic_prefix": target,
            "deduction": [
                "amoxicillin -> medicine -> consumable",
                "eat accepts consumable",
                "therefore eat + amoxicillin should be accepted",
            ],
        },
        "results": results,
        "traces": traces,
        "pass_checks": {
            "semantic_prefix_solves_all_heldout": results["models"]["semantic_prefix"]["test"]["acc"] == 1.0,
            "pair_memory_fails_unseen_positive": pair_target["pred"] == 0 and target["gold"] == 1,
            "eat_amoxicillin_transfers": target["pred"] == 1 and target["prob"] >= 0.5,
        },
        "pilot_pass": (
            results["models"]["semantic_prefix"]["test"]["acc"] == 1.0
            and pair_target["pred"] == 0
            and target["pred"] == 1
        ),
        "limits": [
            "toy ontology is provided, not learned from raw corpus",
            "semantic prefix paths are supervised",
            "does not prove natural language semantics",
            "does not prove WMT translation",
            "next proof must learn or induce prefix nodes from data",
        ],
        "elapsed_sec": time.time() - start,
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "trace.jsonl").write_text(
        "\n".join(json.dumps({"mode": k, "trace": v}, ensure_ascii=False) for k, v in traces.items()) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# S1 Semantic Prefix Compression Probe\n\n"
        "A controlled toy proof for semantic prefix compression and deductive transfer.\n\n"
        "Key case: `amoxicillin -> medicine -> consumable`; `eat` accepts `consumable`; "
        "therefore `eat + amoxicillin` should be accepted although the pair is held out.\n\n"
        "See `summary.json` for metrics and boundaries.\n",
        encoding="utf-8",
    )
    (out / "command.sh").write_text(
        f"python3 ara/s1-echo/src/s1_semantic_prefix_compression_probe.py --out {args.out}\n",
        encoding="utf-8",
    )
    print(json.dumps({"pilot_pass": summary["pilot_pass"], "metrics": summary["metrics"], "key_case": summary["key_case"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
