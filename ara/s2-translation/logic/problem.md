# S2 Problem Statement

## Core Research Question

> Can we replace Transformer's O(N²) attention over tokens with a compact, explicit structural representation (Fold Stack) that emerges from the geometry of TreeHeap 128D vectors?

## Hypothesis

Natural language structure is a **low-dimensional manifold** embedded in a high-dimensional combinatorial space. Grammar occupies only ~200 prototype patterns + graph assembly, not a 32K token search space.

## Three Sub-Questions

1. **Phrase Collapse**: Does phrase-level structure decompose into a small number of prototypes? (PP≈3, VP≈44, NP≈150)
2. **Semantic→Structure**: Can the L1 128D semantic vector predict the target sentence's structure?
3. **Graph Assembly**: Can the connections between phrase nodes be determined by their geometric relationships in TreeHeap space?

## Key Constraints

- No external dependency labels (spaCy, POS tags, syntax trees) used for FEATURE CONSTRUCTION
- Only TreeHeap 128D vectors + tree paths as input
- Pure mathematical transformations (no classification, no training)
- Evaluation reference can use external parsers but not for model input

## Gap

| Component | Current UAS | Target | Gap Analysis |
|-----------|------------|--------|--------------|
| Fold Discovery | F1=87-96% | ✅ | Not the bottleneck |
| Graph Assembly | 56% (nearest) | 70% (oracle) | Last 14%: PP attachment, non-local dependencies |
| Tensor Method | Inconclusive | — | Requires better 128D vectors (more TreeHeap training) |

## Why It Matters

Transformer uses ~90M params, GPU-dependent, O(N²). Fold Stack with TreeHeap would use:
- 1.4M params (64x fewer)
- CPU inference (151 sentences/s, 50x faster)
- Structure emerges from geometry, not learned attention

Reference: S2 blogs, `ara/trace/research_dag.yaml`
