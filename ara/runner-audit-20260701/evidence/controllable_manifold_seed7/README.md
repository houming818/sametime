# S1 controllable manifold probe

Claim: `S1-MANIFOLD-C01`

This CPU-only toy proof sweeps two TreeHeap kernel controls:

- `relation_weight`: how strongly the fold kernel trusts the latent relation field.
- `order_weight`: how strongly the fold kernel trusts linear neighborhood/order.

The task uses four short sentence cases and asks whether predicted merge blocks
match weak gold blocks such as `the cat`, `is running`, `a car`, and the larger
predicate block.

## Result

- low-control mean F1: `0.0830`
- best mean F1: `0.8148`
- high-sum-control mean F1: `0.8148`
- diagonal gain: `0.7318`
- product cells: `22`
- pilot pass: `True`

Best cell:

```json
{
  "relation_weight": 4.0,
  "order_weight": 0.0,
  "mean_f1": 0.8147727272727273,
  "std_f1": 0.0,
  "exact_rate": 0.0
}
```

## Boundary

This is not a language-understanding proof and not a WMT proof. It only shows
whether fold quality can be controlled by kernel knobs on a transparent toy
relation field.
