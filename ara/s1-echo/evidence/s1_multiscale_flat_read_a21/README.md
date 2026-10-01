# A21 Evidence

The smoke and formal runs compare four READ surfaces over one frozen A20
`evidence_fold`, dimension-16 encoder.

- smoke: io task 645, 20 epochs;
- formal: io task 646, 200 epochs;
- counts SHA-256: `c4a2a0aad79a4357cbbd6e1591a8a31a873dfa2918fb62b8bda2d5925917f864`;
- source checkpoint SHA-256: `659ac6ce527f23e310f503433f1a57bf4532d611adce20439282bb8c68b41e07`;
- experiment source SHA-256: `a6f9c3f3339b9ab4019405127cb86c639c8e63e2dcfd65446b6cf22eb6bae268`.

Formal Test NLL:

```text
root_unfold  5.4774518013
leaf_only    5.1357045174
flat_coarse  5.1196112633
flat_all     5.1182570457
```

Both registered claims passed. Scope remains a frozen conditional context-field
codec; this is not real-sequence or translation evidence.

