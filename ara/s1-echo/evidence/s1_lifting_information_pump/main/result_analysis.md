# Result Analysis

The full registered decision is **not supported** because P3 missed its root
token-drop threshold (`0.06958 < 0.10`).  This must not be rewritten as a full
pass after seeing that root-zero block exact fell to zero.

The experiment nevertheless supports a narrower mechanism claim.  The bounded
nonlinear lifting kernel closes through six tested depths with maximum FP32
error `3.70e-6`.  Native state reconstruction MSE is `3.14e-14`, and token plus
16-token block echo are both exactly `1.0`.  Decoding starts from root and
recursively consumes addressed details; there is no leaf or parent-1 READ
bypass.

Every detail resolution is causal.  Cross-sample detail replacement at depths
1..4 drops token accuracy by `0.49878/0.25000/0.12354/0.06323`.  Root zero and
root replacement reduce full-block exact to `0/0.0078125`, although most
individual tokens remain nearest-neighbor readable.

The natural, unmasked next-token objective also provides genuine inductive
pressure.  Learned predictor NLL is `8.03468` versus frozen `8.06377`, passing
the registered `0.02` gain.  Predictor gradient/delta are non-zero, root shuffle
adds `0.21359` NLL, and breaking each of the four recursive pair depths adds
`0.03221/0.03838/0.04324/0.05634` NLL.

Therefore the earlier first-parent shortcut is repaired at the information-flow
level.  What remains open is meaning: the pump has learned a useful private
root code, not a demonstrated semantic hierarchy.
