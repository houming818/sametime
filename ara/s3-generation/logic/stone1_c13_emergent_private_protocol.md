# STONE-1 C13: emergent TreeHeap private protocol

Status: registered; experiment pending

## Prior evidence

This claim does not reopen FOLD/UNFOLD echo. `S1-LIFT-PUMP-C01` already
established depth-1..6 algebraic closure, native state MSE `3.14e-14`, and
token/block echo `1.0/1.0`. `S2-LIFT-WMT-C01` retained closure at `1.73e-14`
while recursive root/detail reads were causal on WMT.

C12 tested a different mechanism: a frozen source encoder followed by a
one-shot MLP synthesis of target root/detail/gate state. It was fast but its
source, address, and detail interventions were neutral and generation collapsed.
The failure therefore belongs to the one-shot state synthesizer, not to the
inverse lifting algebra.

## Claim

`S3-STONE1-EMERGENT-PROTOCOL-C13`:

> Given the already closed TreeHeap codec, joint end-to-end training of the
> source encoder and repeated multi-head subheap communication kernels can form
> a task-private protocol without semantic labels or target-H supervision. The
> resulting target H_state should causally depend on source content, source
> addresses, more than one communication head, and more than one update round.

## Model

The source is encoded by the existing lifting TreeHeap:

\[
H_x=E_{\theta_E}(x).
\]

The target state starts from source root plus zero addressed details. For each
communication round and each target depth, a target parent queries source
subheaps at the same resolution:

\[
A_h=\operatorname{softmax}\left(\frac{Q_h(T)K_h(S)^\top}{\sqrt d}\right),
\qquad C_h=A_hV_h(S).
\]

Independent heads are concatenated and projected. A shared local kernel then
applies bounded residual updates to target root/details:

\[
H_y^{k+1}=H_y^k\oplus
\alpha\,\Delta_{\theta_k}(H_y^k,C_1,\ldots,C_m).
\]

After every round, the fixed encoder `unfold` reconstructs all target levels.
After the final round it reconstructs 128 addressed leaves and a shared token
readout produces logits. No predicted token is fed back, no target token is
provided to the decoder, and no internal target state is supervised.

The training objective is deliberately small:

\[
L=L_{CE}(y\mid x)+\lambda\max(0,m+L_{CE}(y\mid x)-L_{CE}(y\mid\pi(x))).
\]

The second term only requires the matched source to beat a cross-sample source;
it does not prescribe what any root, detail, address, or head means.

## Controlled smoke

- initialize from the C11 checkpoint;
- unfreeze the existing source encoder;
- use the same adjacent raw blocks and variable source lengths;
- fixed 128-piece target, no target teacher forcing;
- 3 communication rounds, 4 independent heads;
- compare initial/final NLL and the already recorded C12 smoke;
- audit source shuffle, empty source, sibling-address mirror, each head, each
  round, and each target detail depth;
- report free-generation distinct-2/4, repeated-run and unique outputs.

## Predictions

1. Validation NLL falls by at least `0.20` and remains finite.
2. Source shuffle and empty source each add at least `0.05` NLL.
3. Mirroring source sibling addresses adds at least `0.01` NLL.
4. Removing at least two different heads adds at least `0.005` NLL each.
5. Removing the last communication round adds at least `0.01` NLL.
6. Zeroing at least two target-detail depths adds at least `0.005` NLL each.
7. Mean distinct-2 and distinct-4 both exceed C12 UNFOLD (`0.0079/0.0080`),
   maximum repeated run is below 128, and unique-output fraction exceeds 0.25.
8. Encoder parameter delta is non-zero and FOLD/UNFOLD closure remains below
   `1e-5` maximum absolute error.

## Falsification and boundary

Reject the private-protocol claim if the model lowers CE while source/address,
head, round, and detail interventions remain neutral. Non-zero tensors,
attention entropy, or a falling training loss alone are not evidence of a
protocol. A passing smoke would establish a causal learned communication
mechanism, not dialogue, translation quality, semantic hierarchy,
consciousness, or superiority over Transformer.

