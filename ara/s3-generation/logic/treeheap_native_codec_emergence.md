# Native TreeHeap Codec Emergence

Date: 2026-07-14
Status: partial support / continuous detail protocol learned / root-plus and finite-code gates rejected
Claim: `S3-TREEHEAP-CODEC-C01`
Predict: `P-S3-TREEHEAP-CODEC-01`

## Question

The multiresolution experiment proved that addressed TreeHeap detail codes can
reconstruct states from a frozen pretrained embedding space. It did not prove
that TreeHeap can create its own encoder/decoder protocol from raw tokens.

This experiment removes the frozen encoder. Starting from random parameters,
it jointly learns:

```text
WRITE -> FOLD + DETAIL -> UNFOLD -> READ
```

using only surface token echo cross-entropy. There are no syntax, route, depth,
category, embedding, or state-reconstruction labels.

## Operators

For token `x_i`, the shared leaf writer is:

```math
h_i = W_\alpha(x_i).
```

The first unrestricted design allowed the detail path to copy the common
component and made the final root irrelevant. The revised analysis kernels are
therefore a symmetric/antisymmetric filter pair:

```math
p_i = D_\theta\left(\frac{h_{2i}+h_{2i+1}}{\sqrt{2}}\right),
```

```math
d_i = Q_\phi\left(\frac{h_{2i}-h_{2i+1}}{\sqrt{2}}\right),
\qquad d_i\in\mathbb{R}^k.
```

The shared synthesis kernel recursively decodes:

```math
(\hat h_{2i},\hat h_{2i+1}) = U_\psi(\hat p_i,d_i).
```

The tied surface reader gives a token probability bucket:

```math
P(x_i\mid H)=\operatorname{softmax}(G_\omega(\hat h_i)).
```

`D`, `Q`, and `U` are TreeHeap convolutions: local receptive fields, shared
parameters, execution at every valid internal address, and recursive reuse
across depth. The decoder receives only one root and addressed details. It has
no token, leaf, or encoder-attention bypass.

## Training Protocol

```text
data                 real Chinese pretraining token blocks
tokenizer            existing 16,000-piece SentencePiece model
length               64 tokens
state dimension      128
root code            128 binary values
detail code          8 binary values per internal address
stored code          128 + 63*8 = 632 bits
raw token upper bound 64*ceil(log2(16000)) = 896 bits
loss                 token echo cross-entropy only
seeds                71411 / 71412 / 71413
```

First run a real-data smoke. Only after finite gradients, decreasing held-out
NLL, and valid evidence output are confirmed may the one-million-block queue
start.

## Predictions

```text
P1  Held-out token top-1 >= 0.90 and 64-token sequence exact >= 0.10.
P2  Shifting details to wrong addresses lowers token top-1 by >= 0.50.
P3  Zeroing all details lowers token top-1 by >= 0.50.
P4  Replacing details with another sample lowers token top-1 by >= 0.50.
P5  The shared codec trained only at length 64 obtains token top-1 >= 0.80
    when recursively evaluated at unseen length 32.
P6  WRITE, FOLD, DETAIL, UNFOLD, and READ all receive finite non-zero
    gradients from the same surface token loss.
P7  Zeroing the root lowers held-out token top-1 by >= 0.10.
```

P7 is a hard gate because this claim names a root-plus-detail codec. It does not
require the root to contain human-readable semantics; it only requires the
coarse channel to carry information unavailable from the difference channels.

## Preregistered Decision

Support this codec claim only if P1 through P7 pass in all three seeds. If echo
works but interventions do not, reject the TreeHeap mechanism interpretation
as a bypass. If length-32 fails, narrow the result to a fixed-depth protocol.

Even a full pass proves only a learned reversible protocol over raw token
sequences. It does not prove semantic categories, world knowledge,
consciousness, useful compression in bits, or masked-prediction reasoning.

## Failed First Smoke

The unrestricted first implementation used `FOLD([L,R])` and
`DETAIL([L,R,P])`. After 20,000 blocks it reached held-out token top-1
`0.9955`, but root-zero top-1 was also `0.9956`. The detail path had learned a
complete bypass. That implementation is rejected for this claim and retained
as diagnostic evidence under `evidence/s3_treeheap_native_codec/smoke_v1/`.

The symmetric/antisymmetric second implementation also reached near-perfect
echo while ignoring the root. A continuous 32D difference code can hash a
finite ordered token pair, so dimension count was not an information bound.
That run is retained under `smoke_v2/`. The third design uses straight-through
binary root and detail codes, giving an explicit 632-bit container. This is a
finite code-capacity test, not yet an entropy-coded file compressor.

## Results

All three runs used real Chinese pretraining blocks on `io`. These are
single-seed mechanism tests, not scale or architecture comparisons.

| Run | Code | Valid token top-1 | Valid exact | Detail shift top-1 | Detail zero top-1 | Root zero top-1 | OOD-32 top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | unrestricted continuous | 0.9955 | 0.7715 | 0.0024 | 0.0000 | 0.9956 | 0.9957 |
| v2 | symmetric continuous | 0.9915 | 0.5918 | 0.0014 | 0.0349 | 0.9889 | 0.9886 |
| v3 | 632-bit straight-through binary | 0.2073 | 0.0000 | 0.0121 | 0.0613 | 0.1882 | 0.2062 |

Every WRITE/FOLD/DETAIL/UNFOLD/READ operator received a finite non-zero
gradient in all runs. Therefore the surface echo loss reached the entire
recursive codec. In v1 and v2, shifting or deleting addressed details destroyed
the output, so a learned address-sensitive encoder/decoder protocol genuinely
emerged. But zeroing the root did essentially nothing. The protocol was a
detail-only codec, not the claimed root-plus-detail codec.

The v3 finite container removed the false assumption that `32D` means `32
bits`. Its addressed binary details were causal, and zeroing the root reduced
top-1 by about 1.9 percentage points, but reconstruction reached only `0.2073`
token top-1 and no exact sequences. Only P6 passed. The current hard binary
codec therefore does not meet the fidelity gate.

## Claim Decision

`S3-TREEHEAP-CODEC-C01` receives **partial support with a narrowed boundary**:

1. Supported: raw-token echo cross-entropy can jointly train shared recursive
   TreeHeap WRITE/FOLD/DETAIL/UNFOLD/READ kernels into an address-sensitive
   continuous encoder/decoder protocol.
2. Rejected: the tested echo objective does not make the root a necessary
   global code; v1 and v2 route almost all information through details.
3. Not supported: the present 632-bit straight-through implementation is not a
   high-fidelity finite codec.

The experiment also explains why increasing float-vector dimensionality alone
cannot establish compression. The learned embedding table provides an external
reference system, and a continuous difference code can identify finite token
pairs with arbitrarily fine precision. Future compression claims must measure
quantized bits or rate-distortion, not activation dimensions.

## Next Claim, Not Part Of This Proof

Do not spend the next run merely tuning echo until P7 passes. Echo can be solved
by reversible local detail codes and therefore cannot by itself demand a useful
global root. The next experiment should train masked-span or next-span
prediction so that the model must infer information absent from its input.
Root and subheap ablations can then test whether the hierarchy carries reusable
predictive structure rather than only an addressable copy code.
