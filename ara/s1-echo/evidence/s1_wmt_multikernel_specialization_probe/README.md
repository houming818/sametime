# S1 WMT multi-kernel specialization probe

Controlled structural perturbation tasks over real WMT English SentencePiece
short sequences.

## Claim

`S1-MK-C01`: structural perturbation tasks can push a TreeHeap kernel bank
toward task-dependent specialization, analogous to the opportunity for
Transformer multi-head attention to differentiate.

## Run

```text
host = io.grepcode.cn
device = cuda
dataset = /mnt/nas/datasets/wmt17/train.zh-en
spm = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 4000
length = 4..8
vocab limit including PAD/MASK = 2049
epochs = 28
```

## Result

| Model | Params | OOD mean exact |
|---|---:|---:|
| `single_kernel_treeheap` | 4,641,545 | 0.0495 |
| `multi_kernel_treeheap` | 4,938,764 | 0.0600 |

Specialization signals:

```text
task_argmax_kernel = {
  echo: 2,
  mask_restore: 1,
  left_query: 3,
  right_query: 3,
  mirror: 0
}
unique_argmax_kernels = 4
max_ood_ablation_exact_drop = 0.1100
```

## Decision

```text
S1-MK-C01 -> open / mixed pilot
```

The gate/ablation evidence supports kernel differentiation. The task accuracy
is too low to support the full capability claim.
