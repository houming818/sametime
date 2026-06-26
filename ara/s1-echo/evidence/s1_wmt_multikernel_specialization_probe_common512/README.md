# S1 WMT multi-kernel specialization probe: common-token run

Controlled structural perturbation tasks over real WMT English SentencePiece
short sequences, with a smaller common-token vocabulary.

## Claim

`S1-MK-C01`: structural perturbation tasks can push a TreeHeap kernel bank
toward task-dependent specialization.

## Run

```text
host = io.grepcode.cn
device = cuda
dataset = /mnt/nas/datasets/wmt17/train.zh-en
spm = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 4000
length = 4..8
vocab limit including PAD/MASK = 513
epochs = 60
```

## Result

| Model | Params | OOD mean exact |
|---|---:|---:|
| `single_kernel_treeheap` | 1,286,921 | 0.1275 |
| `multi_kernel_treeheap` | 1,584,140 | 0.1420 |

OOD task exact:

| Task | Single | Multi |
|---|---:|---:|
| `echo` | 0.0300 | 0.0375 |
| `mask_restore` | 0.0050 | 0.0025 |
| `left_query` | 0.1475 | 0.1775 |
| `right_query` | 0.4525 | 0.4925 |
| `mirror` | 0.0025 | 0.0000 |

Specialization signals:

```text
task_argmax_kernel = {
  echo: 0,
  mask_restore: 1,
  left_query: 0,
  right_query: 3,
  mirror: 2
}
unique_argmax_kernels = 4
max_ood_ablation_exact_drop = 0.3050
drop_kernel_0 left_query = 0.1775
drop_kernel_3 right_query = 0.3050
```

## Decision

```text
S1-MK-C01 -> open / mixed pilot
```

The common-token run strengthens the differentiation evidence, especially for
left/right subheap queries. It still fails the pass gate because OOD mean exact
is low and the multi-kernel improvement over single-kernel is small.
