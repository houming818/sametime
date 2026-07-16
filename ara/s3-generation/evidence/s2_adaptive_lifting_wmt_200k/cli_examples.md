# CLI Checkpoint Spot Check

These are hand-written English inputs, not a scored or cherry-picked test set.
All three models use the same WMT-massive tokenizer and greedy decoding with a
40-piece output limit.

| English | Flat sequence | Old pump | Learned-update pump |
|---|---|---|---|
| The cat is eating some food. | 食物的饮食是食物。 | 猫是吃的食物,但一些食物。 | 吃头吃美味的食物。 |
| I arrived home at seven o'clock. | 我回家七个七个七个七个七个七个七个七个七个。 | 嗯,我再找我。 | I七家七七七七七个七个家庭房子。 |
| All the windows of that house are open. | 所有主都是主窗户,都得直直直。 | 窗体的所有窗户都都打开了。 | 珺梅看来,你属于这个男人的窗户。 |
| The earth is round because gravity pulls matter toward its center. | ⁇ 是因为有其反正性的圆满是多么深感。 | 地球的圆形,因为地球的阴蒂的山脉。 | 地球是,因为是世界上最先进的组成部分。 |
| Modern human-machine interaction is more than pressing a button. | 引发的现代的女孩儿,更紧迫。 | ⁇ 菜是一种更多的超过更多的非 ⁇ 源。 | 现代人类的交互作用 — — — — — ⁇ 按一下按钮,以防范片。 |
| A wide range of restaurants and cafes can be found near the hotel. | 各种各样的咖啡馆和咖啡馆可以享用附近的景点。 | 客人可以在餐厅供应众多咖啡厅内享用免费餐厅和咖啡馆。 | 酒店还提供各种餐厅和餐厅供应各种当地咖啡馆。 |

The spot check agrees with aggregate evaluation: learned update is a modest
mean improvement, not per-sentence dominance. Old pump is clearly better on
the window sentence; learned update carries more of the subject on the modern
human-machine sentence; all models still repeat, omit relations, and sometimes
emit invalid or inappropriate words.

## CLI

```bash
python3 ara/s3-generation/src/s2_adaptive_lifting_translate.py \
  --checkpoint /mnt/nas/ara/s3-generation/evidence/s2_adaptive_lifting_wmt_200k/checkpoint_learned_update.pt \
  --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model \
  --text "The cat is eating some food."
```

Without `--text`, the CLI accepts one English sentence per stdin line. Replace
the checkpoint filename with `checkpoint_old_recursive.pt` or
`checkpoint_flat_seq.pt` for controlled comparison.
