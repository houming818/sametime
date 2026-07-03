# S1 WMT Canonical Echo Probe

Claim: `S1-CANON-WMT-C01`
Predict: `P-S1-CANON-WMT01`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
treeheap_ood_positive_distance = 0.345768
treeheap_ood_negative_distance = 0.992941
treeheap_ood_margin            = 0.647173
treeheap_ood_retrieval@1       = 0.630000
treeheap_ood_retrieval@5       = 0.819500
treeheap_ood_entropy           = 4.044313
treeheap_ood_en_echo_token     = 1.000000
treeheap_ood_zh_echo_token     = 0.998135

bow_ood_margin                 = 0.593911
bow_ood_retrieval@1            = 0.628500
untrained_treeheap_entropy     = 6.973098
```

## Readable TreeHeap OOD Examples

- paired_cos=0.6428, top1=True
  - EN: He had already begun to expel Jews, including by placing them forcibly on ships and sending them to various destinations in the Mediterranean and across the Atlantic.
  - ZH: 他已经开始驱逐犹太人，手段包括强行把他们驱赶上船，运往地中海和大西洋的各个目的地。
  - EN echo: he had already begun to <unk> jews, including by placing them forcibly on ships and sending them to various <unk> in the mediterranean and across the atlantic.
  - ZH echo: 他已经开始驱逐犹太人，手段包括强行把他们驱赶上船，运往地中海和大西洋的各个目的地。
- paired_cos=0.5538, top1=True
  - EN: Bitcoin is a slow, energy-inefficient dinosaur that will never be able to process transactions as quickly or inexpensively as an Excel spreadsheet.
  - ZH: 比特币是一只行动迟缓、能源效率低下的恐龙，永远无法像Excel电子表格那样迅速廉价地处理交易。
  - EN echo: bitcoin is a slow, energy - inefficient <unk> that will never be able to process transactions as quickly or <unk> as an <unk> <unk>.
  - ZH echo: 比特币是一只行动迟缓、能源效率低下的恐龙，永远无法像策电子表格那样迅速廉价地处理交易。
- paired_cos=0.6436, top1=True
  - EN: Other financial instruments, such as “green” stock indices and “green” bonds, can help reallocate investment to sectors that support environmentally sustainable growth.
  - ZH: 其他金融工具，比如“绿色”股票指数和“绿色”债券，有助于将投资配置给支持环境可持续增长的部门。
  - EN echo: other financial instruments, such as green stock indices and green bonds, can help <unk> investment to sectors that support environmentally sustainable growth.
  - ZH echo: 其他金融工具，比如“绿色”股票指数和“绿色”债券，有助于将投资配置给支持环境可持续增长的部门。
- paired_cos=0.5795, top1=True
  - EN: The tradable sector is expanding and is not dependent on leverage to generate aggregate demand.
  - ZH: 可贸易部门正在扩张，不需要依赖杠杆就能提振总需求。
  - EN echo: the tradable sector is expanding and is not dependent on leverage to generate aggregate demand.
  - ZH echo: 可贸易部门正在扩张，不需要依赖杠杆就能提振总需求。
- paired_cos=0.4299, top1=False
  - EN: As in Nigeria, vaccination delays will be highly detrimental for neighboring countries.
  - ZH: 与尼日利亚情形一样，接种工作的中止会对邻国造成极大的负面影响。
  - EN echo: as in nigeria, vaccination delays will be highly <unk> for neighboring countries.
  - ZH echo: 与尼日利亚情形一样，接种工作的中止会对邻国造成极大的负面影响。
- paired_cos=0.5007, top1=False
  - EN: In the first scene, Rachel uses poker to illustrate a concept to a large class sitting in rapt attention, and she schools a graduate teaching assistant.
  - ZH: 在第一幕中，瑞秋使用扑克向全班的注视下演示了一个概念，而她还只是一名研究生教学助理。
  - EN echo: in the first scene, <unk> uses <unk> to illustrate a concept to a large class sitting in <unk> attention, and she schools a graduate teaching <unk>.
  - ZH echo: 在第一幕中，瑞秋使用扑克向全班的注视下演示了一个概念，而她还只是一名研究生教学助理。
- paired_cos=0.7705, top1=True
  - EN: By 2050, Africa’s youth population is expected to reach 840 million.
  - ZH: 到2050年，非洲青年人口预计将达到8.4亿。
  - EN echo: by 2050, africa s youth population is expected to reach <unk> million.
  - ZH echo: 到2050年，非洲青年人口预计将达到8.4亿。
- paired_cos=0.6377, top1=True
  - EN: Germany is facing a deep political transition, as Chancellor Angela Merkel prepares to retire at the end of her current term.
  - ZH: 德国面临深刻的政治变局，总理默克尔准备在这个任期末退休。
  - EN echo: germany is facing a deep political transition, as chancellor angela merkel <unk> to <unk> at the end of her current term.
  - ZH echo: 德国面临深刻的政治变局，总理默克尔准备在这个任期末退休。
