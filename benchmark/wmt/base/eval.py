"""
base/eval.py — sacreBLEU 评测

每个 phase 有自己的 greedy_decode / beam_search（模型 API 不同）。
"""

import sacrebleu


def compute_bleu(refs, hyps):
    """计算 BLEU-4，refs 和 hyps 均为 list[str]"""
    return sacrebleu.corpus_bleu(hyps, [refs]).score
