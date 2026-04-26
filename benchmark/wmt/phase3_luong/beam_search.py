"""
Beam Search 解码

维护 beam_size 条候选路径，每一步扩展所有路径，保留 top-k。
"""

import torch
import torch.nn.functional as F


def beam_search(model, src, src_len, vocab_tgt, beam_size=3, max_len=50, length_penalty=1.0):
    """
    src: (B, S) — 实际只取 batch 中第一条
    返回: (1, T) — 最佳路径的 token ids
    """
    model.eval()
    B = src.size(0)
    assert B == 1, "beam_search 当前只支持 batch=1"

    with torch.no_grad():
        enc_out, hidden = model.encoder(src, src_len)

    # 初始序列: [SOS]
    sequences = [[vocab_tgt.SOS]]
    scores = [0.0]
    finished = []

    for step in range(max_len):
        if not sequences:
            break

        all_candidates = []
        for seq_id, seq in enumerate(sequences):
            tgt = torch.tensor([seq], device=src.device)
            logits, _ = model.decoder(tgt, (enc_out, hidden))
            log_prob = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, V)

            topk_scores, topk_ids = log_prob.topk(beam_size)
            for k in range(beam_size):
                candidate_seq = seq + [topk_ids[0, k].item()]
                candidate_score = scores[seq_id] + topk_scores[0, k].item()
                all_candidates.append((candidate_seq, candidate_score))

        # 排序并保留 top beam_size
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences, scores = [], []
        for seq, score in all_candidates[:beam_size]:
            if seq[-1] == vocab_tgt.EOS:
                # 长度惩罚
                lp = ((len(seq) - 1) ** length_penalty) / (len(seq) ** length_penalty)
                finished.append((seq, score / lp))
            else:
                sequences.append(seq)
                scores.append(score)

        if not sequences:
            break

    # 如果所有路径都结束了，选 finished 中最好的
    if finished:
        finished.sort(key=lambda x: x[1], reverse=True)
        return torch.tensor([finished[0][0]], device=src.device)
    # 否则选当前 sequences 最佳
    if sequences:
        return torch.tensor([sequences[0]], device=src.device)
    return torch.tensor([[vocab_tgt.SOS, vocab_tgt.EOS]], device=src.device)
