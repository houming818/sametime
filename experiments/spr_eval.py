import torch
import torch.nn.functional as F
from spr_dataset import get_gold_eval_anchors, collate_anchors_to_padded

def evaluate_alignment(model, sp, train_anchors, device):
    """
    对模型进行完整的语义空间对齐评测。
    1. 计算训练/对齐锚点对 (Train Anchors) 的 top-1 匹配准确率 (NCE Acc) 和平均 Cosine 相似度
    2. 计算未见过的金标准手工锚点对 (Gold Eval Anchors) 的 top-1 匹配准确率 (Gold Acc) 和平均 Cosine 相似度
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # ── 1. 评测 Train Anchors ──
        if len(train_anchors) > 0:
            en_train_list = [e for e, _ in train_anchors]
            zh_train_list = [z for _, z in train_anchors]
            N_train = len(train_anchors)
            
            en_pad, en_mask = collate_anchors_to_padded(en_train_list, device)
            zh_pad, zh_mask = collate_anchors_to_padded(zh_train_list, device)
            
            en_vecs = F.normalize(model.get_word_representation(en_pad, en_mask), dim=-1)
            zh_vecs = F.normalize(model.get_word_representation(zh_pad, zh_mask), dim=-1)
            
            sim_matrix = en_vecs @ zh_vecs.T
            preds = sim_matrix.argmax(dim=-1)
            targets = torch.arange(N_train, device=device)
            
            train_acc = (preds == targets).float().mean().item() * 100.0
            mean_train_cos = sim_matrix.diag().mean().item()
            
            results['train_acc'] = train_acc
            results['train_cos'] = mean_train_cos
        else:
            results['train_acc'] = 0.0
            results['train_cos'] = 0.0

        # ── 2. 评测 Gold Eval Anchors (手动标注，未在训练中出现) ──
        gold_pairs = get_gold_eval_anchors(sp)
        if len(gold_pairs) > 0:
            en_gold_list = [e for e, _ in gold_pairs]
            zh_gold_list = [z for _, z in gold_pairs]
            N_gold = len(gold_pairs)
            
            en_gold_pad, en_gold_mask = collate_anchors_to_padded(en_gold_list, device)
            zh_gold_pad, zh_gold_mask = collate_anchors_to_padded(zh_gold_list, device)
            
            en_gold_vecs = F.normalize(model.get_word_representation(en_gold_pad, en_gold_mask), dim=-1)
            zh_gold_vecs = F.normalize(model.get_word_representation(zh_gold_pad, zh_gold_mask), dim=-1)
            
            gold_sim_matrix = en_gold_vecs @ zh_gold_vecs.T
            gold_preds = gold_sim_matrix.argmax(dim=-1)
            gold_targets = torch.arange(N_gold, device=device)
            
            gold_acc = (gold_preds == gold_targets).float().mean().item() * 100.0
            mean_gold_cos = gold_sim_matrix.diag().mean().item()
            
            results['gold_acc'] = gold_acc
            results['gold_cos'] = mean_gold_cos
        else:
            results['gold_acc'] = 0.0
            results['gold_cos'] = 0.0
            
    return results
