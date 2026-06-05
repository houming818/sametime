import re

with open('/home/nio/log/holds/SameTime/experiments/spr_l2_train.py', 'r') as f:
    content = f.read()

# 1. Add Beam Search & Checkpointing args
args_replace = """
parser.add_argument('--rep_penalty', type=float, default=1.2)
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='/workspace/checkpoints')
parser.add_argument('--data_path', type=str, default='/mnt/nas/datasets/wmt17/train.zh-en')
"""
content = re.sub(r'parser\.add_argument\(\'--rep_penalty.*?\n', args_replace, content)

args_init_replace = """
REP_PENALTY = args.rep_penalty
BEAM_SIZE = args.beam_size
SAVE_DIR = args.save_dir

import os
os.makedirs(SAVE_DIR, exist_ok=True)
"""
content = re.sub(r'REP_PENALTY = args\.rep_penalty\n', args_init_replace, content)

# 2. Add Beam Search implementation
beam_search_code = """
@torch.no_grad()
def beam_search_decode(encoder, decoder, src_ids, sp, beam_size=BEAM_SIZE, max_len=MAX_LEN):
    encoder.eval()
    decoder.eval()
    
    src_ids = src_ids.unsqueeze(0).to(device)
    l1_mem = encoder(src_ids) # [1, SeqLen, D]
    
    # [score, tgt_ids]
    beams = [(0.0, [sp.bos_id()])]
    
    for _ in range(max_len):
        new_beams = []
        for score, tgt_seq in beams:
            if tgt_seq[-1] == sp.eos_id():
                new_beams.append((score, tgt_seq))
                continue
                
            tgt_tensor = torch.tensor([tgt_seq], device=device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            logits = decoder(tgt_tensor, l1_mem, tgt_mask=tgt_mask)
            next_logits = F.log_softmax(logits[0, -1, :], dim=-1)
            
            # Apply rep penalty
            if REP_PENALTY > 1.0:
                for token_id in set(tgt_seq):
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= REP_PENALTY
                    else:
                        next_logits[token_id] /= REP_PENALTY
            
            topk_log_probs, topk_ids = torch.topk(next_logits, beam_size)
            
            for prob, next_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_beams.append((score + prob, tgt_seq + [next_id]))
                
        # Sort and prune
        beams = sorted(new_beams, key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)[:beam_size]
        
        # Check if all completed
        if all(b[1][-1] == sp.eos_id() for b in beams):
            break
            
    best_seq = beams[0][1]
    if best_seq[0] == sp.bos_id(): best_seq = best_seq[1:]
    if len(best_seq) > 0 and best_seq[-1] == sp.eos_id(): best_seq = best_seq[:-1]
    return best_seq

# Replace greedy with beam in evaluate
"""
content = content.replace("@torch.no_grad()\ndef greedy_decode", beam_search_code + "\n@torch.no_grad()\ndef greedy_decode")
content = content.replace("pred_ids = greedy_decode(encoder, decoder, en_ids, sp)", "pred_ids = beam_search_decode(encoder, decoder, en_ids, sp) if BEAM_SIZE > 1 else greedy_decode(encoder, decoder, en_ids, sp)")

# 3. Add Save checkpoint logic in training loop
save_code = """
        print("-" * 30 + "\\n")
        
        # Save Checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f'l2_decoder_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu': bleu_score
        }, ckpt_path)
"""
content = re.sub(r'print\("-" \* 30 \+ "\\n"\)\n', save_code, content)

with open('/home/nio/log/holds/SameTime/experiments/spr_l2_train.py', 'w') as f:
    f.write(content)
