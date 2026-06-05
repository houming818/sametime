import os
import sys
import time
import math
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# === Hyperparameters ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)

parser.add_argument('--rep_penalty', type=float, default=1.2)
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='/workspace/checkpoints')
parser.add_argument('--data_path', type=str, default='/mnt/nas/datasets/wmt17/train.zh-en')
parser.add_argument('--data_path', type=str, default='/mnt/nas/datasets/wmt17/train.zh-en')
parser.add_argument('--bpe_model', type=str, default='/mnt/nas/datasets/wmt17/sp_bpe.model')
parser.add_argument('--ckpt_l1', type=str, default='/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt')
args = parser.parse_args()

D_MODEL_L1 = 128
D_MODEL_L2 = args.d_model
TD = 5
VOCAB_SIZE = 16000
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
MAX_LEN = 80
CKPT_L1 = args.ckpt_l1
BPE_MODEL = args.bpe_model
DATA_PATH = args.data_path

REP_PENALTY = args.rep_penalty
BEAM_SIZE = args.beam_size
SAVE_DIR = args.save_dir

import os
os.makedirs(SAVE_DIR, exist_ok=True)

# === 1. Data Loader ===
class WMTDataset(Dataset):
    def __init__(self, pairs, sp):
        self.pairs = pairs
        self.sp = sp
        self.pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
        self.bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
        self.eos_id = sp.eos_id() if sp.eos_id() != -1 else 2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, zh = self.pairs[idx]
        
        en_ids = self.sp.encode_as_ids(en)[:MAX_LEN]
        # Target needs BOS and EOS
        zh_ids = [self.bos_id] + self.sp.encode_as_ids(zh)[:MAX_LEN-2] + [self.eos_id]
        
        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(zh_ids, dtype=torch.long)

def collate_fn(batch):
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
    
    en_batch, zh_batch = zip(*batch)
    
    # Pad EN
    en_lens = [len(x) for x in en_batch]
    en_pad = torch.full((len(en_batch), max(en_lens)), pad_id, dtype=torch.long)
    for i, x in enumerate(en_batch):
        en_pad[i, :len(x)] = x
        
    # Pad ZH
    zh_lens = [len(x) for x in zh_batch]
    zh_pad = torch.full((len(zh_batch), max(zh_lens)), pad_id, dtype=torch.long)
    for i, x in enumerate(zh_batch):
        zh_pad[i, :len(x)] = x
        
    return en_pad, zh_pad

# === 2. Model Architecture ===
class FrozenL1Encoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.L0 = nn.Embedding(VOCAB_SIZE, D_MODEL_L1)
        self.t_nodes = nn.ModuleList([nn.Embedding(2**i, D_MODEL_L1) for i in range(TD)])
        self.t_merge = nn.Linear(D_MODEL_L1, D_MODEL_L1)
        
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            self.L0.load_state_dict(ckpt['L0'])
            for i in range(TD):
                self.t_nodes[i].load_state_dict(ckpt['t_nodes'][i])
            self.t_merge.load_state_dict(ckpt['t_merge'])
            print(f"[FrozenL1Encoder] ✅ Successfully loaded L1 weights from {ckpt_path}")
        except Exception as e:
            print(f"[FrozenL1Encoder] ⚠️ Failed to load L1 checkpoint. Using random init. Error: {e}")

        # Freeze L0 and L1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, tok_ids):
        # L0 Lookup
        t = F.normalize(self.L0(tok_ids), dim=-1)
        
        # L1 Path sum
        w = torch.zeros(*tok_ids.shape, D_MODEL_L1, device=tok_ids.device)
        for l in range(TD):
            nidx = torch.clamp(tok_ids // (VOCAB_SIZE // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
            w = w + self.t_nodes[l](nidx)
        w = F.normalize(w, dim=-1)
        
        # Complex Multiply
        tL, tR = t[..., :D_MODEL_L1//2], t[..., D_MODEL_L1//2:]
        wL, wR = w[..., :D_MODEL_L1//2], w[..., D_MODEL_L1//2:]
        out = self.t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))
        
        return out # [Batch, SeqLen, D]

class DecoderL2(nn.Module):
    def __init__(self, d_model=D_MODEL_L2, l1_dim=D_MODEL_L1, nhead=args.nhead, num_layers=args.num_layers):
        super().__init__()
        self.proj_l1 = nn.Linear(l1_dim, d_model)
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, MAX_LEN + 10, d_model) * 0.05)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)
        self.fc_out.weight = self.emb.weight

    def forward(self, tgt_ids, l1_memory, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        seq_len = tgt_ids.size(1)
        tgt_emb = self.emb(tgt_ids) + self.pos_enc[:, :seq_len, :]
        l1_mem_proj = self.proj_l1(l1_memory)
        
        out = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=l1_mem_proj, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        return self.fc_out(out)

# === 3. BLEU Evaluation ===
def get_ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def compute_bleu(hypotheses, references):
    if len(hypotheses) == 0: return 0.0
    total_prec = [0.0] * 4
    total_hyp_len, total_ref_len = 0, 0
    
    for hyp, ref in zip(hypotheses, references):
        total_hyp_len += len(hyp)
        total_ref_len += len(ref)
        for n in range(1, 5):
            hyp_ngrams = Counter(get_ngrams(hyp, n))
            ref_ngrams = Counter(get_ngrams(ref, n))
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hyp) - n + 1, 1)
            if total > 0:
                total_prec[n-1] += min(overlap / total, 1.0)
                
    if total_hyp_len == 0: return 0.0
    bp = min(1.0, total_hyp_len / max(total_ref_len, 1))
    precs = [p / len(hypotheses) for p in total_prec]
    if any(p <= 0 for p in precs[:2]): return 0.0
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in precs) / 4) * 100


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

@torch.no_grad()
def greedy_decode(encoder, decoder, src_ids, sp, max_len=MAX_LEN):
    encoder.eval()
    decoder.eval()
    
    src_ids = src_ids.unsqueeze(0).to(device) # [1, SeqLen]
    l1_mem = encoder(src_ids)
    
    tgt_ids = torch.tensor([[sp.bos_id()]], device=device) # [1, 1]
    
    for _ in range(max_len):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
        logits = decoder(tgt_ids, l1_mem, tgt_mask=tgt_mask)
        next_logits = logits[0, -1, :]
        
        # Apply repetition penalty
        if REP_PENALTY > 1.0:
            for token_id in set(tgt_ids[0].tolist()):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= REP_PENALTY
                else:
                    next_logits[token_id] *= REP_PENALTY
                    
        next_token = next_logits.argmax().item()
        
        tgt_ids = torch.cat([tgt_ids, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == sp.eos_id():
            break
            
    # return ids excluding BOS and EOS
    res = tgt_ids[0].tolist()
    if res[0] == sp.bos_id(): res = res[1:]
    if len(res) > 0 and res[-1] == sp.eos_id(): res = res[:-1]
    return res

@torch.no_grad()
def evaluate_model(encoder, decoder, test_pairs, sp, num_samples=500):
    subset = random.sample(test_pairs, min(num_samples, len(test_pairs)))
    hyps, refs = [], []
    
    for en, zh in subset:
        en_ids = torch.tensor(sp.encode_as_ids(en)[:MAX_LEN])
        zh_ids = sp.encode_as_ids(zh)
        
        pred_ids = beam_search_decode(encoder, decoder, en_ids, sp) if BEAM_SIZE > 1 else greedy_decode(encoder, decoder, en_ids, sp)
        
        hyps.append(sp.decode_ids(pred_ids).split())
        refs.append(sp.decode_ids(zh_ids).split())
        
    bleu_score = compute_bleu(hyps, refs)
    return bleu_score

# === 4. Main Training Loop ===
def main():
    print("=== SPR Phase 3: L2 Decoder Translation Training ===")
    print(f"Args: {args}")
    
    # 1. Load Tokenizer & Data
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    pad_id = sp.pad_id()
    print(f"Loaded Tokenizer. Vocab={sp.get_piece_size()}")
    
    all_pairs = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                zh, en = line.strip().split('\t', 1)
                all_pairs.append((en.strip().lower(), zh.strip()))
                
    print(f"Total Sentences: {len(all_pairs)}")
    
    # Split 98k train, 2k test
    train_pairs = all_pairs[:-2000]
    test_pairs = all_pairs[-2000:]
    
    train_dataset = WMTDataset(train_pairs, sp)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
                              
    print(f"Train Dataset: {len(train_pairs)}, Test Dataset: {len(test_pairs)}")
    
    # 2. Setup Model
    encoder = FrozenL1Encoder(CKPT_L1).to(device)
    decoder = DecoderL2(d_model=D_MODEL_L1).to(device)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # 3. Train Loop
    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        total_loss, total_tokens = 0, 0
        
        for i, (en_pad, zh_pad) in enumerate(train_loader):
            en_pad = en_pad.to(device)
            zh_pad = zh_pad.to(device)
            
            tgt_input = zh_pad[:, :-1]
            tgt_label = zh_pad[:, 1:]
            
            src_pad_mask = (en_pad == pad_id)
            tgt_pad_mask = (tgt_input == pad_id)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward
            l1_mem = encoder(en_pad)
            logits = decoder(tgt_input, l1_mem, src_pad_mask=src_pad_mask, 
                             tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
            
            # Loss
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * tgt_label.numel()
            total_tokens += tgt_label.numel()
            
            if (i+1) % 100 == 0:
                avg_loss = total_loss / total_tokens
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
                total_loss, total_tokens = 0, 0
                
        # Eval at end of epoch
        print(f"\n--- Epoch {epoch} Evaluation ---")
        bleu = evaluate_model(encoder, decoder, test_pairs, sp, num_samples=100)
        elapsed = time.time() - start_time
        print(f"Validation BLEU: {bleu:.2f} | Elapsed: {elapsed:.0f}s")
        
        # Display samples
        decoder.eval()
        sample_pair = test_pairs[0]
        en_str, zh_ref = sample_pair
        en_ids = torch.tensor(sp.encode_as_ids(en_str)[:MAX_LEN])
        pred_ids = beam_search_decode(encoder, decoder, en_ids, sp) if BEAM_SIZE > 1 else greedy_decode(encoder, decoder, en_ids, sp)
        print(f"  EN: {en_str}")
        print(f"  REF: {zh_ref}")
        print(f"  HYP: {sp.decode_ids(pred_ids)}")
        
        print("-" * 30 + "
")
        
        # Save Checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f'l2_decoder_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu': bleu_score
        }, ckpt_path)

if __name__ == "__main__":
    main()