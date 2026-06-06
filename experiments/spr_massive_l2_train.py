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
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--d_model', type=int, default=1024, help="L2 Decoder Hidden Size")
parser.add_argument('--nhead', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)

parser.add_argument('--rep_penalty', type=float, default=1.2)
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='/mnt/nas/datasets/wmt_massive/checkpoints')
parser.add_argument('--data_path', type=str, default='/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv')
parser.add_argument('--bpe_model', type=str, default='/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model')
parser.add_argument('--ckpt_l1', type=str, default='/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep3.pt')
parser.add_argument('--poc', action='store_true', help="Run small POC with 5000 rows to fast fail")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL_L1 = 128
D_MODEL_L2 = args.d_model
TD = 5
VOCAB_SIZE = 32000
BATCH_SIZE = args.batch_size
EPOCHS = 1 if args.poc else args.epochs
LR = args.lr
MAX_LEN = 60

CKPT_L1 = args.ckpt_l1
BPE_MODEL = args.bpe_model
DATA_PATH = args.data_path
SAVE_DIR = args.save_dir
REP_PENALTY = args.rep_penalty
BEAM_SIZE = args.beam_size

os.makedirs(SAVE_DIR, exist_ok=True)

# === 1. Data Loader ===
import array

class MassiveWMTDataset(Dataset):
    def __init__(self, sp, data_path, is_test=False):
        self.sp = sp
        self.pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
        self.bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
        self.eos_id = sp.eos_id() if sp.eos_id() != -1 else 2
        
        self.offsets = array.array('Q')
        print(f"Building index for {data_path} ...")
        with open(data_path, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
        
        # Split train/test
        if is_test:
            self.offsets = self.offsets[-2000:]
            if args.poc:
                self.offsets = self.offsets[:50] # Just 50 for testing POC
        else:
            self.offsets = self.offsets[:-2000]
            if args.poc:
                self.offsets = self.offsets[:5000] # 5000 for train POC
            
        print(f"Indexed {len(self.offsets)} pairs.")
        
        self.f = open(data_path, 'r', encoding='utf-8')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.f.seek(self.offsets[idx])
        line = self.f.readline()
        parts = line.strip().split('\t')
        if len(parts) == 2:
            zh = parts[0]
            en = parts[1].lower()
        else:
            zh, en = "", ""
            
        en_ids = self.sp.encode_as_ids(en)[:MAX_LEN]
        zh_ids = [self.bos_id] + self.sp.encode_as_ids(zh)[:MAX_LEN-2] + [self.eos_id]
        
        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(zh_ids, dtype=torch.long)

    def get_text(self, idx):
        self.f.seek(self.offsets[idx])
        line = self.f.readline()
        parts = line.strip().split('\t')
        if len(parts) == 2:
            return parts[1].lower(), parts[0]
        return "", ""

def collate_fn(batch):
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
    
    en_batch, zh_batch = zip(*batch)
    
    en_lens = [len(x) for x in en_batch]
    en_pad = torch.full((len(en_batch), max(max(en_lens), 1)), pad_id, dtype=torch.long)
    for i, x in enumerate(en_batch):
        if len(x) > 0: en_pad[i, :len(x)] = x
        
    zh_lens = [len(x) for x in zh_batch]
    zh_pad = torch.full((len(zh_batch), max(max(zh_lens), 1)), pad_id, dtype=torch.long)
    for i, x in enumerate(zh_batch):
        if len(x) > 0: zh_pad[i, :len(x)] = x
        
    return en_pad, zh_pad

# === 2. Model Architecture ===
class FrozenL1Encoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        # Load weights temporarily to collapse them
        L0 = nn.Embedding(VOCAB_SIZE, D_MODEL_L1)
        t_nodes = nn.ModuleList([nn.Embedding(2**i, D_MODEL_L1) for i in range(TD)])
        t_merge = nn.Linear(D_MODEL_L1, D_MODEL_L1)
        
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            L0.load_state_dict(ckpt['L0'])
            for i in range(TD):
                t_nodes[i].load_state_dict(ckpt['t_nodes'][i])
            t_merge.load_state_dict(ckpt['t_merge'])
            print(f"[FrozenL1Encoder] ✅ Successfully loaded L1 weights from {ckpt_path} for collapse.")
        except Exception as e:
            print(f"[FrozenL1Encoder] ⚠️ Failed to load L1 checkpoint. Using random init. Error: {e}")

        # Collapse the tree mathematically!
        print("[FrozenL1Encoder] Performing Mathematical Collapse of L1 manifold...")
        all_ids = torch.arange(VOCAB_SIZE)
        
        # We don't need gradients for collapse
        with torch.no_grad():
            t = F.normalize(L0(all_ids), dim=-1)
            w = torch.zeros(VOCAB_SIZE, D_MODEL_L1)
            for l in range(TD):
                nidx = torch.clamp(all_ids // (VOCAB_SIZE // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(all_ids)
                w = w + t_nodes[l](nidx)
            w = F.normalize(w, dim=-1)
            
            tL, tR = t[:, :D_MODEL_L1//2], t[:, D_MODEL_L1//2:]
            wL, wR = w[:, :D_MODEL_L1//2], w[:, D_MODEL_L1//2:]
            collapsed_embeds = t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

        self.collapsed_emb = nn.Embedding(VOCAB_SIZE, D_MODEL_L1)
        self.collapsed_emb.weight.data.copy_(collapsed_embeds)
        self.collapsed_emb.weight.requires_grad = False
        print("[FrozenL1Encoder] ✅ Mathematical Collapse complete. O(N) -> O(1) complexity.")

    def forward(self, tok_ids):
        return self.collapsed_emb(tok_ids)

class DecoderL2(nn.Module):
    def __init__(self, d_model=D_MODEL_L2, l1_dim=D_MODEL_L1, nhead=args.nhead, num_layers=args.num_layers):
        super().__init__()
        self.proj_l1 = nn.Linear(l1_dim, d_model)
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, MAX_LEN + 10, d_model) * 0.05)
        
        # norm_first=True (Pre-LN) is crucial for stable training of deep transformers, preventing loss=nan
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=d_model*4, activation="gelu", norm_first=True
        )
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

# === 3. Evaluation & Decoding ===
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
    l1_mem = encoder(src_ids)
    
    bos = sp.bos_id() if sp.bos_id() != -1 else 1
    eos = sp.eos_id() if sp.eos_id() != -1 else 2
    
    beams = [(0.0, [bos])]
    
    for _ in range(max_len):
        new_beams = []
        for score, tgt_seq in beams:
            if tgt_seq[-1] == eos:
                new_beams.append((score, tgt_seq))
                continue
                
            tgt_tensor = torch.tensor([tgt_seq], device=device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            logits = decoder(tgt_tensor, l1_mem, tgt_mask=tgt_mask)
            next_logits = F.log_softmax(logits[0, -1, :], dim=-1)
            
            if REP_PENALTY > 1.0:
                for token_id in set(tgt_seq):
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= REP_PENALTY
                    else:
                        next_logits[token_id] /= REP_PENALTY
            
            topk_log_probs, topk_ids = torch.topk(next_logits, beam_size)
            
            for prob, next_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_beams.append((score + prob, tgt_seq + [next_id]))
                
        beams = sorted(new_beams, key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)[:beam_size]
        
        if all(b[1][-1] == eos for b in beams):
            break
            
    best_seq = beams[0][1]
    if best_seq[0] == bos: best_seq = best_seq[1:]
    if len(best_seq) > 0 and best_seq[-1] == eos: best_seq = best_seq[:-1]
    return best_seq

@torch.no_grad()
def evaluate_model(encoder, decoder, test_dataset, sp, num_samples=100):
    subset_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    hyps, refs = [], []
    
    for idx in subset_indices:
        en, zh = test_dataset.get_text(idx)
        en_ids = torch.tensor(sp.encode_as_ids(en)[:MAX_LEN])
        zh_ids = sp.encode_as_ids(zh)
        
        pred_ids = beam_search_decode(encoder, decoder, en_ids, sp)
        
        hyps.append(sp.decode_ids(pred_ids).split())
        refs.append(sp.decode_ids(zh_ids).split())
        
    bleu_score = compute_bleu(hyps, refs)
    return bleu_score, hyps, refs, subset_indices

# === 4. Main Training Loop ===
def main():
    print(f"=== SPR Phase 3: MASSIVE L2 Decoder Translation Training ===")
    print(f"Args: {args}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
    print(f"Loaded Tokenizer. Vocab={sp.get_piece_size()}")
    
    print(f"Loading massive dataset index...")
    
    train_dataset = MassiveWMTDataset(sp, DATA_PATH, is_test=False)
    test_dataset = MassiveWMTDataset(sp, DATA_PATH, is_test=True)
    
    # num_workers=0 because file handle self.f can't be easily shared if we fork
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
                              
    encoder = FrozenL1Encoder(CKPT_L1).to(device)
    decoder = DecoderL2(d_model=D_MODEL_L2, num_layers=args.num_layers, nhead=args.nhead).to(device)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    scaler = torch.cuda.amp.GradScaler() # Enable AMP for faster training on large model
    
    print("\n--- Starting Massive L2 Training ---")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        total_loss, total_tokens = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (en_pad, zh_pad) in enumerate(pbar):
            en_pad = en_pad.to(device)
            zh_pad = zh_pad.to(device)
            
            tgt_input = zh_pad[:, :-1]
            tgt_label = zh_pad[:, 1:]
            
            src_pad_mask = (en_pad == pad_id)
            tgt_pad_mask = (tgt_input == pad_id)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                l1_mem = encoder(en_pad)
                logits = decoder(tgt_input, l1_mem, src_pad_mask=src_pad_mask, 
                                 tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            tokens_in_batch = (tgt_label != pad_id).sum().item()
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
            
            if (i+1) % 100 == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
                total_loss, total_tokens = 0, 0
                
        print(f"\n--- Epoch {epoch} Evaluation ---")
        bleu, hyps, refs, indices = evaluate_model(encoder, decoder, test_dataset, sp, num_samples=50)
        elapsed = time.time() - start_time
        print(f"Validation BLEU: {bleu:.2f} | Elapsed: {elapsed:.0f}s")
        
        # Display a sample
        sample_idx = indices[0]
        en_str, zh_ref = test_dataset.get_text(sample_idx)
        print(f"  EN:  {en_str}")
        print(f"  REF: {zh_ref}")
        print(f"  HYP: {' '.join(hyps[0])}")
        print("-" * 30 + "\n")
        
        ckpt_path = os.path.join(SAVE_DIR, f'l2_massive_decoder_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu': bleu
        }, ckpt_path)

if __name__ == "__main__":
    main()
