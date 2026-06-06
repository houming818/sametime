import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Massive Hypers ===
D_MODEL = 128
TD = 5
VOCAB_SIZE = 32000
BATCH_SIZE = 512 # L1 can handle large batch size since no autoregressive decoder
EPOCHS = 3
LR = 5e-4
MAX_LEN = 40
DATA_PATH = '/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv'
BPE_MODEL = '/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model'
SAVE_DIR = '/mnt/nas/datasets/wmt_massive/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Dataset
class MassiveAlignDataset(Dataset):
    def __init__(self, sp, limit=None):
        self.sp = sp
        self.pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
        self.pairs = []
        
        print("Loading massive dataset into memory (may take a minute)...")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    zh, en = parts
                    self.pairs.append((en.lower(), zh))
                    
        print(f"Loaded {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, zh = self.pairs[idx]
        en_ids = self.sp.encode_as_ids(en)[:MAX_LEN]
        zh_ids = self.sp.encode_as_ids(zh)[:MAX_LEN]
        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(zh_ids, dtype=torch.long)

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

# 2. Model
class L1Aligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.L0 = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.t_nodes = nn.ModuleList([nn.Embedding(2**i, D_MODEL) for i in range(TD)])
        self.t_merge = nn.Linear(D_MODEL, D_MODEL)
        
    def forward(self, tok_ids):
        t = F.normalize(self.L0(tok_ids), dim=-1)
        
        w = torch.zeros(*tok_ids.shape, D_MODEL, device=tok_ids.device)
        for l in range(TD):
            nidx = torch.clamp(tok_ids // (VOCAB_SIZE // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
            w = w + self.t_nodes[l](nidx)
        w = F.normalize(w, dim=-1)
        
        tL, tR = t[..., :D_MODEL//2], t[..., D_MODEL//2:]
        wL, wR = w[..., :D_MODEL//2], w[..., D_MODEL//2:]
        return self.t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

# 3. Training Loop
def main():
    print("=== SPR Phase 1&2: Massive L1 Anchor Tree Training ===")
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    pad_id = sp.pad_id() if sp.pad_id() != -1 else 0
    
    # Due to RAM limits, we might not load all 14M at once in python list if RAM is tight.
    # But 14M tuples of strings is ~ 1-2 GB, should be fine.
    dataset = MassiveAlignDataset(sp)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    model = L1Aligner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    tau = 0.05
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, (en_pad, zh_pad) in enumerate(pbar):
            en_pad, zh_pad = en_pad.to(device), zh_pad.to(device)
            
            optimizer.zero_grad()
            
            en_emb = model(en_pad) # [B, L1, D]
            zh_emb = model(zh_pad) # [B, L2, D]
            
            # Sentence level InfoNCE for alignment
            en_mask = (en_pad != pad_id).float().unsqueeze(-1)
            zh_mask = (zh_pad != pad_id).float().unsqueeze(-1)
            
            en_sent = (en_emb * en_mask).sum(1) / (en_mask.sum(1) + 1e-9)
            zh_sent = (zh_emb * zh_mask).sum(1) / (zh_mask.sum(1) + 1e-9)
            
            en_sent = F.normalize(en_sent, dim=-1)
            zh_sent = F.normalize(zh_sent, dim=-1)
            
            # Similarity matrix [B, B]
            sim = torch.matmul(en_sent, zh_sent.t()) / tau
            labels = torch.arange(sim.size(0), device=device)
            
            loss_en = F.cross_entropy(sim, labels)
            loss_zh = F.cross_entropy(sim.t(), labels)
            loss = (loss_en + loss_zh) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_loss = total_loss / len(loader)
        print(f"\\nEpoch {epoch} Average Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'L0': model.L0.state_dict(),
            't_nodes': [m.state_dict() for m in model.t_nodes],
            't_merge': model.t_merge.state_dict()
        }, os.path.join(SAVE_DIR, f'anchor_tree_massive_ep{epoch}.pt'))

if __name__ == '__main__':
    main()
