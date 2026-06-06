import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import argparse
import os

# --- Model Definitions ---
VOCAB_SIZE = 32000
D_MODEL_L1 = 128
TD = 5

class FrozenL1Encoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.L0 = nn.Embedding(VOCAB_SIZE, D_MODEL_L1)
        self.t_nodes = nn.ModuleList([nn.Embedding(2**i, D_MODEL_L1) for i in range(TD)])
        self.t_merge = nn.Linear(D_MODEL_L1, D_MODEL_L1)
        
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.L0.load_state_dict(ckpt['L0'])
        for i in range(TD):
            self.t_nodes[i].load_state_dict(ckpt['t_nodes'][i])
        self.t_merge.load_state_dict(ckpt['t_merge'])

    def forward(self, tok_ids):
        t = F.normalize(self.L0(tok_ids), dim=-1)
        w = torch.zeros(*tok_ids.shape, D_MODEL_L1, device=tok_ids.device)
        for l in range(TD):
            nidx = torch.clamp(tok_ids // (VOCAB_SIZE // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
            w = w + self.t_nodes[l](nidx)
        w = F.normalize(w, dim=-1)
        tL, tR = t[..., :D_MODEL_L1//2], t[..., D_MODEL_L1//2:]
        wL, wR = w[..., :D_MODEL_L1//2], w[..., D_MODEL_L1//2:]
        out = self.t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))
        return out

class DecoderL2(nn.Module):
    def __init__(self, d_model=1024, num_layers=6, nhead=16):
        super().__init__()
        self.proj_l1 = nn.Linear(D_MODEL_L1, d_model)
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 70, d_model) * 0.05)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4, activation="gelu")
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)
        self.fc_out.weight = self.emb.weight

    def forward(self, l1_mem, tgt_ids, tgt_mask):
        B, S_tgt = tgt_ids.size()
        S_src = l1_mem.size(1)
        
        mem = self.proj_l1(l1_mem)
        tgt = self.emb(tgt_ids) * (1024 ** 0.5)
        tgt = tgt + self.pos_enc[:, :S_tgt, :]
        
        out = self.transformer_decoder(tgt, mem, tgt_mask=tgt_mask)
        return self.fc_out(out)

# --- Inference Logic ---
def beam_search_decode(encoder, decoder, src_ids, sp, device, beam_size=4, max_len=64, rep_penalty=1.2):
    encoder.eval()
    decoder.eval()
    
    src_ids = src_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        l1_mem = encoder(src_ids)
    
    bos = sp.bos_id() if sp.bos_id() != -1 else 1
    eos = sp.eos_id() if sp.eos_id() != -1 else 2
    
    beams = [(0.0, [bos])]
    
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for score, tgt_seq in beams:
                if tgt_seq[-1] == eos:
                    new_beams.append((score, tgt_seq))
                    continue
                    
                tgt_tensor = torch.tensor([tgt_seq], device=device)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
                
                logits = decoder(l1_mem, tgt_tensor, tgt_mask)
                next_logits = logits[0, -1, :]
                
                # Penalty
                for tok in set(tgt_seq):
                    if next_logits[tok] > 0:
                        next_logits[tok] /= rep_penalty
                    else:
                        next_logits[tok] *= rep_penalty
                        
                probs = F.log_softmax(next_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, beam_size)
                
                for p, i in zip(topk_probs, topk_ids):
                    new_beams.append((score + p.item(), tgt_seq + [i.item()]))
                    
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            
            if all(b[1][-1] == eos for b in beams):
                break
                
    best_seq = beams[0][1]
    if eos in best_seq:
        best_seq = best_seq[:best_seq.index(eos)]
    if bos in best_seq:
        best_seq = best_seq[best_seq.index(bos)+1:]
        
    return sp.decode_ids(best_seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help="Input English text")
    parser.add_argument('--ckpt_l1', type=str, default='/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep3.pt')
    parser.add_argument('--ckpt_l2', type=str, default='/mnt/nas/datasets/wmt_massive/checkpoints/l2_massive_decoder_ep5.pt')
    parser.add_argument('--bpe_model', type=str, default='/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model')
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load BPE
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    # Load Models
    encoder = FrozenL1Encoder(args.ckpt_l1).to(device)
    decoder = DecoderL2().to(device)
    
    ckpt_l2 = torch.load(args.ckpt_l2, map_location='cpu', weights_only=True)
    decoder.load_state_dict(ckpt_l2['model_state_dict'])
    print(f"Loaded L2 weights from epoch {ckpt_l2.get('epoch', '?')}, val BLEU: {ckpt_l2.get('bleu', '?'):.2f}")
    
    src_ids = torch.tensor(sp.encode_as_ids(args.text.lower()))
    
    out_text = beam_search_decode(encoder, decoder, src_ids, sp, device, beam_size=args.beam_size)
    print(f"\n[EN]: {args.text}")
    print(f"[ZH]: {out_text}\n")

if __name__ == "__main__":
    main()
