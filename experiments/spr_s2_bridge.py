"""
SPR S2 Translation Bridge — Root-to-Root Rigid Alignment (v2)
Architecture:
  DE: E_de + 4 fixed templates → argmax root_hash_de
  EN: E_en + 4 fixed templates → root_hash_en (gold, for training)
  Bridge: Residual MLP maps root_de → root_en (MSE alignment)
  Decoder: GRU autoregressive, gold leaves at train, self-built leaves at inference
Training: 3-phase — Phase0 echo pretrain, Phase1 root MSE + CE joint, Phase2 finetune
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR S2 BRIDGE v2 — Root-to-Root")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data ────
print("loading...")
pairs = []
with open(train_file) as f:
    for i,l in enumerate(f):
        if i>=50000: break
        if "\t" in l: pairs.append(tuple(c.strip().lower().split() for c in l.split("\t")[:2]))
train_pairs = pairs[:40000]; val_pairs = pairs[40000:40500]
print(f"train={len(train_pairs)} val={len(val_pairs)}")

word2id_de = {"<pad>":0,"<unk>":1}; word2id_en = {"<pad>":0,"<unk>":1}
freq_de,freq_en = Counter(),Counter()
for de,en in train_pairs:
    for w in de: freq_de[w]+=1
    for w in en: freq_en[w]+=1
for w,c in freq_de.most_common():
    if c>=2: word2id_de[w]=len(word2id_de)
for w,c in freq_en.most_common():
    if c>=2: word2id_en[w]=len(word2id_en)
for de,en in val_pairs:
    for w in de:
        if w not in word2id_de: word2id_de[w]=len(word2id_de)
    for w in en:
        if w not in word2id_en: word2id_en[w]=len(word2id_en)

V_de,V_en,d = len(word2id_de),len(word2id_en),128
id2word_en = {v:k for k,v in word2id_en.items()}
print(f"DE={V_de} EN={V_en} d={d}")

# ──── Shared: Fixed Templates ────
torch.manual_seed(42)
SIGN_MASK = torch.tensor([1.,-1.]*(d//2+1),device=device)[:d]
TEMPLATES = {'Left_Heavy':lambda n,d:1,'Right_Heavy':lambda n,d:max(1,n-1),'Balanced':lambda n,d:max(1,n//2),'Spec_Head':lambda n,d:3 if n>=6 else max(1,n//2)}
TEMPLATE_NAMES = list(TEMPLATES.keys())
MAX_LEN=50

def gen_paths(T,fn):
    def _gen(embs,_,depth=1):
        n=len(embs)
        if n<=1: return [('',0)] if n==1 else []
        s=fn(n,depth);L=max(1,min(s,n-1))
        l=_gen(embs[:L],None,depth+1);r=_gen(embs[L:],None,depth+1)
        return [('L'+p,i) for p,i in l]+[('R'+p,i+L) for p,i in r]
    results=_gen(list(range(T)),None);paths=['']*T
    for p,i in results: paths[i]=p
    return paths
template_paths={t:{T:gen_paths(T,fn) for T in range(2,MAX_LEN+1)} for t,fn in TEMPLATES.items()}

def _build_tree(embs,paths):
    T=len(embs)
    cur={('leaf',paths[t]):embs[t] for t in range(T) if t<len(paths)}
    md=max(len(p) for p in paths) if paths else 0
    for depth in range(md,0,-1):
        for pfx in set(p[:depth-1] if depth>1 else '' for p in paths if len(p)>=max(depth-1,0)):
            lk=('leaf',pfx+'L') if depth>1 else ('leaf','L')
            rk=('leaf',pfx+'R') if depth>1 else ('leaf','R')
            if lk in cur and rk in cur:
                lft=cur.pop(lk);rgt=cur.pop(rk)
                merged=lft+SIGN_MASK*torch.roll(rgt,shifts=depth)
                cur[('node',pfx)]=merged/(merged.norm()+1e-8)
    return next(iter(cur.values())).squeeze() if cur else torch.zeros(d,device=embs.device)

def compute_root(E,ids,tname):
    T=len(ids)
    if T<2: return E(torch.tensor([ids[0] if T>=1 else 0],device=device))
    k=min(T,MAX_LEN);paths=template_paths[tname].get(k,gen_paths(k,TEMPLATES[tname]))
    if len(paths)!=T: paths=gen_paths(T,TEMPLATES[tname])
    return _build_tree(E(torch.tensor(ids,device=device)),paths)

def get_best_root(E,ids):
    ids=ids[:MAX_LEN]
    if len(ids)<2: return E(torch.tensor([ids[0] if ids else 0],device=device))
    bv,br=-1e9,None
    for tn in TEMPLATE_NAMES:
        r=compute_root(E,ids,tn);sv=r.norm().item()
        if sv>bv: bv,br=sv,r
    return br

def get_pos_emb(T,dim=128):
    pos=torch.arange(T,device=device).float().unsqueeze(1)
    div=10000**(torch.arange(0,dim//2,device=device).float()*2/dim)
    phase=pos/div
    return torch.cat([torch.sin(phase),torch.cos(phase)],dim=-1)

# ──── Modules ────
class Bridge(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d),nn.LayerNorm(d))
    def forward(self,x): return x+self.net(x)

class GRUDecoder(nn.Module):
    def __init__(self,V): super().__init__(); self.d=d;self.gru=nn.GRUCell(d,d);self.W_out=nn.Linear(d,V)
    def forward(self,leaf_hashes,gold_ids=None,p_teacher=1.0):
        T=leaf_hashes.shape[0];h=torch.zeros(d,device=leaf_hashes.device);logits=[];prev=torch.zeros(d,device=leaf_hashes.device)
        for t in range(T):
            inp=leaf_hashes[t]
            if t>0:
                if gold_ids is not None and random.random()<p_teacher: inp=inp+0.3*leaf_hashes[min(t,T-1)]
                else: inp=inp+0.3*prev
            h=self.gru(inp,h);out=self.W_out(h);logits.append(out)
            with torch.no_grad(): pid=out.argmax(dim=-1).item()
            prev=leaf_hashes[min(pid,T-1)]
        return torch.stack(logits,dim=0)

# ──── Init ────
E_de=nn.Embedding(V_de,d).to(device);E_en=nn.Embedding(V_en,d).to(device)
bridge=Bridge(d).to(device);decoder=GRUDecoder(V_en).to(device)
nn.init.normal_(E_de.weight,0,0.02);nn.init.normal_(E_en.weight,0,0.02)
nP=sum(p.numel() for m in [E_de,E_en,bridge,decoder] for p in m.parameters())
print(f"params={nP/1e6:.1f}M")

# ──── BLEU ────
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def compute_bleu(refs,hyps):
    C=Counter;ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(refs,hyps):
            rc=C(ng(r,n));hc=C(ng(h,n));ttl+=sum(hc.values());mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1) if ttl>0 else 1.0)
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(refs,hyps) if len(h)>0]
    bp=min(2.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

val_de = [[word2id_de.get(w,1) for w in de] for de,en in val_pairs[:300] if len(de)>=2 and len(en)>=2]
val_en = [[word2id_en.get(w,1) for w in en] for de,en in val_pairs[:300] if len(de)>=2 and len(en)>=2]
print(f"val={len(val_de)} pairs")

# ══════════════════════════════════════════
# PHASE 0: EN echo pretrain
# ══════════════════════════════════════════
print(f"\n{'='*60}\nPHASE 0: EN echo pretrain (2 epochs × 10K sents)\n{'='*60}")
opt0=torch.optim.Adam(list(E_en.parameters())+list(decoder.parameters()),lr=0.003)
t0=time.time()
for ep in range(2):
    random.shuffle(train_pairs);tl,tt=0,0
    p_t=max(0.2,1.0-ep/10.0)
    for bi in range(0, 10000, 16):
        batch_s = train_pairs[bi:bi+16]
        opt0.zero_grad();loss_sum=torch.tensor(0.0,device=device);n_sents=0
        for de_s, en_s in batch_s:
            ids_en=[word2id_en.get(w,1) for w in en_s[:MAX_LEN]]
            if len(ids_en)<3: continue
            ids_t=torch.tensor(ids_en,device=device);T=len(ids_en)
            leaf=E_en(ids_t)+0.5*get_pos_emb(T);leaf=leaf/(leaf.norm(dim=-1,keepdim=True)+1e-8)
            logits=decoder(leaf,ids_t,p_teacher=p_t);loss_sum+=F.cross_entropy(logits,ids_t);n_sents+=1
        if n_sents==0: continue
        (loss_sum/n_sents).backward();opt0.step();tl+=loss_sum.item()/n_sents;tt+=1
    if ep%1==0 or ep==1: print(f"  echo EN ep {ep} loss={tl/max(tt,1):.4f}")

# ══════════════════════════════════════════
# PHASE 1: Bridge + Decoder Joint Training
# ══════════════════════════════════════════
# PHASE 1: Root MSE + Gold-Leaf Decoder
# ══════════════════════════════════════════
print(f"\n{'='*60}\nPHASE 1: Bridge MSE + Decoder CE (5 epochs × 3K sents)\n{'='*60}")
opt1=torch.optim.Adam(list(E_de.parameters())+list(bridge.parameters())+list(E_en.parameters())+list(decoder.parameters()),lr=0.003)
sched1=torch.optim.lr_scheduler.CosineAnnealingLR(opt1,T_max=5)
for ep in range(5):
    random.shuffle(train_pairs);tl,tt=0,0
    p_t=max(0.2,1.0-ep/5.0)
    
    for bi in range(0,3000,8):
        batch = train_pairs[bi:bi+8]
        opt1.zero_grad();bl=torch.tensor(0.0,device=device);n=0
        
        for de_s, en_s in batch:
            ids_de=[word2id_de.get(w,1) for w in de_s[:MAX_LEN]]
            ids_en=[word2id_en.get(w,1) for w in en_s[:MAX_LEN]]
            if len(ids_de)<3 or len(ids_en)<3: continue
            
            root_de=get_best_root(E_de,ids_de)
            root_en_pred=bridge(root_de)
            
            with torch.no_grad(): root_en_gold=get_best_root(E_en,ids_en)
            
            ids_en_t=torch.tensor(ids_en,device=device);T=len(ids_en)
            leaf_gold=E_en(ids_en_t)+0.5*get_pos_emb(T)
            leaf_gold=leaf_gold/(leaf_gold.norm(dim=-1,keepdim=True)+1e-8)
            
            # Decoder uses predicted root — learns to tolerate bridge output
            combined=leaf_gold+0.1*root_en_pred.unsqueeze(0).expand_as(leaf_gold)
            logits=decoder(combined,ids_en_t,p_teacher=p_t)
            loss=F.mse_loss(root_en_pred,root_en_gold.detach())+F.cross_entropy(logits,ids_en_t)
            bl+=loss;n+=1
        
        if n==0: continue;(bl/n).backward()
        torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]['params'],2.0);opt1.step()
        tl+=(bl/n).item();tt+=1
    
    if tt==0: continue;sched1.step()
    if ep%2==0 or ep==4:
        E_de.eval();E_en.eval();bridge.eval();decoder.eval()
        rf,hp=[],[]
        with torch.no_grad():
            for ids_de,ids_en in zip(val_de[:50],val_en[:50]):
                root_de=get_best_root(E_de,ids_de);root_en_pred=bridge(root_de)
                ids_en_t=torch.tensor(ids_en,device=device);T=len(ids_en)
                leaf=E_en(ids_en_t)+0.5*get_pos_emb(T);leaf=leaf/(leaf.norm(dim=-1,keepdim=True)+1e-8)
                combined=leaf+0.1*root_en_pred.unsqueeze(0).expand_as(leaf)
                logits=decoder(combined);pred=logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids_en);hp.append(pred[:T])
        b=compute_bleu(rf,hp);acc=100*sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in rf))
        print(f"  ep {ep:3d} loss={tl/tt:.4f} BLEU={b:.1f} tok_acc={acc:.1f}% time={time.time()-t0:.0f}s")
        E_de.train();E_en.train();bridge.train();decoder.train()

# ══════════════════════════════════════════
# PHASE 2: Joint fine-tune
# ══════════════════════════════════════════
print(f"\n{'='*60}\nPHASE 2: Joint fine-tune (3 epochs, lr=1e-4)\n{'='*60}")
opt2=torch.optim.Adam(list(E_de.parameters())+list(E_en.parameters())+list(bridge.parameters())+list(decoder.parameters()),lr=1e-4)
sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=3)
for ep in range(3):
    random.shuffle(train_pairs);tl,tt=0,0
    p_t=max(0.2,1.0-(ep+5)/10.0)
    
    for bi in range(0,2000,8):
        batch = train_pairs[bi:bi+8]
        opt2.zero_grad();bl=torch.tensor(0.0,device=device);n=0
        
        for de_s, en_s in batch:
            ids_de=[word2id_de.get(w,1) for w in de_s[:MAX_LEN]]
            ids_en=[word2id_en.get(w,1) for w in en_s[:MAX_LEN]]
            if len(ids_de)<3 or len(ids_en)<3: continue
            
            root_de=get_best_root(E_de,ids_de);root_en_pred=bridge(root_de)
            with torch.no_grad(): root_en_gold=get_best_root(E_en,ids_en)
            
            ids_en_t=torch.tensor(ids_en,device=device);T=len(ids_en)
            leaf_gold=E_en(ids_en_t)+0.5*get_pos_emb(T);leaf_gold=leaf_gold/(leaf_gold.norm(dim=-1,keepdim=True)+1e-8)
            combined=leaf_gold+0.1*root_en_pred.unsqueeze(0).expand_as(leaf_gold)
            logits=decoder(combined,ids_en_t,p_teacher=p_t)
            loss=F.mse_loss(root_en_pred,root_en_gold.detach())+F.cross_entropy(logits,ids_en_t)
            bl+=loss;n+=1
        
        if n==0: continue;(bl/n).backward()
        torch.nn.utils.clip_grad_norm_(opt2.param_groups[0]['params'],2.0);opt2.step()
        tl+=(bl/n).item();tt+=1
    
    if tt==0: continue;sched2.step()
    if ep%1==0 or ep==2:
        E_de.eval();E_en.eval();bridge.eval();decoder.eval();rf,hp=[],[]
        with torch.no_grad():
            for ids_de,ids_en in zip(val_de[:50],val_en[:50]):
                root_de=get_best_root(E_de,ids_de);root_en_pred=bridge(root_de)
                ids_en_t=torch.tensor(ids_en,device=device);T=len(ids_en)
                leaf=E_en(ids_en_t)+0.5*get_pos_emb(T);leaf=leaf/(leaf.norm(dim=-1,keepdim=True)+1e-8)
                combined=leaf+0.1*root_en_pred.unsqueeze(0).expand_as(leaf)
                logits=decoder(combined);pred=logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids_en);hp.append(pred[:T])
        b=compute_bleu(rf,hp);acc=100*sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in rf))
        print(f"  ep {ep:3d} loss={tl/tt:.4f} BLEU={b:.1f} tok_acc={acc:.1f}% time={time.time()-t0:.0f}s")
        E_de.train();E_en.train();bridge.train();decoder.train()

# ══════════════════════════════════════════
# PHASE 3: Auto-regressive wild inference
# ══════════════════════════════════════════
print(f"\n{'='*60}\nPHASE 3: Wild autoregressive inference (no gold leaves)\n{'='*60}")
E_de.eval();E_en.eval();bridge.eval();decoder.eval()
rf,hp=[],[]

for ids_de,ids_en in zip(val_de,val_en):
    root_de=get_best_root(E_de,ids_de);root_en_pred=bridge(root_de)
    T_en=len(ids_en)
    
    # Auto-regressive generation: build leaves from predicted tokens
    h=torch.zeros(d,device=device);prev_emb=torch.zeros(d,device=device)
    pred_ids=[]
    
    for t in range(min(T_en+5,MAX_LEN)):
        pos=get_pos_emb(t+1)[t] if t<MAX_LEN else get_pos_emb(MAX_LEN)[-1]
        if t==0: current_leaf=prev_emb+0.5*pos
        else:
            tok=torch.tensor([pred_ids[-1]],device=device)
            current_leaf=E_en(tok).squeeze(0)+0.5*pos
        current_leaf=current_leaf/(current_leaf.norm()+1e-8)
        
        inp=current_leaf+0.1*root_en_pred
        if t>0: inp=inp+0.3*prev_emb
        
        h_in=inp.unsqueeze(0)  # [1,d]
        h_h=h.unsqueeze(0)     # [1,d]
        h=decoder.gru(h_in,h_h).squeeze(0)
        out=decoder.W_out(h);pid=out.argmax(dim=-1).item()
        
        if pid==word2id_en.get("<pad>",0): break
        pred_ids.append(pid)
        prev_emb=E_en(torch.tensor([pid],device=device)).squeeze(0)
    
    rf.append(ids_en);hp.append(pred_ids[:T_en])

b=compute_bleu(rf,hp)
acc=100*sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in rf))
print(f"\nFINAL (wild inference): BLEU-4 = {b:.1f}  Token_Accuracy = {acc:.1f}%")

print(f"\n=== samples ===")
for i in range(5):
    de,en=val_pairs[i];ide=[word2id_de.get(w,1) for w in de[:MAX_LEN]];ien=[word2id_en.get(w,1) for w in en[:MAX_LEN]]
    if len(ide)<3 or len(ien)<3: continue
    with torch.no_grad():
        root_de=get_best_root(E_de,ide);rep=bridge(root_de);T=len(ien)
        h=torch.zeros(d,device=device);pe=torch.zeros(d,device=device);preds=[]
        for t in range(min(T+5,MAX_LEN)):
            ps=get_pos_emb(t+1)[t] if t<MAX_LEN else get_pos_emb(MAX_LEN)[-1]
            cl=pe+0.5*ps if t==0 else E_en(torch.tensor([preds[-1]],device=device)).squeeze(0)+0.5*ps
            cl=cl/(cl.norm()+1e-8);inp=cl+0.1*rep
            if t>0: inp=inp+0.3*pe
            h=decoder.gru(inp.unsqueeze(0),h.unsqueeze(0) if t>0 else torch.zeros(1,d,device=device)).squeeze(0)
            pid=decoder.W_out(h).argmax(dim=-1).item()
            if pid==word2id_en.get("<pad>",0): break
            preds.append(pid);pe=E_en(torch.tensor([pid],device=device)).squeeze(0)
    print(f"  DE: {' '.join(de[:8])}")
    print(f"  EN: {' '.join(en[:8])}")
    print(f"  PR: {' '.join(id2word_en.get(p,'?') for p in preds[:8])}")
    print()

torch.save({'E_de':E_de.state_dict(),'E_en':E_en.state_dict(),'bridge':bridge.state_dict(),'decoder':decoder.state_dict(),'w2id_de':word2id_de,'w2id_en':word2id_en},'/tmp/spr_s2_v2.pt')
print(f"Checkpoint: /tmp/spr_s2_v2.pt")
