"""
SPR v4 Experiment Grid — 8 autoencode + bridge combinations
Usage:
  python3 spr_v4_grid.py --phase auto --partition heap --l1 shared
  python3 spr_v4_grid.py --phase bridge --l1 independent --bridge token
  python3 spr_v4_grid.py --phase final --best bridge_tree_independent_token
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sentencepiece as spm, sys, os
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {}; i = 1
while i < len(sys.argv):
    if sys.argv[i].startswith('--'):
        k = sys.argv[i][2:]; v = sys.argv[i+1] if i+1 < len(sys.argv) else ''; args[k] = v; i += 2
    else: i += 1

phase = args.get('phase', 'auto'); partition = args.get('partition', 'tree')
l1_mode = args.get('l1', 'independent'); bridge_mode = args.get('bridge', 'token')
best_exp = args.get('best', '')

exp_name = f"{phase}_{partition}_{l1_mode}" + (f"_{bridge_mode}" if phase == 'bridge' else '')
print(f"device={device} SPR v4 GRID [{exp_name}]")
print("=" * 60)

sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d, MAX_LEN = sp.get_piece_size(), 128, 50
save_dir = '/mnt/nas/datasets/wmt17/checkpoints'; os.makedirs(save_dir, exist_ok=True)

print("loading data...")
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh,en=l.strip().split("\t",1)
            zh_toks=sp.encode_as_ids(zh.strip()); en_toks=sp.encode_as_ids(en.strip().lower())
            if len(zh_toks)>=2 and len(en_toks)>=2: pairs.append((zh_toks,en_toks))
train_en=[]
with open("/data/datasets/wmt14/wmt14.train.de-en") as f:
    for i,l in enumerate(f):
        if i>=50000: break
        if "\t" in l: train_en.append(sp.encode_as_ids(l.split("\t",1)[1].strip().lower()))
val_en=[sp.encode_as_ids(l.split("\t",1)[1].strip().lower()) for l in open("/data/datasets/wmt14/wmt14.validation.de-en")][:300]

n_auto_data = int(args.get("--data", "100000"))

auto_en=[ids[:MAX_LEN] for ids in train_en[:n_auto_data//2] if len(ids)>=3]
auto_zh=[zh[:MAX_LEN] for zh,_ in pairs[:n_auto_data//2] if len(zh)>=2]
all_auto=auto_en+auto_zh
bridge_pairs=pairs[-50000:]
print(f"auto={len(all_auto)} bridge={len(bridge_pairs)} val={len(val_en)}")

def heap_size(T):
    k=1
    while (1<<(k-1))<T: k+=1
    return 1<<(k-1),k
def pad_to_heap(ids,T): nl,_=heap_size(T); return torch.tensor(ids+[0]*(nl-T),device=device),nl

class BiGRU(nn.Module):
    def __init__(self,d): super().__init__(); self.enc=nn.GRU(d,d//2,bidirectional=True,batch_first=True); self.ep=nn.Linear(d,d); self.dec=nn.GRU(d,d//2,bidirectional=True,batch_first=True); self.dp=nn.Linear(d,d)
    def fe(self,x): return self.ep(self.enc(x)[0])
    def fd(self,x): return self.dp(self.dec(x)[0])

L0=nn.Embedding(V,d).to(device); nn.init.normal_(L0.weight,0,0.02)
L1_en=BiGRU(d).to(device); L1_zh=BiGRU(d).to(device) if l1_mode=='independent' else L1_en
if l1_mode=='shared': L1_zh=L1_en
Wb=nn.Linear(d,d).to(device); nn.init.normal_(Wb.weight,0,0.01); nn.init.zeros_(Wb.bias)

def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def bleu(rf,hp):
    C=Counter;ps=[]
    for n in range(1,5):
        m,t=0,0
        for r,h in zip(rf,hp): rc=C(ng(r,n));hc=C(ng(h,n));t+=sum(hc.values());m+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(m/max(t,1) if t>0 else 1.0)
    bp=max(0,min(1,sum(1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0)/max(len(rf),1)))
    return math.exp(min(999,sum(math.log(max(p,1e-10)) for p in ps)/4))*100

val_data=[(ids[:MAX_LEN],ids[:MAX_LEN]) for ids in val_en if len(ids)>=2]

def tree_loss(ctx,ids_target,nL,wgt):
    h=ctx.squeeze(0);T=len(ids_target)
    if T<4 or nL<3: return torch.tensor(0.0,device=device)
    loss=torch.tensor(0.0,device=device);ls=nL-1
    for ni in range(nL-1):
        f,l=T,0
        for t in range(T):
            node=t+ls
            while node>ni: node=(node-1)//2
            if node==ni:
                if t<f:f=t
                if t>l:l=t
        if f<l:
            op=h[f:l+1].mean(dim=0);nk=l-f+1
            loss=loss+wgt*F.cross_entropy((op@L0.weight.T).unsqueeze(0).expand(nk,-1),ids_target[f:l+1])
    return loss

def heap_loss(ctx,ids_target):
    h=ctx.squeeze(0);T=len(ids_target)
    if T<4: return torch.tensor(0.0,device=device)
    np2=T//2; eh=h[0:2*np2:2]; oh=h[1:2*np2:2]
    return 0.1*(F.cross_entropy(eh@L0.weight.T,ids_target[1:2*np2:2])+F.cross_entropy(oh@L0.weight.T,ids_target[0:2*np2:2]))

if phase=='auto':
    AB_EPOCHS=50 if partition=='heap' else 200; t0=time.time()
    params=[L0]; [params.append(c) for c in [L1_en,L1_zh] if c not in params]
    opt=torch.optim.Adam(sum([list(p.parameters()) for p in params],[]),lr=0.003)
    print(f"PHASE AUTO: partition={partition} l1={l1_mode} epochs={AB_EPOCHS}")

    for ep in range(AB_EPOCHS):
        L0.train();L1_en.train();L1_zh.train()
        random.shuffle(all_auto);tl,ti=0,0
        for bi in range(0,5000,16):
            batch=all_auto[bi:bi+16]
            if not batch: continue
            opt.zero_grad();bl,ns=torch.tensor(0.0,device=device),0
            for ids in batch:
                T=min(len(ids),MAX_LEN);ids=ids[:T]
                ids_d=ids[:];k=0
                for i in range(len(ids_d)):
                    if random.random()>0.25 or k<2:k+=1
                    else:ids_d[i]=1
                ids_pad,nL=pad_to_heap(ids_d,T);ids_tgt,_=pad_to_heap(ids,T)
                with torch.no_grad(): emb=L0(ids_pad).unsqueeze(0)
                ctx=L1_en.fe(emb);dec=L1_en.fd(ctx)
                loss=F.cross_entropy(dec.squeeze(0)[:T]@L0.weight.T,ids_tgt[:T])
                if partition=='tree': loss=loss+tree_loss(ctx,ids_tgt[:T],nL,0.001)
                else: loss=loss+heap_loss(ctx,ids_tgt[:T])
                bl+=loss;ns+=1
            if ns:(bl/ns).backward();torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],2.0);opt.step()
            ti+=1;tl+=(bl/ns).item() if ns else 0
        if ep%max(AB_EPOCHS//6,1)==0 or ep==AB_EPOCHS-1:
            L0.eval();L1_en.eval();rf,hp=[],[]
            with torch.no_grad():
                for ids,_ in val_data[:30]:
                    T=min(len(ids),MAX_LEN);ids_pad,_=pad_to_heap(ids[:T],T)
                    emb=L0(ids_pad).unsqueeze(0)
                    logits=L1_en.fd(L1_en.fe(emb)).squeeze(0)[:T]@L0.weight.T
                    rf.append(ids[:T]);hp.append(logits.argmax(dim=-1).cpu().tolist())
            b=bleu(rf,hp);ta=100*sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi)/max(1,sum(len(r) for r in rf))
            print(f"  ep {ep:3d} loss={tl/ti:.4f} BLEU={b:.1f} tok_acc={ta:.1f}% {time.time()-t0:.0f}s")

    L1_en.eval();rf_r,hp_r=[],[]
    for ids in [s[:MAX_LEN] for s in val_en[:50] if len(s)>=3]:
        T=min(len(ids),MAX_LEN);ids=ids[:T]
        ids_b=ids[:];k=0
        for i in range(len(ids_b)):
            if random.random()>0.25 or k<2:k+=1
            else:ids_b[i]=1
        ids_pad,_=pad_to_heap(ids_b,T)
        ctx=L1_en.fe(L0(ids_pad).unsqueeze(0))
        pred=L1_en.fd(ctx).squeeze(0)[:T]@L0.weight.T
        rf_r.append(ids[:T]);hp_r.append(pred.argmax(dim=-1).cpu().tolist())
    rb=bleu(rf_r,hp_r)
    torch.save({'L0':L0.state_dict(),'l1_mode':l1_mode,'took':time.time()-t0,'partition':partition,'repair_bleu':rb},f'{save_dir}/{exp_name}.pt')
    print(f"Repair BLEU={rb:.1f} -> {save_dir}/{exp_name}.pt")

elif phase=='bridge':
    ckpt_path=f'{save_dir}/auto_{partition}_{l1_mode}.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path=f'{save_dir}/auto_heap_{l1_mode}.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path=f'{save_dir}/auto_tree_{l1_mode}.pt'
    ckpt=torch.load(ckpt_path,map_location=device)
    L0.load_state_dict(ckpt['L0']);rb_old=ckpt.get('repair_bleu','?')
    print(f"Loaded {ckpt_path} repair_BLEU={rb_old}")
    C_EPOCHS=100;t0=time.time()
    print(f"PHASE BRIDGE: l1={l1_mode} bridge={bridge_mode} epochs={C_EPOCHS}")
    for p in L0.parameters():p.requires_grad=False
    for p in L1_en.enc.parameters():p.requires_grad=False
    for p in L1_en.ep.parameters():p.requires_grad=False
    if l1_mode=='independent':
        for p in L1_zh.enc.parameters():p.requires_grad=False
        for p in L1_zh.ep.parameters():p.requires_grad=False
    opt=torch.optim.Adam(list(Wb.parameters())+list(L1_zh.dec.parameters())+list(L1_zh.dp.parameters()),lr=0.003)
    for ep in range(C_EPOCHS):
        Wb.train();tl,ti=0,0;random.shuffle(bridge_pairs)
        for bi in range(0,3000,16):
            batch=bridge_pairs[bi:bi+16]
            if not batch: continue
            opt.zero_grad();bl,ns=torch.tensor(0.0,device=device),0
            for zh,en in batch:
                Te,Tz=min(len(en),MAX_LEN),min(len(zh),MAX_LEN);T=min(Te,Tz)
                if T<2: continue
                en_ids,zh_ids=en[:T],zh[:T]
                ids_pad_en,_=pad_to_heap(en_ids,T);ids_pad_zh,_=pad_to_heap(zh_ids,T)
                with torch.no_grad():
                    he=L1_en.fe(L0(ids_pad_en).unsqueeze(0)).squeeze(0)[:T]
                    hz=L1_zh.fe(L0(ids_pad_zh).unsqueeze(0)).squeeze(0)[:T]
                if bridge_mode=='token':
                    ha=Wb(he)
                    dz=L1_zh.fd(ha.unsqueeze(0))
                    loss=F.cross_entropy(dz.squeeze(0)[:T]@L0.weight.T,ids_pad_zh[:T])+0.5*F.mse_loss(ha,hz)
                else:
                    loss=F.mse_loss(Wb(he.mean(dim=0)),hz.mean(dim=0))
                bl+=loss;ns+=1
            if ns:(bl/ns).backward();opt.step()
            ti+=1;tl+=(bl/ns).item() if ns else 0
        if ep%30==0 or ep==C_EPOCHS-1:
            t_elapsed=time.time()-t0
            if bridge_mode=='token':
                L1_zh.eval();Wb.eval();rf,hp=[],[]
                with torch.no_grad():
                    for zh,en in bridge_pairs[-300:-200]:
                        Te,Tz=min(len(en),MAX_LEN),min(len(zh),MAX_LEN);T=min(Te,Tz)
                        if T<2:continue
                        ids_pad_en,_=pad_to_heap(en[:T],T)
                        he=L1_en.fe(L0(ids_pad_en).unsqueeze(0)).squeeze(0)[:T]
                        ha=Wb(he)
                        dz=L1_zh.fd(ha.unsqueeze(0))
                        lo=dz.squeeze(0)[:T]@L0.weight.T
                        rf.append(zh[:T]);hp.append(lo.argmax(dim=-1).cpu().tolist())
                if len(rf)>=5:
                    print(f"  ep{ep:4d} loss={tl/ti:.4f} bridge_BLEU={bleu(rf,hp):.1f} {t_elapsed:.0f}s")
                    # Single-word bridge test (no word order)
                    if ep%30==0:
                        sw_ok,sw_n,shown=0,0,0
                        for zh_i,en_i in bridge_pairs[-200:-180]:
                            for t in range(min(min(len(en_i),len(zh_i)),3)):
                                e_t=torch.tensor([en_i[t]],device=device)
                                with torch.no_grad():
                                    he_t=L1_en.fe(L0(e_t).unsqueeze(0)).squeeze(0)
                                    ha_t=Wb(he_t) if bridge_mode=='token' else Wb(he_t.mean(0))  
                                    dz_t=L1_zh.fd(ha_t.unsqueeze(0))
                                    pid=dz_t.squeeze(0)@L0.weight.T
                                    pred_id=pid.argmax(-1).item()
                                    if pred_id==zh_i[t]:sw_ok+=1
                                    sw_n+=1
                                    if shown<5:
                                        print(f"    {sp.decode_ids([en_i[t]]):15s} → {sp.decode_ids([pred_id]):15s} (gold: {sp.decode_ids([zh_i[t]]):15s})");shown+=1
                        if sw_n>0: print(f"    single-word acc: {sw_ok}/{sw_n}={100*sw_ok/sw_n:.1f}%")
                L1_zh.train();Wb.train()
            else:
                print(f"  ep{ep:4d} loss={tl/ti:.6f} {t_elapsed:.0f}s")
    torch.save({'Wb':Wb.state_dict(),'l1_mode':l1_mode,'bridge_mode':bridge_mode,'took':time.time()-t0,'partition':partition},f'{save_dir}/{exp_name}.pt')
    print(f"Saved: {save_dir}/{exp_name}.pt")

elif phase=='final':
    ckpt_path=f'{save_dir}/bridge_{partition}_{l1_mode}_{bridge_mode}.pt'
    if not os.path.exists(ckpt_path): ckpt_path=f'{save_dir}/{best_exp}.pt'
    print(f"Loading {ckpt_path}")
    ckpt=torch.load(ckpt_path,map_location=device)
    Wb.load_state_dict(ckpt['Wb'])
    bm=ckpt.get('bridge_mode','token');lm=ckpt.get('l1_mode','independent')
    ack=f'{save_dir}/auto_{ckpt.get("partition","tree")}_{lm}.pt'
    if not os.path.exists(ack): ack=f'{save_dir}/auto_heap_{lm}.pt'
    if os.path.exists(ack): L0.load_state_dict(torch.load(ack,map_location=device)['L0'])
    L0.eval();L1_en.eval();L1_zh.eval();Wb.eval()
    print(f"\nEN->ZH samples [{best_exp}]:")
    for zh,en in bridge_pairs[-30:-15]:
        Te,Tz=min(len(en),MAX_LEN),min(len(zh),MAX_LEN);T=min(Te,Tz)
        if T<3: continue
        with torch.no_grad():
            ids_pad_en,_=pad_to_heap(en[:T],T)
            he=L1_en.fe(L0(ids_pad_en).unsqueeze(0)).squeeze(0)[:T]
            ha=Wb(he) if bm=='token' else Wb(he.mean(dim=0)).unsqueeze(0).expand(T,-1)
            dz=L1_zh.fd(ha.unsqueeze(0))
            lo=dz.squeeze(0)[:T]@L0.weight.T
            pred=sp.decode_ids(lo.argmax(dim=-1).cpu().tolist())
        print(f"  EN: {sp.decode_ids(en[:T])[:80]}")
        print(f"  ZH: {sp.decode_ids(zh[:T])[:80]}")
        print(f"  PR: {pred[:80]}")
        print()
