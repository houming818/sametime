"""d=128/256, contiguous freq split, BLI P@1"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time, sentencepiece as spm
device='cuda'; td=5; V=16000; tau=0.07
sp=spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x!=0 for x in ids)

with open('/workspace/multi_sense_anchors.json') as f: dict_data=json.load(f)
pairs=set()
for en,senses in dict_data.items():
    for zh,pos in senses:
        if pos in ('n','v','a','adj','adv','vi','vt') and 1<=len(zh)<=8:
            ei=sp.encode_as_ids(en); zi=sp.encode_as_ids(zh)
            if ok(ei) and ok(zi) and 2<=len(en)<=15: pairs.add((en,zh))
pairs=list(pairs); pairs.sort(key=lambda x:sp.encode_as_ids(x[0])[0])
pairs = pairs[:25000]  # top 25K first, THEN shuffle
import random; random.seed(42); random.shuffle(pairs)
print(f"ECDICT: {len(pairs)} (top 25K by freq, shuffled)")

def run(d, n_train):
    t0=time.time()
    train=pairs[:n_train]; test=pairs[n_train:n_train+1500]
    
    L0=nn.Embedding(V,d).to(device); t_nodes=nn.ModuleList([nn.Embedding(2**i,d).to(device) for i in range(td)])
    t_merge=nn.Linear(d,d).to(device)
    if d==128:
        ckpt=torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt',map_location=device,weights_only=True)
        L0.load_state_dict(ckpt['L0']); t_merge.load_state_dict(ckpt['t_merge'])
        for i,tn in enumerate(t_nodes): tn.load_state_dict(ckpt['t_nodes'][i])
    else:
        nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)
        nn.init.normal_(L0.weight,0,0.02)
        for tn in t_nodes: nn.init.normal_(tn.weight,0,0.1)
    for p in L0.parameters(): p.requires_grad=True
    for p in t_merge.parameters(): p.requires_grad=True
    for tn in t_nodes:
        for p in tn.parameters(): p.requires_grad=True

    def hw(tok_ids):
        w=torch.zeros(len(tok_ids),d,device=device)
        for l in range(td):
            nidx=torch.clamp(tok_ids//(V//(2**l)),0,(2**l)-1) if l>0 else torch.zeros_like(tok_ids)
            w=w+t_nodes[l](nidx)
        w=F.normalize(w,dim=-1); t=F.normalize(L0.weight[tok_ids],dim=-1)
        tL,tR=t[...,:d//2],t[...,d//2:]; wL,wR=w[...,:d//2],w[...,d//2:]
        return t_merge(torch.cat([tL*wL-tR*wR,tL*wR+tR*wL],-1))

    tr_ts=[(en,zh,torch.tensor(sp.encode_as_ids(en),device=device),torch.tensor(sp.encode_as_ids(zh),device=device)) for en,zh in train]
    N=len(tr_ts); zh_tr=torch.zeros(N,d,device=device)
    with torch.no_grad():
        for ai,(_,_,_,zi) in enumerate(tr_ts): zh_tr[ai]=F.normalize(hw(zi).mean(dim=0),dim=-1)
    Nt=len(test); en_t=torch.zeros(Nt,d,device=device); zh_t=torch.zeros(Nt,d,device=device)
    with torch.no_grad():
        for i,(enw,zhw) in enumerate(test):
            ei=torch.tensor(sp.encode_as_ids(enw),device=device); zi=torch.tensor(sp.encode_as_ids(zhw),device=device)
            en_t[i]=F.normalize(hw(ei).mean(dim=0),dim=-1); zh_t[i]=F.normalize(hw(zi).mean(dim=0),dim=-1)
    def p1():
        with torch.no_grad():
            s=en_t@zh_t.T/tau; return 100*(s.argmax(-1)==torch.arange(Nt,device=device)).float().mean().item()
    p0=p1(); params=list(L0.parameters())+list(t_merge.parameters())+[p for tn in t_nodes for p in tn.parameters()]
    opt=torch.optim.Adam(params,lr=0.003); B=128; best=p0
    for ep in range(60):
        idx=list(range(N)); random.shuffle(idx)
        for bi in range(0,N,B):
            batch=idx[bi:bi+B]; opt.zero_grad(); losses=[]
            for ai in batch:
                ei=tr_ts[ai][2]; hw_e=F.normalize(hw(ei).mean(dim=0),dim=-1)
                logits=(hw_e.unsqueeze(0)@zh_tr.T)/tau; losses.append(F.cross_entropy(logits,torch.tensor([ai],device=device)))
            if losses: (sum(losses)/len(losses)).backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if ep%20==0 or ep==59:
            with torch.no_grad():
                for ai,(_,_,_,zi) in enumerate(tr_ts): zh_tr[ai]=F.normalize(hw(zi).mean(dim=0),dim=-1)
            p=p1(); best=max(best,p)
    return p0,best,time.time()-t0

for d,n in [(128,10000),(128,20000),(256,10000),(256,20000)]:
    p0,pb,tt=run(d,n)
    print(f"  d={d} {n//1000}K init={p0:.1f}% best={pb:.1f}% {tt:.0f}s")
print("Done.")
