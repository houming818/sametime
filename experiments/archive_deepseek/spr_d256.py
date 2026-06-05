"""维度缩放: d=256, 20K train, 1500 test"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time
import sentencepiece as spm
device='cuda'; td=5; V=16000; tau=0.07
sp=spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x!=0 for x in ids)
d=256

with open('/workspace/multi_sense_anchors.json') as f: dict_data=json.load(f)
pairs_all=set()
for en,senses in dict_data.items():
    for zh,pos in senses:
        if pos in ('n','v','a','adj','adv','vi','vt') and 1<=len(zh)<=8:
            ei=sp.encode_as_ids(en); zi=sp.encode_as_ids(zh)
            if ok(ei) and ok(zi) and 2<=len(en)<=15: pairs_all.add((en,zh))
pairs_all=list(pairs_all); pairs_all.sort(key=lambda x:sp.encode_as_ids(x[0])[0])
pairs_all=pairs_all[:25000]; random.seed(42); random.shuffle(pairs_all)
train_pairs=pairs_all[:20000]; test_pairs=pairs_all[20000:21500]
print(f"Train: {len(train_pairs)}  Test: {len(test_pairs)}  d={d}")

L0=nn.Embedding(V,d).to(device); t_nodes=nn.ModuleList([nn.Embedding(2**i,d).to(device) for i in range(td)])
t_merge=nn.Linear(d,d).to(device); nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)
nn.init.normal_(L0.weight,0,0.02)
for tn in t_nodes: nn.init.normal_(tn.weight,0,0.1)

def heap_world(tok_ids):
    w=torch.zeros(len(tok_ids),d,device=device)
    for l in range(td):
        nidx=torch.clamp(tok_ids//(V//(2**l)),0,(2**l)-1) if l>0 else torch.zeros_like(tok_ids)
        w=w+t_nodes[l](nidx)
    w=F.normalize(w,dim=-1); t=F.normalize(L0.weight[tok_ids],dim=-1)
    tL,tR=t[...,:d//2],t[...,d//2:]; wL,wR=w[...,:d//2],w[...,d//2:]
    return t_merge(torch.cat([tL*wL-tR*wR,tL*wR+tR*wL],-1))

train_ts=[(en,zh,torch.tensor(sp.encode_as_ids(en),device=device),torch.tensor(sp.encode_as_ids(zh),device=device)) for en,zh in train_pairs]
N=len(train_ts); zh_train=torch.zeros(N,d,device=device)
with torch.no_grad():
    for ai,(_,_,_,zi) in enumerate(train_ts): zh_train[ai]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)

def p1():
    Nt=len(test_pairs); en=torch.zeros(Nt,d,device=device); zh=torch.zeros(Nt,d,device=device)
    with torch.no_grad():
        for i,(enw,zhw) in enumerate(test_pairs):
            ei=torch.tensor(sp.encode_as_ids(enw),device=device); zi=torch.tensor(sp.encode_as_ids(zhw),device=device)
            en[i]=F.normalize(heap_world(ei).mean(dim=0),dim=-1); zh[i]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)
        sims=en@zh.T/tau; return 100*(sims.argmax(-1)==torch.arange(Nt,device=device)).float().mean().item()

t0=time.time(); p1_0=p1(); print(f"init P@1={p1_0:.1f}%")
trainable=list(L0.parameters())+list(t_merge.parameters())+[p for tn in t_nodes for p in tn.parameters()]
opt=torch.optim.Adam(trainable,lr=0.003); B=128; best=p1_0

for ep in range(60):
    indices=list(range(N)); random.shuffle(indices)
    for bi in range(0,N,B):
        batch=indices[bi:bi+B]; opt.zero_grad()
        losses=[]
        for ai in batch:
            ei=train_ts[ai][2]; hw=F.normalize(heap_world(ei).mean(dim=0),dim=-1)
            logits=(hw.unsqueeze(0)@zh_train.T)/tau; losses.append(F.cross_entropy(logits,torch.tensor([ai],device=device)))
        if losses: (sum(losses)/len(losses)).backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
    if ep%10==0 or ep==59:
        with torch.no_grad():
            for ai,(_,_,_,zi) in enumerate(train_ts): zh_train[ai]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)
        p=p1(); best=max(best,p); print(f"  ep {ep:3d} P@1={p:.1f}% {time.time()-t0:.0f}s")

print(f"\nd={d}: init={p1_0:.1f}% best={best:.1f}%  tt={time.time()-t0:.0f}s")
