"""出度对比: K=2 (31 nodes) vs K=4 (341 nodes), BLI P@1"""
import torch, torch.nn as nn, torch.nn.functional as F, random, json, time
import sentencepiece as spm
device='cuda'; td=5; V=16000; tau=0.07
sp=spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
def ok(ids): return all(x!=0 for x in ids)

with open('/workspace/multi_sense_anchors.json') as f: dict_data=json.load(f)
pairs=set()
for en,senses in dict_data.items():
    for zh,pos in senses:
        if pos in ('n','v','a','adj','adv','vi','vt') and 1<=len(zh)<=6:
            ei=sp.encode_as_ids(en); zi=sp.encode_as_ids(zh)
            if ok(ei) and ok(zi) and 2<=len(en)<=15: pairs.add((en,zh))
pairs=list(pairs); pairs.sort(key=lambda x:sp.encode_as_ids(x[0])[0])
pairs=pairs[:25000]; random.seed(42); random.shuffle(pairs)
train_pairs=pairs[:20000]; test_pairs=pairs[20000:21500]
print(f"Train: {len(train_pairs)} Test: {len(test_pairs)}")
from core import TreeNodes, HeapWorld

def run(branching, label):
    t0=time.time()
    tree=TreeNodes(dim=128,depth=td,init='rand',combine='add',vocab=V,branching=branching).to(device)
    hw=HeapWorld(vocab=V,dim=128,tree=tree).to(device)
    nn.init.normal_(hw.embedding.weight,0,0.02)
    for p in hw.embedding.parameters(): p.requires_grad=True
    for p in tree.parameters(): p.requires_grad=True
    for p in hw.merge.parameters(): p.requires_grad=True
    
    def heap_world(tok_ids):
        return hw.forward(tok_ids.unsqueeze(0)).squeeze(0)
    
    tr_ts=[(en,zh,torch.tensor(sp.encode_as_ids(en),device=device),torch.tensor(sp.encode_as_ids(zh),device=device)) for en,zh in train_pairs]
    N=len(tr_ts); zh_tr=torch.zeros(N,128,device=device)
    with torch.no_grad():
        for ai,(_,_,_,zi) in enumerate(tr_ts): zh_tr[ai]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)
    Nt=len(test_pairs); en_t=torch.zeros(Nt,128,device=device); zh_t=torch.zeros(Nt,128,device=device)
    with torch.no_grad():
        for i,(enw,zhw) in enumerate(test_pairs):
            ei=torch.tensor(sp.encode_as_ids(enw),device=device); zi=torch.tensor(sp.encode_as_ids(zhw),device=device)
            en_t[i]=F.normalize(heap_world(ei).mean(dim=0),dim=-1); zh_t[i]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)
    def p1():
        with torch.no_grad():
            s=en_t@zh_t.T/tau; return 100*(s.argmax(-1)==torch.arange(Nt,device=device)).float().mean().item()
    p0=p1(); params=list(hw.embedding.parameters())+list(hw.merge.parameters())+[p for pn in tree.embeddings for p in pn.parameters()]
    opt=torch.optim.Adam(params,lr=0.003); B=128; best=p0
    for ep in range(60):
        idx=list(range(N)); random.shuffle(idx)
        for bi in range(0,N,B):
            batch=idx[bi:bi+B]; opt.zero_grad(); losses=[]
            for ai in batch:
                ei=tr_ts[ai][2]; hw_e=F.normalize(heap_world(ei).mean(dim=0),dim=-1)
                logits=(hw_e.unsqueeze(0)@zh_tr.T)/tau; losses.append(F.cross_entropy(logits,torch.tensor([ai],device=device)))
            if losses: (sum(losses)/len(losses)).backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if ep%20==0 or ep==59:
            with torch.no_grad():
                for ai,(_,_,_,zi) in enumerate(tr_ts): zh_tr[ai]=F.normalize(heap_world(zi).mean(dim=0),dim=-1)
            p=p1(); best=max(best,p)
    print(f"  {label}: nodes={tree.total_nodes} paths={branching**(td-1)} init={p0:.1f}% best={best:.1f}% {time.time()-t0:.0f}s")

run(2, "K=2")
run(4, "K=4")
print("Done.")
