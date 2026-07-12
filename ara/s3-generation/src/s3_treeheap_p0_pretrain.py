#!/usr/bin/env python3
"""P0 real-Chinese next-span pretraining for the TreeHeap seq2seq stack."""
from __future__ import annotations

import argparse, itertools, json, math, random, time
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset, DataLoader

from s3_wmt_treeheap_seq2seq import TreeHeapSeq2Seq, FlatSeq2Seq, BowSeq2Seq, ce


SOURCES = {"news": 0.50, "wiki": 0.30, "web": 0.20}


def files(root: Path, source: str) -> List[Path]:
    base = root / "Chinese-Train-Datasets"
    if source == "news": return [base / "new2016zh/news2016zh_train.json"]
    if source == "wiki": return sorted((base / "wiki_zh").rglob("wiki_*"))
    if source == "web": return [base / "webtext2019zh/webtext_zh_train.json"]
    raise ValueError(source)


def text_from(row: dict, source: str) -> str:
    if source == "wiki": return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
    return (str(row.get("title", "")) + "\n" + str(row.get("content", ""))).strip()


def document_stream(root: Path, source: str) -> Iterator[str]:
    paths = files(root, source)
    while True:
        for path in paths:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try: row = json.loads(line)
                    except json.JSONDecodeError: continue
                    text = text_from(row, source)
                    if len(text) >= 32: yield text


class MixedDocs:
    def __init__(self, root: Path, seed: int):
        self.rng = random.Random(seed)
        self.names = list(SOURCES); self.weights = [SOURCES[x] for x in self.names]
        self.streams = {name: document_stream(root, name) for name in self.names}
    def __iter__(self) -> Iterator[str]:
        while True:
            name = self.rng.choices(self.names, self.weights)[0]
            yield next(self.streams[name])


class SpanBlocks(IterableDataset):
    def __init__(self, root: Path, spm_path: str, seed: int, context: int, target: int):
        self.root, self.spm_path, self.seed, self.context, self.target = root, spm_path, seed, context, target
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        sp = spm.SentencePieceProcessor(model_file=self.spm_path); eos = sp.eos_id(); total = self.context + self.target; buf: List[int] = []
        for text in MixedDocs(self.root, self.seed):
            buf.extend(sp.encode(text, out_type=int)); buf.append(eos)
            while len(buf) >= total:
                chunk, buf = buf[:total], buf[total:]
                yield torch.tensor(chunk[:self.context]), torch.tensor(chunk[self.context:])


def collate(batch): return torch.stack([x for x,_ in batch]), torch.stack([y for _,y in batch])


def train_tokenizer(root: Path, out: Path, samples: int, vocab: int, seed: int) -> Path:
    corpus = out / "tokenizer_sample.txt"; out.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8") as f:
        for _, text in zip(range(samples), MixedDocs(root, seed)): f.write(text.replace("\n", " ") + "\n")
    prefix = out / f"p0_zh_{vocab}"
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(prefix), vocab_size=vocab, model_type="bpe", character_coverage=0.9995, bos_id=1, eos_id=2, pad_id=-1, unk_id=0)
    return prefix.with_suffix(".model")


def eval_nll(model, loader, device, bos, pad, limit=32):
    model.eval(); total = count = 0.0; samples=[]
    with torch.no_grad():
        for i,(src,tgt) in enumerate(loader):
            src,tgt=src.to(device),tgt.to(device); length=torch.full((src.shape[0],),src.shape[1],device=device,dtype=torch.long)
            logits=model(src,length,tgt,bos); total += float(ce(logits,tgt,pad).item()); count += 1
            if i+1>=limit: break
    return total/max(1,count)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/home/nio/datasets/pretrain"); ap.add_argument("--evidence-dir",default="ara/s3-generation/evidence/s3_p0_world_observation_smoke"); ap.add_argument("--mode",choices=["tokenizer","train"],default="train"); ap.add_argument("--spm-model"); ap.add_argument("--tokenizer-samples",type=int,default=50000); ap.add_argument("--vocab",type=int,default=16000); ap.add_argument("--seed",type=int,default=17); ap.add_argument("--model",choices=["treeheap","flat_seq","bow"],default="treeheap"); ap.add_argument("--context",type=int,default=64); ap.add_argument("--target",type=int,default=32); ap.add_argument("--batch",type=int,default=64); ap.add_argument("--steps",type=int,default=500); ap.add_argument("--valid-every",type=int,default=100); ap.add_argument("--valid-batches",type=int,default=32); ap.add_argument("--dim",type=int,default=192); ap.add_argument("--hidden",type=int,default=192); ap.add_argument("--lr",type=float,default=0.002); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); args=ap.parse_args()
    root,out=Path(args.root),Path(args.evidence_dir); out.mkdir(parents=True,exist_ok=True)
    if args.mode=="tokenizer":
        model=train_tokenizer(root,out,args.tokenizer_samples,args.vocab,args.seed); print(model); return
    if not args.spm_model: raise ValueError("--spm-model is required for train")
    sp=spm.SentencePieceProcessor(model_file=args.spm_model); pad,vocab,bos,eos=sp.get_piece_size(),sp.get_piece_size()+1,sp.bos_id(),sp.eos_id()
    cls={"treeheap":TreeHeapSeq2Seq,"flat_seq":FlatSeq2Seq,"bow":BowSeq2Seq}[args.model]; model=cls(vocab,args.dim,args.hidden).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    train=iter(DataLoader(SpanBlocks(root,args.spm_model,args.seed,args.context,args.target),batch_size=args.batch,collate_fn=collate,num_workers=0)); valid=DataLoader(SpanBlocks(root,args.spm_model,args.seed+999,args.context,args.target),batch_size=args.batch,collate_fn=collate,num_workers=0)
    trace=[]; started=time.time()
    for step in range(1,args.steps+1):
        src,tgt=next(train); src,tgt=src.to(args.device),tgt.to(args.device); length=torch.full((src.shape[0],),args.context,device=args.device,dtype=torch.long); logits=model(src,length,tgt,bos); loss=ce(logits,tgt,pad); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%args.valid_every==0 or step==args.steps:
            val=eval_nll(model,valid,args.device,bos,pad,args.valid_batches); row={"step":step,"train_nll":float(loss.item()),"valid_nll":val,"valid_ppl":math.exp(min(20,val))}; trace.append(row); torch.save({"state_dict":model.state_dict(),"config":vars(args),"step":step,"trace":trace},out/"checkpoint_latest.pt"); print(json.dumps(row),flush=True)
    src,tgt=next(train); src=src.to(args.device); length=torch.full((src.shape[0],),args.context,device=args.device,dtype=torch.long); pred=model.generate(src,length,bos,eos,args.target).cpu(); examples=[]
    for x,y,z in zip(src[:8].cpu(),tgt[:8],pred[:8]): examples.append({"context":sp.decode(x.tolist()),"reference":sp.decode(y.tolist()),"continuation":sp.decode(z.tolist())})
    summary={"claim":"S3-P0-WORLD-OBS-C01","seconds":time.time()-started,"model":args.model,"parameters":sum(p.numel() for p in model.parameters()),"config":vars(args),"trace":trace,"examples":examples}; torch.save({"state_dict":model.state_dict(),"config":vars(args)},out/"checkpoint.pt"); (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8"); (out/"trace.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in trace)+"\n",encoding="utf8"); (out/"examples.json").write_text(json.dumps(examples,indent=2,ensure_ascii=False),encoding="utf8"); (out/"README.md").write_text("# P0 TreeHeap World-Observation Pretraining\n\nReal Chinese next-span self-supervision over news, wiki, and web text.\n",encoding="utf8")

if __name__=="__main__": main()
