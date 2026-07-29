#!/usr/bin/env python3
"""C14 autoregressive decoder whose only target history is a TreeHeap."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_lifting_pump_wmt as lift
import s3_stone1_c10_raw_train as atomic
import s3_stone1_c12_hstate_unfold as c12
import s3_wmt_treeheap_seq2seq as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/nio/datasets/wmt17/train.zh-en")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt17/sp_bpe.model")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_stone1_c14_target_tree_autoregressive")
    p.add_argument("--seed", type=int, default=75034)
    p.add_argument("--train-samples", type=int, default=5000)
    p.add_argument("--valid-samples", type=int, default=500)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--max-scan", type=int, default=100000)
    p.add_argument("--min-len", type=int, default=8)
    p.add_argument("--max-len", type=int, default=24)
    p.add_argument("--heap-width", type=int, default=32)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class TargetTreeState:
    def __init__(self, levels, masks, details):
        self.levels = levels  # leaf -> root
        self.masks = masks
        self.details = details


class TargetTreeDecoder(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, heap_width: int,
                 pad: int, bos: int):
        super().__init__()
        self.encoder = lift.LiftingEncoder(vocab, dim, heap_width, pad)
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad, self.bos = pad, bos
        self.depth = nn.Embedding(self.depths + 1, dim)
        self.path_projection = nn.Linear(self.depths, dim, bias=False)
        self.target_score = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        self.target_query = nn.Sequential(
            nn.LayerNorm(5 * dim), nn.Linear(5 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.source_stop = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        self.source_branch = nn.Linear(dim, dim, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, vocab),
        )

    def zero_target(self, batch: int, device, dtype) -> TargetTreeState:
        levels, masks, details = [], [], []
        for depth in range(self.depths + 1):
            nodes = self.heap_width >> depth
            levels.append(torch.zeros(batch, nodes, self.encoder.embedding.embedding_dim,
                                      device=device, dtype=dtype))
            masks.append(torch.zeros(batch, nodes, device=device, dtype=torch.bool))
            if depth < self.depths:
                details.append(torch.zeros(batch, nodes >> 1,
                                           self.encoder.embedding.embedding_dim,
                                           device=device, dtype=dtype))
        return TargetTreeState(levels, masks, details)

    def next_path(self, step: int, device, dtype) -> torch.Tensor:
        code = torch.zeros(self.depths, device=device, dtype=dtype)
        for bit in range(self.depths):
            shift = self.depths - bit - 1
            code[bit] = ((step >> shift) & 1) * 2.0 - 1.0
        return self.path_projection(code)

    def read_target(self, state: TargetTreeState, step: int,
                    intervention: str = "native") -> torch.Tensor:
        batch = state.levels[0].shape[0]
        zero = state.levels[0].new_zeros(batch, state.levels[0].shape[-1])
        if intervention == "target_zero" or step == 0:
            root = zero
            path_context = zero
        else:
            last = step - 1
            path = []
            for depth in range(self.depths, -1, -1):
                index = last >> depth
                node = state.levels[depth][:, index]
                if intervention == "target_root_only" and depth != self.depths:
                    node = torch.zeros_like(node)
                path.append(node + self.depth.weight[self.depths - depth][None])
            stacked = torch.stack(path, dim=1)
            root = state.levels[-1][:, 0]
            seed = root + self.encoder.embedding.weight[self.bos][None]
            score = self.target_score(torch.cat(
                (seed[:, None].expand_as(stacked), stacked), dim=-1,
            )).squeeze(-1)
            weight = score.softmax(-1)
            path_context = (weight[:, :, None] * stacked).sum(1)
        bos = self.encoder.embedding.weight[self.bos][None].expand_as(root)
        address = self.next_path(step, root.device, root.dtype)[None].expand_as(root)
        feature = torch.cat((root, path_context, bos, root - path_context,
                             root * path_context + address), dim=-1)
        return self.target_query(feature)

    def read_source(self, query: torch.Tensor, levels, masks,
                    route_mode: str = "native"):
        active = torch.ones(query.shape[0], 1, device=query.device, dtype=query.dtype)
        context = torch.zeros_like(query)
        route = []
        for depth, (nodes, valid) in enumerate(zip(levels, masks)):
            active = active * valid.to(active.dtype)
            last = depth == len(levels) - 1
            if last or route_mode == "force_root":
                stop = torch.ones_like(active)
            else:
                q = query[:, None].expand_as(nodes)
                depth_state = self.depth.weight[depth][None, None]
                stop = torch.sigmoid(self.source_stop(
                    torch.cat((q, nodes + depth_state), dim=-1),
                ).squeeze(-1))
            stopped = active * stop
            context = context + (stopped[:, :, None] * nodes).sum(1)
            route.append(stopped.sum(1).mean())
            if last or route_mode == "force_root":
                break
            expand = active * (1.0 - stop)
            children = levels[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            scores = (self.source_branch(query)[:, None, None] * children).sum(-1)
            scores = scores / math.sqrt(nodes.shape[-1])
            probability = F.softmax(scores.masked_fill(~child_valid, -1e9), dim=-1)
            active = (expand[:, :, None] * probability).reshape(nodes.shape[0], -1)
        if len(route) < len(levels):
            route.extend(query.new_zeros(()) for _ in range(len(levels) - len(route)))
        return context, torch.stack(route)

    def predict(self, state, step, source_levels, source_masks,
                target_intervention="native", source_route="native"):
        query = self.read_target(state, step, target_intervention)
        context, route = self.read_source(query, source_levels, source_masks, source_route)
        logits = self.output(torch.cat((query, context, query - context,
                                        query * context), dim=-1))
        return logits, route

    def write(self, state: TargetTreeState, token: torch.Tensor,
              step: int, active: torch.Tensor) -> TargetTreeState:
        if step >= self.heap_width:
            raise ValueError("target TreeHeap capacity exceeded")
        levels, masks, details = list(state.levels), list(state.masks), list(state.details)
        leaf = levels[0].clone(); leaf_mask = masks[0].clone()
        value = self.encoder.embedding(token)
        leaf[:, step] = torch.where(active[:, None], value, leaf[:, step])
        leaf_mask[:, step] = leaf_mask[:, step] | active
        levels[0], masks[0] = leaf, leaf_mask
        for depth in range(self.depths):
            parent_index = step >> (depth + 1)
            left_index, right_index = parent_index * 2, parent_index * 2 + 1
            left, right = levels[depth][:, left_index], levels[depth][:, right_index]
            valid = masks[depth][:, left_index] | masks[depth][:, right_index]
            detail = (right - self.encoder.predictor(left)) * valid[:, None]
            parent = (left + 0.5 * detail) * valid[:, None]
            parent_level = levels[depth + 1].clone()
            parent_mask = masks[depth + 1].clone()
            detail_level = details[depth].clone()
            parent_level[:, parent_index] = parent
            parent_mask[:, parent_index] = valid
            detail_level[:, parent_index] = detail
            levels[depth + 1], masks[depth + 1] = parent_level, parent_mask
            details[depth] = detail_level
        return TargetTreeState(levels, masks, details)

    def source_states(self, source, length, source_shuffle=False):
        if source_shuffle:
            source, length = source.roll(1, 0), length.roll(1, 0)
        leaf, root, details, levels, masks = self.encoder.states(source, length)
        return leaf, root, details, levels, masks

    def teacher(self, source, length, target, intervention="native"):
        source_shuffle = intervention == "source_shuffle"
        source_route = "force_root" if intervention == "source_root_only" else "native"
        target_intervention = intervention if intervention in {
            "target_zero", "target_root_only",
        } else "native"
        _, _, _, source_levels, source_masks = self.source_states(
            source, length, source_shuffle,
        )
        state = self.zero_target(source.shape[0], source.device,
                                 source_levels[0].dtype)
        logits, routes = [], []
        for step in range(target.shape[1]):
            current, route = self.predict(
                state, step, source_levels, source_masks,
                target_intervention, source_route,
            )
            logits.append(current); routes.append(route)
            active = target[:, step].ne(self.pad)
            state = self.write(state, target[:, step], step, active)
        return torch.stack(logits, dim=1), torch.stack(routes).mean(0)

    @torch.no_grad()
    def greedy(self, source, length, eos: int, max_len: int):
        _, _, _, source_levels, source_masks = self.source_states(source, length)
        state = self.zero_target(source.shape[0], source.device,
                                 source_levels[0].dtype)
        done = torch.zeros(source.shape[0], device=source.device, dtype=torch.bool)
        outputs, routes = [], []
        for step in range(min(max_len, self.heap_width)):
            logits, route = self.predict(state, step, source_levels, source_masks)
            token = logits.argmax(-1)
            outputs.append(token); routes.append(route)
            active = ~done
            state = self.write(state, token, step, active)
            done |= token.eq(eos)
            if bool(done.all()):
                break
        return torch.stack(outputs, dim=1), torch.stack(routes).mean(0)


@torch.no_grad()
def evaluate(model, loader, args, pad, eos, sp, intervention="native", generate=False):
    model.eval(); loss_sum = tokens = 0
    hypotheses, references, examples, runs = [], [], [], []
    route_sum = None; batches = 0
    for source, length, target, _ in loader:
        source, length, target = source.to(args.device), length.to(args.device), target.to(args.device)
        logits, route = model.teacher(source, length, target, intervention)
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                          ignore_index=pad, reduction="sum"))
        tokens += int(target.ne(pad).sum())
        route_sum = route if route_sum is None else route_sum + route
        batches += 1
        if generate:
            predicted, _ = model.greedy(source, length, eos, args.heap_width)
            for src, hyp, ref in zip(source, predicted, target):
                h = base.clean(hyp.cpu().tolist(), eos, pad)
                r = base.clean(ref.cpu().tolist(), eos, pad)
                hypotheses.append(h); references.append(r)
                runs.append(c12.repeated_run(h))
                if len(examples) < 8:
                    examples.append({"source": sp.decode(base.clean(src.cpu().tolist(), eos, pad)),
                                     "hypothesis": sp.decode(h), "reference": sp.decode(r)})
    result = {"nll": loss_sum / max(1, tokens), "tokens": tokens,
              "route_mass": (route_sum / batches).cpu().tolist()}
    if generate:
        result.update({"token_bleu4": base.bleu4(hypotheses, references),
                       "nonempty": sum(bool(row) for row in hypotheses) / len(hypotheses),
                       "max_repeat_run": max(runs),
                       "unique_output_fraction": len({tuple(row) for row in hypotheses}) / len(hypotheses),
                       "examples": examples})
    return result


@torch.no_grad()
def closure(model, loader, args):
    model.eval()
    source, length, _, _ = next(iter(loader))
    source, length = source.to(args.device), length.to(args.device)
    leaf, _, _, levels, _ = model.source_states(source, length)
    error = (leaf - levels[-1]).float()
    return {"mse": float(error.square().mean()), "max_abs": float(error.abs().max())}


def main() -> None:
    args = parse_args()
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold target plus EOS")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    cfg = base.Config(
        data=args.data, spm_model=args.spm_model, evidence_dir=args.evidence_dir,
        model="target_tree_autoregressive", seed=args.seed,
        train_samples=args.train_samples, valid_samples=args.valid_samples,
        test_samples=args.test_samples, max_scan=args.max_scan,
        min_len=args.min_len, max_len=args.max_len, dim=args.dim,
        hidden=args.hidden, batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, device=args.device, num_workers=0,
    )
    rows, pieces = base.load_rows(cfg, sp)
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    splits = [rows[:args.train_samples],
              rows[args.train_samples:args.train_samples + args.valid_samples],
              rows[args.train_samples + args.valid_samples:]]
    loaders = [DataLoader(base.ParallelDataset(rows), batch_size=args.batch_size,
                          shuffle=index == 0, collate_fn=base.collate(pad))
               for index, rows in enumerate(splits)]
    model = TargetTreeDecoder(vocab, args.dim, args.hidden, args.heap_width, pad, bos).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    initial = evaluate(model, loaders[1], args, pad, eos, sp)
    initial_test = evaluate(model, loaders[2], args, pad, eos, sp)
    trace, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = count = 0.0
        for source, length, target, _ in loaders[0]:
            source, length, target = source.to(args.device), length.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.teacher(source, length, target)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=pad)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()); count += 1
        valid = evaluate(model, loaders[1], args, pad, eos, sp)
        row = {"epoch": epoch, "train_nll": total / count,
               "valid": valid, "elapsed_sec": time.time() - started}
        trace.append(row); print(json.dumps(row), flush=True)
    native = evaluate(model, loaders[2], args, pad, eos, sp, generate=True)
    interventions = {name: evaluate(model, loaders[2], args, pad, eos, sp, name)
                     for name in ("source_shuffle", "target_zero",
                                  "target_root_only", "source_root_only")}
    damage = {name: row["nll"] - native["nll"] for name, row in interventions.items()}
    codec = closure(model, loaders[2], args)
    gates = {
        "learned": initial_test["nll"] - native["nll"] >= 0.50,
        "source_causal": damage["source_shuffle"] >= 0.20,
        "target_history_causal": damage["target_zero"] >= 0.20,
        "target_path_causal": damage["target_root_only"] >= 0.02,
        "source_recursive_causal": damage["source_root_only"] >= 0.02,
        "generation_noncollapsed": native["token_bleu4"] > 1.0 and native["nonempty"] > 0.95 and native["max_repeat_run"] < args.heap_width,
        "closure": codec["mse"] < 1e-10,
    }
    summary = {"claim": "S3-TARGET-TREE-AUTOREGRESSIVE-C14",
               "status": "smoke_complete", "host": socket.gethostname(),
               "config": asdict(cfg), "args": vars(args),
               "parameters": sum(p.numel() for p in model.parameters()),
               "initial": initial, "initial_test": initial_test,
               "test": native, "interventions": interventions,
               "damage": damage, "closure": codec, "gates": gates,
               "trace": trace, "elapsed_sec": time.time() - started}
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    atomic.atomic_json(output / "summary.json", summary)
    print(json.dumps({"test": native, "damage": damage, "closure": codec,
                      "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
