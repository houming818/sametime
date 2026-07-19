#!/usr/bin/env python3
"""Depth-curriculum lifting pretraining followed by matched WMT transfer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as adaptive
import s2_lifting_pump_wmt as pump
import s3_lifting_subheap_pretrain as subheap
import s3_recursive_lifting_depth_checkpoint_audit as depth_audit
import s3_wmt_treeheap_seq2seq as base


CAP_CYCLE = (6, 6, 6, 6, 5, 4, 3, 2, 1, 0)
EVAL_CAPS = (0, 2, 4, 6)


def cap_for_step(step: int, depths: int) -> int:
    return min(depths, CAP_CYCLE[(step - 1) % len(CAP_CYCLE)])


def finite_gradients(model) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def train_logits(model, source, length, target, bos: int, cap: int):
    logits, _ = depth_audit.teacher_with_cap(model, source, length, target, bos, cap)
    return logits


@torch.no_grad()
def evaluate_missing_span(model, loader, args, cap: int, intervention: str = "native") -> dict:
    model.eval()
    loss_sum = token_count = batches = 0
    route_sum = None
    route_rows = 0
    for batch_no, (source, target, _, _) in enumerate(loader, 1):
        source, target = source.to(args.device), target.to(args.device)
        length = torch.full(
            (source.shape[0],), source.shape[1], device=args.device, dtype=torch.long,
        )
        if intervention == "native":
            _, _, _, levels, masks = model.states(source, length)
        else:
            levels, masks = subheap.transformed_levels(model, source, length, intervention)
        decoder = model.decoder
        hidden = levels[0].new_zeros((source.shape[0], decoder.hidden))
        previous = torch.full(
            (source.shape[0],), args.bos, device=args.device, dtype=torch.long,
        )
        logits = []
        for token_index in range(target.shape[1]):
            context, mass = depth_audit.read_with_cap(decoder, hidden, levels, masks, cap)
            hidden = decoder.cell(
                torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
            )
            logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
            previous = target[:, token_index]
            depth_mass = mass.sum(0).double().cpu()
            if route_sum is None:
                route_sum = torch.zeros(model.encoder.depths + 1, dtype=torch.double)
            route_sum[: depth_mass.numel()] += depth_mass
            route_rows += mass.shape[0]
        logits = torch.stack(logits, dim=1)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=args.pad, reduction="sum",
        ))
        token_count += int(target.ne(args.pad).sum())
        batches += 1
        if batch_no >= args.pretrain_valid_batches:
            break
    nll = loss_sum / max(1, token_count)
    return {
        "cap": cap,
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": token_count,
        "route_depth_mass": (route_sum / max(1, route_rows)).tolist(),
    }


def make_pretrain_loader(args, split: str, seed: int, forced_width: int = 0):
    return subheap.make_loader(
        args, split, "subheap", seed, forced_width,
    )


def run_pretrain(args, sp, output: Path) -> dict:
    print(json.dumps({"stage": "pretrain", "status": "start"}), flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = adaptive.make_model(
        "learned_update", args.vocab, args.dim, args.hidden,
        args.heap_width, args.pad,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr)
    train_loader = make_pretrain_loader(args, "train", args.seed)
    train_iter = iter(train_loader)
    valid_loader = make_pretrain_loader(args, "valid", args.seed + 1008, 8)

    initial = {
        str(cap): evaluate_missing_span(model, valid_loader, args, cap)
        for cap in EVAL_CAPS
    }
    trace: List[dict] = []
    best_score = float("inf")
    best_state = None
    finite = True
    started = time.time()
    for step in range(1, args.pretrain_steps + 1):
        model.train()
        source, target, _, _ = next(train_iter)
        source, target = source.to(args.device), target.to(args.device)
        length = torch.full(
            (source.shape[0],), source.shape[1], device=args.device, dtype=torch.long,
        )
        cap = cap_for_step(step, model.encoder.depths)
        logits = train_logits(model, source, length, target, args.bos, cap)
        loss = base.ce(logits, target, args.pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_validate = (
            step == 1 or step % args.pretrain_valid_every == 0
            or step == args.pretrain_steps
        )
        if should_validate:
            valid = {
                str(eval_cap): evaluate_missing_span(
                    model, valid_loader, args, eval_cap,
                )
                for eval_cap in EVAL_CAPS
            }
            score = sum(row["nll"] for row in valid.values()) / len(valid)
            row = {
                "stage": "pretrain",
                "step": step,
                "cap": cap,
                "train_nll": float(loss.detach()),
                "mean_cap_nll": score,
                "valid": valid,
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
            latest = {
                "name": "learned_update",
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "config": vars(args),
                "step": step,
                "trace": trace,
            }
            torch.save(latest, output / "checkpoint_pretrain_latest.pt")
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(latest["state_dict"])
                torch.save({**latest, "state_dict": best_state}, output / "checkpoint_pretrain_best.pt")

    if best_state is None:
        raise RuntimeError("pretraining produced no checkpoint")
    model.load_state_dict(best_state)
    final = {
        str(cap): evaluate_missing_span(model, valid_loader, args, cap)
        for cap in EVAL_CAPS
    }
    full_native = final[str(model.encoder.depths)]["nll"]
    interventions = {}
    for name in ("source_shuffle", "root_zero", "sibling_swap"):
        result = evaluate_missing_span(
            model, valid_loader, args, model.encoder.depths, name,
        )
        interventions[name] = {
            "nll": result["nll"],
            "damage": result["nll"] - full_native,
        }
    closure = subheap.closure(model, valid_loader, args.device)
    initial_mean = sum(row["nll"] for row in initial.values()) / len(initial)
    final_mean = sum(row["nll"] for row in final.values()) / len(final)
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": finite,
        "seconds": time.time() - started,
        "initial": initial,
        "final": final,
        "mean_cap_nll_gain": initial_mean - final_mean,
        "interventions": interventions,
        "closure": closure,
        "trace": trace,
        "checkpoint": "checkpoint_pretrain_best.pt",
    }
    (output / "pretrain_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    return result


def make_wmt_loader(rows, args, shuffle: bool):
    generator = torch.Generator().manual_seed(args.seed + 404) if shuffle else None
    return DataLoader(
        base.ParallelDataset(rows), batch_size=args.wmt_batch, shuffle=shuffle,
        generator=generator, num_workers=args.num_workers,
        collate_fn=base.collate(args.pad), pin_memory=args.device.startswith("cuda"),
    )


def load_wmt_rows(args, sp):
    load_args = SimpleNamespace(
        data=args.wmt_data,
        train_samples=args.wmt_train_samples,
        valid_samples=args.wmt_valid_samples,
        test_samples=args.wmt_test_samples,
        max_scan=args.wmt_max_scan,
        min_len=args.wmt_min_len,
        max_len=args.wmt_max_len,
        source_col=1,
        target_col=0,
        seed=args.seed,
    )
    rows, sampling = adaptive.load_rows(load_args, sp)
    a = args.wmt_train_samples
    b = a + args.wmt_valid_samples
    return rows[:a], rows[a:b], rows[b:], sampling


def update_stream_hash(hasher, source, target, batch_no: int):
    if batch_no <= 1024:
        hasher.update(source.contiguous().numpy().tobytes())
        hasher.update(target.contiguous().numpy().tobytes())


def train_wmt_variant(
    name: str,
    initial_state,
    train_rows,
    valid_rows,
    test_rows,
    args,
    sp,
    output: Path,
) -> dict:
    torch.manual_seed(args.seed + 909)
    torch.cuda.manual_seed_all(args.seed + 909)
    model = adaptive.make_model(
        "learned_update", args.vocab, args.dim, args.hidden,
        args.heap_width, args.pad,
    ).to(args.device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.wmt_lr)
    train_loader = make_wmt_loader(train_rows, args, True)
    valid_loader = make_wmt_loader(valid_rows, args, False)
    test_loader = make_wmt_loader(test_rows, args, False)
    cfg = SimpleNamespace(
        device=args.device,
        max_len=args.wmt_max_len,
    )
    stream_hash = hashlib.sha256()
    stream_batch = 0
    trace = []
    best_nll = float("inf")
    best_state = None
    finite = True
    global_step = 0
    started = time.time()
    for epoch in range(1, args.wmt_epochs + 1):
        model.train()
        loss_sum = steps = 0
        for source, length, target, _ in train_loader:
            global_step += 1
            stream_batch += 1
            update_stream_hash(stream_hash, source, target, stream_batch)
            source, length, target = (
                source.to(args.device, non_blocking=True),
                length.to(args.device, non_blocking=True),
                target.to(args.device, non_blocking=True),
            )
            cap = cap_for_step(global_step, model.encoder.depths)
            logits = train_logits(model, source, length, target, args.bos, cap)
            loss = base.ce(logits, target, args.pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite = finite and finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        valid = pump.evaluate(
            model, valid_loader, cfg, args.pad, args.bos, args.eos, sp,
        )
        row = {
            "stage": "wmt",
            "model": name,
            "epoch": epoch,
            "train_nll": loss_sum / max(1, steps),
            "valid_nll": valid["nll"],
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        current_state = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        torch.save({
            "name": "learned_update",
            "variant": name,
            "state_dict": current_state,
            "config": vars(args),
            "epoch": epoch,
            "trace": trace,
        }, output / f"checkpoint_{name}_latest.pt")
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best_state = copy.deepcopy(current_state)
            torch.save({
                "name": "learned_update",
                "variant": name,
                "state_dict": best_state,
                "config": vars(args),
                "epoch": epoch,
                "trace": trace,
            }, output / f"checkpoint_{name}_best.pt")

    if best_state is None:
        raise RuntimeError(f"{name} produced no WMT checkpoint")
    model.load_state_dict(best_state)
    test = pump.evaluate(
        model, test_loader, cfg, args.pad, args.bos, args.eos, sp, generate=True,
    )
    depth_results = [
        depth_audit.evaluate_cap(
            model, test_loader, args.device, args.pad, args.bos, cap, 0,
        )
        for cap in range(model.encoder.depths + 1)
    ]
    interventions = {
        "source_shuffle": pump.evaluate(
            model, test_loader, cfg, args.pad, args.bos, args.eos, sp,
            intervention="source_shuffle",
        ),
        "root_shuffle": pump.evaluate(
            model, test_loader, cfg, args.pad, args.bos, args.eos, sp,
            intervention="root_shuffle",
        ),
        "detail_shuffle": [
            pump.evaluate(
                model, test_loader, cfg, args.pad, args.bos, args.eos, sp,
                intervention=f"detail_shuffle_{depth}",
            )
            for depth in range(model.encoder.depths)
        ],
        "pair_break": [
            pump.evaluate(
                model, test_loader, cfg, args.pad, args.bos, args.eos, sp,
                pair_break_depth=depth,
            )
            for depth in range(model.encoder.depths)
        ],
    }
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": finite,
        "seconds": time.time() - started,
        "stream_hash_first_1024_batches": stream_hash.hexdigest(),
        "trace": trace,
        "test": test,
        "depth_results": depth_results,
        "interventions": interventions,
        "closure": adaptive.closure_audit(model, test_loader, args.device),
        "checkpoint": f"checkpoint_{name}_best.pt",
    }
    (output / f"wmt_{name}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    return result


def build_gates(pretrain, models) -> tuple[dict, dict]:
    scratch = models["scratch"]
    pretrained = models["pretrained"]
    scratch_nll = scratch["test"]["nll"]
    pretrained_nll = pretrained["test"]["nll"]
    scratch_bleu = scratch["test"]["token_bleu4"]
    pretrained_bleu = pretrained["test"]["token_bleu4"]
    route = pretrained["test"]["route_depth_mass"]
    depth_nll = [row["nll"] for row in pretrained["depth_results"]]
    improvements = [depth_nll[index - 1] - depth_nll[index] for index in range(1, len(depth_nll))]
    detail_damage = [
        row["nll"] - pretrained_nll
        for row in pretrained["interventions"]["detail_shuffle"]
    ]
    pair_damage = [
        row["nll"] - pretrained_nll
        for row in pretrained["interventions"]["pair_break"]
    ]
    derived = {
        "pretrain_mean_cap_nll_gain": pretrain["mean_cap_nll_gain"],
        "pretrain_source_shuffle_damage": pretrain["interventions"]["source_shuffle"]["damage"],
        "pretrain_root_zero_damage": pretrain["interventions"]["root_zero"]["damage"],
        "pretrain_sibling_swap_damage": pretrain["interventions"]["sibling_swap"]["damage"],
        "scratch_nll": scratch_nll,
        "pretrained_nll": pretrained_nll,
        "transfer_nll_gain": scratch_nll - pretrained_nll,
        "scratch_token_bleu4": scratch_bleu,
        "pretrained_token_bleu4": pretrained_bleu,
        "transfer_bleu_gain": pretrained_bleu - scratch_bleu,
        "gain_over_old_treeheap_bleu": pretrained_bleu - 9.909146185051828,
        "gap_to_historical_flat_bleu": 10.571508716197724 - pretrained_bleu,
        "route_depth_mass": route,
        "depth_nll": depth_nll,
        "depth_improvements": improvements,
        "root_shuffle_damage": pretrained["interventions"]["root_shuffle"]["nll"] - pretrained_nll,
        "detail_shuffle_damage": detail_damage,
        "pair_break_damage": pair_damage,
    }
    gates = {
        "P0_finite_closed": (
            pretrain["finite_gradients"]
            and all(row["finite_gradients"] for row in models.values())
            and pretrain["closure"]["state_mse"] < 1e-10
            and all(row["closure"]["state_mse"] < 1e-10 for row in models.values())
        ),
        "P1_pretraining_learns": (
            derived["pretrain_mean_cap_nll_gain"] >= 0.20
            and derived["pretrain_source_shuffle_damage"] >= 0.10
        ),
        "P2_transfer": (
            derived["transfer_nll_gain"] >= 0.03
            and derived["transfer_bleu_gain"] >= 0.20
        ),
        "P3_historical_progress": derived["gain_over_old_treeheap_bleu"] > 0.0,
        "P4_multiresolution_use": (
            sum(value >= 0.05 for value in route[:-1]) >= 2
            and route[-1] <= 0.75
        ),
        "P5_positive_depth_growth": (
            depth_nll[0] - depth_nll[-1] >= 0.10
            and sum(value >= 0.01 for value in improvements) >= 3
        ),
        "P6_structural_causality": (
            derived["root_shuffle_damage"] >= 0.05
            and sum(value >= 0.05 for value in detail_damage) >= 3
            and sum(value >= 0.05 for value in pair_damage) >= 3
        ),
        "P7_fair_stream": (
            scratch["stream_hash_first_1024_batches"]
            == pretrained["stream_hash_first_1024_batches"]
        ),
    }
    return derived, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_lifting_pretrain_transfer_full")
    parser.add_argument("--seed", type=int, default=71919)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=50000)
    parser.add_argument("--pretrain-valid-every", type=int, default=2500)
    parser.add_argument("--pretrain-valid-batches", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--wmt-train-samples", type=int, default=200000)
    parser.add_argument("--wmt-valid-samples", type=int, default=5000)
    parser.add_argument("--wmt-test-samples", type=int, default=5000)
    parser.add_argument("--wmt-max-scan", type=int, default=2000000)
    parser.add_argument("--wmt-min-len", type=int, default=8)
    parser.add_argument("--wmt-max-len", type=int, default=32)
    parser.add_argument("--wmt-batch", type=int, default=64)
    parser.add_argument("--wmt-epochs", type=int, default=5)
    parser.add_argument("--wmt-lr", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.block_length != args.heap_width or args.heap_width != 64:
        raise ValueError("registered run requires block_length == heap_width == 64")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    args.pad = sp.get_piece_size()
    args.vocab = args.pad + 1
    args.bos = sp.bos_id()
    args.eos = sp.eos_id()
    args.root = args.pretrain_root
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.time()
    pretrain = run_pretrain(args, sp, output)
    checkpoint = torch.load(
        output / pretrain["checkpoint"], map_location="cpu", weights_only=False,
    )
    train_rows, valid_rows, test_rows, sampling = load_wmt_rows(args, sp)
    models = {}
    for name, state in (("scratch", None), ("pretrained", checkpoint["state_dict"])):
        models[name] = train_wmt_variant(
            name, state, train_rows, valid_rows, test_rows, args, sp, output,
        )
    derived, gates = build_gates(pretrain, models)
    decision = "supported" if all(gates.values()) else "partial" if any(gates.values()) else "not_supported"
    summary = {
        "claim": "S2-LIFT-PRETRAIN-TRANSFER-C01",
        "decision": decision,
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "sampling": sampling,
        "pretrain": pretrain,
        "models": models,
        "derived": derived,
        "gates": gates,
        "boundary": (
            "Single-seed internal token-BLEU transfer experiment; not standard "
            "SacreBLEU, production MT, or architecture superiority."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "derived": derived, "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
