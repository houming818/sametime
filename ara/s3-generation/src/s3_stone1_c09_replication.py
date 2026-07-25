#!/usr/bin/env python3
"""Replicate the fixed-root EOS TreeHeap decoder on a frozen platform."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import socket
import statistics
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_private_protocol_data_dose as data_dose
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_fixed_root_noise_repair as c08
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "logic"
    / "stone1_c09_platform_contract.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--contract", default=str(CONTRACT_PATH))
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=500)
    return parser.parse_args()


def contract_args(contract, cli):
    corpus = contract["corpus"]
    splits = contract["splits"]
    tokenizer = contract["tokenizer"]
    model = contract["model"]
    optimization = contract["optimization"]
    return argparse.Namespace(
        data=corpus["path"],
        spm_model=tokenizer["path"],
        baseline_summary=(
            "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
            "s3_stone1_canonical_codec/summary.json"
        ),
        c04_checkpoint=model["source_checkpoint"],
        evidence_dir=cli.evidence_dir,
        train_samples=splits["train_rows"],
        valid_samples=splits["validation_rows"],
        test_samples=splits["test_rows"],
        baseline_max_scan=300_000,
        pool_max_scan=corpus["pool_scan_rows"],
        source_rows=corpus["declared_rows"],
        source_col=corpus["source_column"],
        target_col=corpus["target_column"],
        min_len=splits["min_tokens"],
        max_len=splits["max_tokens"],
        data_seed=corpus["data_seed"],
        pool_seed=corpus["pool_seed"],
        model_seed=optimization["model_seeds"][0],
        noise_seed=74231,
        batch_size=optimization["batch_size"],
        eval_interval=cli.eval_interval,
        lr=optimization["learning_rate"],
        grad_clip=optimization["gradient_clip"],
        num_workers=cli.num_workers,
        device=model.get("device", contract["hardware"]["device"]),
        heap_width=model["heap_width"],
        leaf_cut=model["leaf_cut"],
        dim=model["state_dim"],
        hidden=model["decoder_hidden"],
        fixed_steps=optimization["updates"],
        code_commit=cli.code_commit,
        smoke=False,
        base_train_samples=30_000,
        doses=[30_000, splits["train_rows"]],
    )


def verify_contract(contract, args, manifest, pieces):
    source = Path(args.data)
    tokenizer = Path(args.spm_model)
    checkpoint = Path(args.c04_checkpoint)
    expected = contract
    checks = {
        "source_bytes": source.stat().st_size == expected["corpus"]["bytes"],
        "tokenizer_bytes": (
            tokenizer.stat().st_size == expected["tokenizer"]["bytes"]
        ),
        "tokenizer_sha256": (
            c01.file_digest(tokenizer) == expected["tokenizer"]["sha256"]
        ),
        "tokenizer_pieces": pieces == expected["tokenizer"]["pieces"],
        "checkpoint_bytes": (
            checkpoint.stat().st_size
            == expected["model"]["source_checkpoint_bytes"]
        ),
        "checkpoint_sha256": (
            c01.file_digest(checkpoint)
            == expected["model"]["source_checkpoint_sha256"]
        ),
        "train_rows": args.train_samples == expected["splits"]["train_rows"],
        "train_sha256": (
            manifest["splits"]["train_sha256"][str(args.train_samples)]
            == expected["splits"]["train_sha256"]
        ),
        "validation_sha256": (
            manifest["splits"]["validation_sha256"]
            == expected["splits"]["validation_sha256"]
        ),
        "test_sha256": (
            manifest["splits"]["test_sha256"]
            == expected["splits"]["test_sha256"]
        ),
        "one_epoch_exposure": (
            args.batch_size * args.fixed_steps
            == expected["optimization"]["sample_exposures"]
        ),
        "fixed_tree_shape": (
            args.heap_width == 64
            and args.dim == 320
            and args.hidden == 512
            and args.leaf_cut == 1
        ),
        "fixed_recipe": (
            args.lr == 0.002
            and args.grad_clip == 1.0
            and c06.DEPTH_FLOOR == 0.02
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"C09 platform contract mismatch: {failed}")
    return checks


@torch.no_grad()
def evaluate(
    model, loader, args, pad, bos, eos, pieces, sp,
    intervention="native", generate=False,
):
    model.eval()
    loss_sum = tokens = count = nonempty = repeated = 0
    route_sum = None
    route_batches = 0
    hypotheses, references, examples = [], [], []
    for source, length, target, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        fixed, visible_length = c08.fixed_source(
            source, length, "eos_tail", args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        logits, route = model.teacher(
            fixed, visible_length, target, bos,
            route_mode="depth_floor", intervention=intervention,
        )
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
        route_cpu = route.detach().float().cpu()
        route_sum = route_cpu if route_sum is None else route_sum + route_cpu
        route_batches += 1
        if not generate:
            continue
        predicted, _ = model.greedy(
            fixed, visible_length, bos, eos, target.shape[1],
            route_mode="depth_floor", intervention=intervention,
        )
        for index in range(source.shape[0]):
            hyp = base.clean(predicted[index].cpu().tolist(), eos, pad)
            ref = base.clean(target[index].cpu().tolist(), eos, pad)
            hypotheses.append(hyp)
            references.append(ref)
            nonempty += int(bool(hyp))
            repeated += int(c01.severe_repetition(hyp))
            count += 1
            if len(examples) < 8:
                original = base.clean(
                    source[index, : int(length[index])].cpu().tolist(), eos, pad,
                )
                examples.append({
                    "en": sp.decode(original),
                    "reference_zh": sp.decode(ref),
                    "hypothesis_zh": sp.decode(hyp),
                })
    nll = loss_sum / max(1, tokens)
    result = {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": tokens,
        "route_mass_by_level": (
            route_sum / max(1, route_batches)
        ).tolist(),
    }
    if generate:
        result.update({
            "token_bleu4": base.bleu4(hypotheses, references),
            "nonempty": nonempty / max(1, count),
            "severe_repetition_rate": repeated / max(1, count),
            "examples": examples,
        })
    return result


def train_seed(
    seed, args, rows, valid_loader, test_loader, vocab, pieces,
    pad, bos, eos, sp, output,
):
    args = copy.copy(args)
    args.model_seed = seed
    c01.set_seed(seed)
    model = c06.load_model(args, vocab, pad).to(args.device)
    encoder_before = c05.tensor_digest(model.encoder)
    trainable = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad, seed + args.train_samples,
    )
    trace = []
    finite = True
    branch_nonzero = branch_observations = 0
    window_loss = 0.0
    window_steps = 0
    started = time.time()
    for step in range(1, args.fixed_steps + 1):
        _, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        fixed, visible_length = c08.fixed_source(
            source, length, "eos_tail", args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        model.train()
        logits, _ = model.teacher(
            fixed, visible_length, target, bos, route_mode="depth_floor",
        )
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        gradient = model.decoder.branch.weight.grad
        if gradient is not None:
            branch_observations += 1
            branch_nonzero += int(float(gradient.detach().abs().max()) > 0.0)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1
        if step % args.eval_interval == 0 or step == args.fixed_steps:
            valid = evaluate(
                model, valid_loader, args, pad, bos, eos, pieces, sp,
            )
            row = {
                "seed": seed,
                "step": step,
                "train_nll_window": window_loss / max(1, window_steps),
                "valid_nll": valid["nll"],
                "route_mass_by_level": valid["route_mass_by_level"],
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = 0.0
            window_steps = 0

    final_valid = evaluate(
        model, valid_loader, args, pad, bos, eos, pieces, sp,
    )
    final_test = evaluate(
        model, test_loader, args, pad, bos, eos, pieces, sp, generate=True,
    )
    detail_shuffle = []
    for depth in range(model.encoder.depths):
        damaged = evaluate(
            model, test_loader, args, pad, bos, eos, pieces, sp,
            intervention=f"detail_shuffle_{depth}",
        )
        detail_shuffle.append({
            "depth": depth,
            "nll": damaged["nll"],
            "damage_nll": damaged["nll"] - final_test["nll"],
        })
    encoder_after = c05.tensor_digest(model.encoder)
    checkpoint = output / "checkpoints" / f"decoder_eos_seed{seed}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "seed": seed,
        "condition": "eos_tail",
        "route_mode": "depth_floor",
        "decoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.decoder.state_dict().items()
        },
        "source_checkpoint": args.c04_checkpoint,
    }, checkpoint)
    return {
        "seed": seed,
        "final_valid": final_valid,
        "final_test": final_test,
        "detail_shuffle": detail_shuffle,
        "max_detail_shuffle_damage_nll": max(
            row["damage_nll"] for row in detail_shuffle
        ),
        "branch_grad_nonzero_fraction": (
            branch_nonzero / max(1, branch_observations)
        ),
        "finite_gradients": finite,
        "encoder_unchanged": encoder_before == encoder_after,
        "trace": trace,
        "seconds": time.time() - started,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": c01.file_digest(checkpoint),
        },
    }


def main():
    cli = parse_args()
    contract = json.loads(Path(cli.contract).read_text(encoding="utf-8"))
    args = contract_args(contract, cli)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "platform_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8",
    )
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8",
    )
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    pieces = sp.get_piece_size()
    platform_checks = verify_contract(contract, args, manifest, pieces)
    baseline = c03.load_baseline(args.baseline_summary)
    platform_checks.update(
        c03.verify_frozen_platform(manifest, baseline, False)
    )
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)

    probe = c06.load_model(args, vocab, pad)
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in probe.parameters()),
        "encoder_frozen": sum(
            parameter.numel() for parameter in probe.encoder.parameters()
        ),
        "decoder_trainable": sum(
            parameter.numel() for parameter in probe.decoder.parameters()
        ),
    }
    del probe
    torch.cuda.reset_peak_memory_stats()

    runs = []
    for seed in contract["optimization"]["model_seeds"]:
        run = train_seed(
            seed, args, rows, valid_loader, test_loader, vocab, pieces,
            pad, bos, eos, sp, output,
        )
        runs.append(run)
        (output / "partial_summary.json").write_text(
            json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        torch.cuda.empty_cache()

    nlls = [run["final_test"]["nll"] for run in runs]
    bleus = [run["final_test"]["token_bleu4"] for run in runs]
    aggregate = {
        "mean_test_nll": statistics.fmean(nlls),
        "test_nll_population_std": statistics.pstdev(nlls),
        "mean_token_bleu4": statistics.fmean(bleus),
        "min_nonempty": min(
            run["final_test"]["nonempty"] for run in runs
        ),
        "max_severe_repetition_rate": max(
            run["final_test"]["severe_repetition_rate"] for run in runs
        ),
        "min_detail_shuffle_damage_nll": min(
            run["max_detail_shuffle_damage_nll"] for run in runs
        ),
        "min_branch_grad_nonzero_fraction": min(
            run["branch_grad_nonzero_fraction"] for run in runs
        ),
    }
    gates = {
        "Q1_mean_test_nll_at_most_3_90": (
            aggregate["mean_test_nll"] <= 3.90
        ),
        "Q2_mean_bleu4_at_least_13_5": (
            aggregate["mean_token_bleu4"] >= 13.5
        ),
        "Q3_nll_std_at_most_0_05": (
            aggregate["test_nll_population_std"] <= 0.05
        ),
        "Q4_every_seed_nonempty": aggregate["min_nonempty"] == 1.0,
        "Q5_every_seed_repetition_at_most_0_10": (
            aggregate["max_severe_repetition_rate"] <= 0.10
        ),
        "S1_every_seed_detail_damage_at_least_0_10": (
            aggregate["min_detail_shuffle_damage_nll"] >= 0.10
        ),
        "S2_every_seed_branch_gradient_nonzero": (
            aggregate["min_branch_grad_nonzero_fraction"] > 0.0
        ),
        "S3_every_seed_depth_floor_present": all(
            min(run["final_test"]["route_mass_by_level"]) >= 0.019
            for run in runs
        ),
        "S4_encoder_frozen": all(run["encoder_unchanged"] for run in runs),
        "S5_finite_gradients": all(run["finite_gradients"] for run in runs),
        "E1_peak_vram_at_most_4_gib": (
            torch.cuda.max_memory_allocated() <= 4 * 1024**3
        ),
        "E2_checkpoints_at_most_300_mib": all(
            run["checkpoint"]["bytes"] <= 300 * 1024**2 for run in runs
        ),
        "E3_platform_contract_exact": all(platform_checks.values()),
    }
    q_pass = all(value for key, value in gates.items() if key.startswith("Q"))
    s_pass = all(value for key, value in gates.items() if key.startswith("S"))
    e_pass = all(value for key, value in gates.items() if key.startswith("E"))
    status = (
        "stone1_supported_on_frozen_platform"
        if q_pass and s_pass and e_pass
        else "translation_demo_only"
        if q_pass and e_pass
        else "treeheap_mechanism_only"
        if s_pass and e_pass
        else "not_supported_under_frozen_platform"
    )
    summary = {
        "experiment_id": "s3_stone1_c09_replication",
        "claim": "S3-STONE1-FROZEN-PLATFORM-REPLICATION-C09",
        "predict": "P-S3-STONE1-FROZEN-PLATFORM-REPLICATION-09",
        "status": status,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0),
        "git_commit": cli.code_commit or c01.git_revision(),
        "seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "platform_contract": contract,
        "platform_checks": platform_checks,
        "dataset": manifest,
        "parameter_counts": parameter_counts,
        "runs": runs,
        "aggregate": aggregate,
        "gates": gates,
        "boundary": (
            "C09 validates one fixed 1M-pair EOS-tail platform. It does not "
            "establish full-corpus scaling, a removable pressure floor, "
            "state-of-the-art translation, conversation, or world knowledge."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "aggregate": aggregate,
        "gates": gates,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
