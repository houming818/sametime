#!/usr/bin/env python3
"""One-pass NioClean comparison of uniform, QK, and path-aware QK routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SRC = Path.cwd() / "ara" / "s3-generation" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_SRC))
import s2_adaptive_lifting_wmt as adaptive  # noqa: E402
import s2_lifting_pump_wmt as prior  # noqa: E402
import s3_tree_qk_path_encoding_smoke as qk  # noqa: E402
import s3_wmt_treeheap_seq2seq as base  # noqa: E402


CLAIM = "S3-TREE-QK-PATH-C02"
ARMS = ("uniform", "qk_no_path", "qk_learned_path")
EXPECTED_ROWS = 7_304_358
EXPECTED_BYTES = 1_535_169_936
EXPECTED_SHA256 = "299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8"
EXPECTED_EVAL_SHA256 = "5271a5fb90633eedff5b8746147c38c768c0f2bd9a0480ca94c3f4ac36c3a3e9"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def optimizer_to(optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def learning_rate(step: int, total_steps: int, args) -> float:
    if step <= args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    progress = min(
        1.0,
        (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps),
    )
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * weight


def set_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def build_evaluation(processor, args):
    selection = SimpleNamespace(
        data=args.eval_data,
        seed=args.seed,
        source_col=1,
        target_col=0,
        train_samples=6000,
        valid_samples=1000,
        test_samples=1000,
        max_scan=120000,
        min_len=16,
        max_len=31,
    )
    rows, sampling = adaptive.load_rows(selection, processor)
    digest = qk.selected_rows_hash(rows)
    if digest != args.expected_eval_sha256:
        raise RuntimeError(
            f"evaluation selection hash {digest} != {args.expected_eval_sha256}"
        )
    excluded = {(tuple(source), tuple(target)) for source, target in rows}
    return rows[6000:7000], rows[7000:8000], excluded, sampling, digest


def batch_encode(processor, texts: list[str], threads: int):
    if not texts:
        return []
    return processor.encode(texts, out_type=int, num_threads=threads)


def iter_full_batches(path: Path, processor, excluded: set, args, start):
    """Yield source/target batches with a byte-resumable sequential cursor."""
    batch = []
    batch_cursor = start["cursor"]
    batch_offset = start["byte_offset"]
    counts = {
        "rows": 0,
        "invalid": 0,
        "excluded": 0,
        "examples": 0,
        "source_truncated": 0,
        "target_truncated": 0,
    }
    with path.open("rb") as handle:
        handle.seek(start["byte_offset"])
        cursor = start["cursor"]
        while cursor < args.expected_rows:
            chunk = []
            for _ in range(args.raw_chunk):
                raw = handle.readline()
                if not raw:
                    break
                cursor += 1
                after = handle.tell()
                parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t", 1)
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    chunk.append((cursor, after, None, None))
                    continue
                chunk.append((cursor, after, parts[0].strip(), parts[1].strip()))
            if not chunk:
                if not raw:
                    break
                continue
            source_rows = batch_encode(
                processor,
                [row[3] for row in chunk if row[2] is not None],
                args.tokenizer_threads,
            )
            target_rows = batch_encode(
                processor,
                [row[2] for row in chunk if row[2] is not None],
                args.tokenizer_threads,
            )
            encoded = iter(zip(source_rows, target_rows))
            for line_cursor, after, zh, _ in chunk:
                counts["rows"] += 1
                if zh is None:
                    counts["invalid"] += 1
                    continue
                source_full, target_full = next(encoded)
                source = source_full[: args.heap_width - 1] + [processor.eos_id()]
                target = target_full[: args.target_cap - 1] + [processor.eos_id()]
                counts["source_truncated"] += int(
                    len(source_full) > args.heap_width - 1
                )
                counts["target_truncated"] += int(
                    len(target_full) > args.target_cap - 1
                )
                if (tuple(source), tuple(target)) in excluded:
                    counts["excluded"] += 1
                    continue
                counts["examples"] += 1
                batch.append((source, target))
                batch_cursor, batch_offset = line_cursor, after
                if len(batch) >= args.batch_size:
                    yield batch_cursor, batch_offset, batch, dict(counts)
                    batch = []
            if cursor >= args.expected_rows:
                break
        if batch:
            yield cursor, handle.tell(), batch, dict(counts)
        elif cursor == args.expected_rows:
            yield cursor, handle.tell(), [], dict(counts)
        if cursor != args.expected_rows:
            raise RuntimeError(f"corpus ended at row {cursor}, expected {args.expected_rows}")


def make_eval_loader(rows, batch_size: int, pad: int):
    return DataLoader(
        base.ParallelDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=base.collate(pad),
    )


def make_models(args, vocab: int, pad: int):
    models = {}
    optimizers = {}
    initial_hashes = {}
    for arm in ARMS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = qk.TreeQKModel(
            arm, vocab, args.dim, args.hidden, args.heap_width, pad
        ).to(args.device)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.0
        )
        initial_hashes[arm] = qk.state_hash(
            (name, value)
            for name, value in model.state_dict().items()
            if "learned_paths" not in name and "fixed_paths" not in name
        )
    return models, optimizers, initial_hashes


def save_checkpoint(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(models, optimizers, progress, contract):
    return {
        "claim": CLAIM,
        "contract": contract,
        "progress": progress,
        "models": {arm: model.state_dict() for arm, model in models.items()},
        "optimizers": {
            arm: optimizer.state_dict() for arm, optimizer in optimizers.items()
        },
    }


@torch.no_grad()
def validation_nll(model, loader, args, pad: int, bos: int):
    model.eval()
    loss_sum = token_sum = 0.0
    for source, length, target, _ in loader:
        source = source.to(args.device)
        length = length.to(args.device)
        target = target.to(args.device)
        logits, _ = model.teacher(source, length, target, bos)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="sum",
        )
        tokens = int(target.ne(pad).sum())
        loss_sum += float(loss)
        token_sum += tokens
    return loss_sum / max(1, token_sum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv",
    )
    parser.add_argument(
        "--eval-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv"
    )
    parser.add_argument(
        "--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model"
    )
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_BYTES)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--expected-eval-sha256", default=EXPECTED_EVAL_SHA256)
    parser.add_argument("--heap-width", type=int, default=32)
    parser.add_argument("--target-cap", type=int, default=64)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--raw-chunk", type=int, default=2048)
    parser.add_argument("--tokenizer-threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--wake-every", type=int, default=5000)
    parser.add_argument("--audit-batches", type=int, default=8)
    parser.add_argument("--generation-samples", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.heap_width < 2 or args.heap_width & (args.heap_width - 1):
        raise ValueError("heap width must be a power of two")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = args.evidence_dir
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    if data_path.stat().st_size != args.expected_bytes:
        raise RuntimeError(
            f"data bytes {data_path.stat().st_size} != {args.expected_bytes}"
        )
    data_sha256 = qk.sha256_file(data_path)
    if data_sha256 != args.expected_sha256:
        raise RuntimeError(f"data hash {data_sha256} != {args.expected_sha256}")

    processor = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces = processor.get_piece_size()
    pad, bos, eos, vocab = pieces, processor.bos_id(), processor.eos_id(), pieces + 1
    valid_rows, test_rows, excluded, selection, eval_sha256 = build_evaluation(
        processor, args
    )
    valid_loader = make_eval_loader(valid_rows, args.eval_batch_size, pad)
    test_loader = make_eval_loader(test_rows, args.eval_batch_size, pad)
    models, optimizers, initial_hashes = make_models(args, vocab, pad)
    # The exact number of excluded matches is only known after the stream. Use
    # the full row count as the deterministic scheduler horizon.
    total_steps = math.ceil(args.expected_rows / args.batch_size)
    first_test_batch = next(iter(test_loader))
    step_zero = {
        arm: qk.initial_batch_nll(model, first_test_batch, args, pad, bos)
        for arm, model in models.items()
    }
    contract = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "data": {
            "path": str(data_path),
            "rows": args.expected_rows,
            "bytes": args.expected_bytes,
            "sha256": data_sha256,
        },
        "evaluation": {
            "selected_rows_sha256": eval_sha256,
            "selection": selection,
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "excluded_candidates": len(excluded),
        },
        "tokenizer_sha256": qk.sha256_file(Path(args.spm_model)),
        "arms": list(ARMS),
        "shared_initial_state_equal": len(set(initial_hashes.values())) == 1,
        "shared_initial_sha256": next(iter(initial_hashes.values())),
        "step_zero_nll": step_zero,
        "no_path_vs_learned_step_zero_delta": abs(
            step_zero["qk_no_path"] - step_zero["qk_learned_path"]
        ),
        "config": vars(args) | {"evidence_dir": str(args.evidence_dir)},
        "estimated_steps": total_steps,
    }
    write_json(output / "contract.json", contract)

    checkpoint = output / "checkpoint_latest.pt"
    start = {
        "step": 0,
        "cursor": 0,
        "byte_offset": 0,
        "processed_examples": 0,
        "processed_tokens": 0,
        "counts": {},
    }
    if args.resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["contract"]["data"]["sha256"] != data_sha256:
            raise RuntimeError("resume data hash mismatch")
        if payload["contract"]["evaluation"]["selected_rows_sha256"] != eval_sha256:
            raise RuntimeError("resume evaluation hash mismatch")
        for arm in ARMS:
            models[arm].load_state_dict(payload["models"][arm])
            optimizers[arm].load_state_dict(payload["optimizers"][arm])
            optimizer_to(optimizers[arm], args.device)
        start = payload["progress"]

    step = int(start["step"])
    processed_examples = int(start["processed_examples"])
    processed_tokens = int(start["processed_tokens"])
    prior_counts = dict(start.get("counts", {}))
    window_loss = {arm: 0.0 for arm in ARMS}
    window_steps = 0
    finite = {arm: True for arm in ARMS}
    started = time.time()
    collate = base.collate(pad)
    stream = iter_full_batches(
        data_path,
        processor,
        excluded,
        args,
        {"cursor": int(start["cursor"]), "byte_offset": int(start["byte_offset"])},
    )
    final_cursor = int(start["cursor"])
    final_offset = int(start["byte_offset"])
    stream_counts = {}
    for cursor, byte_offset, rows, stream_counts in stream:
        if not rows:
            final_cursor, final_offset = cursor, byte_offset
            continue
        step += 1
        current_lr = learning_rate(step, total_steps, args)
        source, length, target, _ = collate(rows)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        tokens = int(target.ne(pad).sum())
        for arm in ARMS:
            model, optimizer = models[arm], optimizers[arm]
            model.train()
            set_lr(optimizer, current_lr)
            logits, _ = model.teacher(source, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite[arm] = finite[arm] and prior.finite_gradients(model)
            if not finite[arm] or not math.isfinite(float(loss.detach())):
                raise RuntimeError(f"non-finite state in {arm} at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            window_loss[arm] += float(loss.detach())
            del logits, loss
        processed_examples += len(rows)
        processed_tokens += tokens
        window_steps += 1
        final_cursor, final_offset = cursor, byte_offset
        at_end = cursor >= args.expected_rows
        if step % args.wake_every == 0 or at_end:
            validation = {
                arm: validation_nll(model, valid_loader, args, pad, bos)
                for arm, model in models.items()
            }
            total_counts = {
                key: int(prior_counts.get(key, 0)) + int(stream_counts.get(key, 0))
                for key in set(prior_counts) | set(stream_counts)
            }
            progress = {
                "step": step,
                "cursor": final_cursor,
                "byte_offset": final_offset,
                "processed_examples": processed_examples,
                "processed_tokens": processed_tokens,
                "counts": total_counts,
            }
            wake = {
                "event": "wake",
                **progress,
                "lr": current_lr,
                "train_window_nll": {
                    arm: window_loss[arm] / max(1, window_steps) for arm in ARMS
                },
                "valid_nll": validation,
                "elapsed_sec": time.time() - started,
            }
            append_jsonl(output / "wakes.jsonl", wake)
            write_json(output / "wake_latest.json", wake)
            save_checkpoint(
                checkpoint,
                checkpoint_payload(models, optimizers, progress, contract),
            )
            print(json.dumps(wake, ensure_ascii=False), flush=True)
            window_loss = {arm: 0.0 for arm in ARMS}
            window_steps = 0

    if final_cursor != args.expected_rows:
        raise RuntimeError(
            f"full pass cursor {final_cursor} != expected {args.expected_rows}"
        )

    cfg = SimpleNamespace(device=args.device)
    results = {}
    examples = {}
    for arm, model in models.items():
        test = prior.evaluate(model, test_loader, cfg, pad, bos, eos, processor)
        generation_loader = make_eval_loader(
            test_rows[: args.generation_samples], args.eval_batch_size, pad
        )
        generation = prior.evaluate(
            model,
            generation_loader,
            cfg,
            pad,
            bos,
            eos,
            processor,
            generate=True,
        )
        interventions = {}
        for name in qk.INTERVENTIONS:
            changed = prior.evaluate(
                model,
                test_loader,
                cfg,
                pad,
                bos,
                eos,
                processor,
                intervention=name,
            )
            interventions[name] = {
                "nll": changed["nll"],
                "damage": changed["nll"] - test["nll"],
            }
        results[arm] = {
            "parameters": qk.parameter_count(model),
            "finite_gradients": finite[arm],
            "final_model_sha256": qk.state_hash(model.state_dict().items()),
            "test": test,
            "generation": {key: value for key, value in generation.items() if key != "examples"},
            "repetition": qk.repetition_audit(
                model, test_rows, args, pad, bos, eos
            ),
            "interventions": interventions,
            "route_audit": qk.route_audit(model, test_loader, args, bos, pad),
        }
        examples[arm] = generation.get("examples", [])

    total_counts = {
        key: int(prior_counts.get(key, 0)) + int(stream_counts.get(key, 0))
        for key in set(prior_counts) | set(stream_counts)
    }
    summary = {
        "claim": CLAIM,
        "claim_scope": "one-seed complete NioClean corpus pass",
        "host": socket.gethostname(),
        "seconds_since_start_or_resume": time.time() - started,
        "contract": contract,
        "progress": {
            "step": step,
            "cursor": final_cursor,
            "byte_offset": final_offset,
            "processed_examples": processed_examples,
            "processed_tokens": processed_tokens,
            "counts": total_counts,
        },
        "results": results,
        "boundary": (
            "One full pass and one seed can compare these arms under this fixed "
            "contract; it cannot establish general architecture or energy superiority."
        ),
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    examples_path = output / "examples.json"
    write_json(examples_path, examples)
    readme_path = output / "README.md"
    readme_path.write_text(
        "# Tree QK path full NioClean corpus pass\n\n"
        "Three arms consume the same single sequential pass of all 7,304,358 "
        "released NioClean pairs. Large resumable checkpoints remain on io and "
        "are excluded from the portable evidence package.\n",
        encoding="utf-8",
    )
    portable = [
        output / "contract.json",
        output / "wakes.jsonl",
        output / "wake_latest.json",
        summary_path,
        examples_path,
        readme_path,
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{qk.sha256_file(path)}  {path.name}\n" for path in portable),
        encoding="utf-8",
    )
    save_checkpoint(
        output / "checkpoint_final.pt",
        checkpoint_payload(
            models,
            optimizers,
            summary["progress"],
            contract,
        ),
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "cursor": final_cursor,
                "results": {
                    arm: {
                        "test_nll": row["test"]["nll"],
                        "bleu4": row["generation"]["token_bleu4"],
                        "effective_leaves": row["route_audit"]["native_effective_leaves"],
                        "path_shuffle_damage": row["interventions"]["path_shuffle"]["damage"],
                    }
                    for arm, row in results.items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
