#!/usr/bin/env python3
"""Compare deterministic and uncertain teacher targets for TreeHeap."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import shutil
import socket
import statistics
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_private_protocol_data_dose as data_dose
import s3_stone1_c09_replication as c09
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_fixed_root_noise_repair as c08
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


ARMS = ("gold", "top1", "topk", "shuffled")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--teacher", default="Helsinki-NLP/opus-mt-en-zh")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--train-samples", type=int, default=300_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--teacher-batch", type=int, default=64)
    parser.add_argument("--student-batch", type=int, default=32)
    parser.add_argument("--teacher-beams", type=int, default=4)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=2e-4)
    parser.add_argument("--model-seed", type=int, default=71912)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache", default="")
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--select-from-summary", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_args(cli, output):
    contract = json.loads(c09.CONTRACT_PATH.read_text(encoding="utf-8"))
    args = c09.contract_args(contract, argparse.Namespace(
        evidence_dir=str(output),
        eval_interval=cli.eval_interval,
        num_workers=cli.num_workers,
        code_commit=cli.code_commit,
    ))
    args.train_samples = cli.train_samples
    args.valid_samples = cli.valid_samples
    args.test_samples = cli.test_samples
    args.batch_size = cli.student_batch
    args.model_seed = cli.model_seed
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    if cli.smoke:
        args.train_samples = 2_048
        args.valid_samples = 128
        args.test_samples = 128
        args.base_train_samples = 2_048
        args.doses = [2_048]
        args.baseline_max_scan = 100_000
        args.pool_max_scan = 200_000
        args.batch_size = 16
    return args, contract


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_ids(ids, eos, pad):
    return base.clean(list(ids), eos, pad)


def generate_records(
    rows, sp, cli, cache_path, eos, pad, device, label,
):
    tokenizer = AutoTokenizer.from_pretrained(
        cli.teacher, revision=cli.teacher_revision,
    )
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        cli.teacher,
        revision=cli.teacher_revision,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        use_safetensors=False,
    ).to(device)
    teacher.eval()
    records = []
    teacher_hypotheses = []
    references = []
    started = time.time()
    for start in range(0, len(rows), cli.teacher_batch):
        batch = rows[start : start + cli.teacher_batch]
        texts = [
            ">>cmn_Hans<< " + sp.decode(clean_ids(source, eos, pad))
            for source, _ in batch
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            generated = teacher.generate(
                **encoded,
                num_beams=cli.teacher_beams,
                num_return_sequences=cli.teacher_beams,
                max_new_tokens=64,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True,
        )
        scores = generated.sequences_scores.reshape(
            len(batch), cli.teacher_beams,
        ).float()
        weights = F.softmax(
            scores / cli.teacher_temperature, dim=-1,
        ).cpu().tolist()
        for index, (source, gold) in enumerate(batch):
            candidates = decoded[
                index * cli.teacher_beams : (index + 1) * cli.teacher_beams
            ]
            candidate_ids = [
                sp.encode(text, out_type=int)[:63] + [eos]
                for text in candidates
            ]
            records.append({
                "source": source,
                "gold": gold,
                "candidates": candidate_ids,
                "candidate_text": candidates,
                "sequence_scores": scores[index].cpu().tolist(),
                "weights": weights[index],
            })
            teacher_hypotheses.append(candidate_ids[0])
            references.append(gold)
        if (start // cli.teacher_batch + 1) % 100 == 0:
            print(json.dumps({
                "event": "teacher_generation",
                "split": label,
                "rows": min(start + cli.teacher_batch, len(rows)),
                "total": len(rows),
                "elapsed_sec": time.time() - started,
            }), flush=True)
    del teacher
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records, {
        **summarize_records(records),
        "top1_token_bleu4": base.bleu4(teacher_hypotheses, references),
    }


def load_records(path):
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def summarize_records(records):
    entropies = []
    top_gaps = []
    unique_counts = []
    for record in records:
        weights = record["weights"]
        entropies.append(-sum(
            weight * math.log(max(weight, 1e-12)) for weight in weights
        ))
        top_gaps.append(weights[0] - weights[1] if len(weights) > 1 else 1.0)
        unique_counts.append(len({
            tuple(candidate) for candidate in record["candidates"]
        }))
    return {
        "rows": len(records),
        "mean_top1_weight": statistics.fmean(
            record["weights"][0] for record in records
        ),
        "mean_probability_entropy_nats": statistics.fmean(entropies),
        "mean_top1_top2_gap": statistics.fmean(top_gaps),
        "mean_unique_candidates": statistics.fmean(unique_counts),
        "all_candidates_identical_fraction": statistics.fmean(
            count == 1 for count in unique_counts
        ),
    }


def pad_targets(targets, pad, device):
    width = max(len(target) for target in targets)
    tensor = torch.full(
        (len(targets), width), pad, dtype=torch.long, device=device,
    )
    for index, target in enumerate(targets):
        tensor[index, : len(target)] = torch.tensor(
            target, dtype=torch.long, device=device,
        )
    return tensor


@torch.no_grad()
def product_evaluate(
    model, loader, args, sp, pieces, pad, bos, eos, arm, output,
):
    hypotheses = []
    references = []
    rows = []
    repeated_ngram_sentences = {2: 0, 3: 0, 4: 0}
    for source, length, target, _ in loader:
        source = source.to(args.device)
        length = length.to(args.device)
        fixed, visible_length = c08.fixed_source(
            source, length, "eos_tail", args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        prediction, _ = model.greedy(
            fixed, visible_length, bos, eos, 64, route_mode="depth_floor",
        )
        prediction = prediction.cpu()
        source_cpu = source.cpu()
        target_cpu = target.cpu()
        for index in range(source.shape[0]):
            source_ids = clean_ids(
                source_cpu[index, : int(length[index])].tolist(), eos, pad,
            )
            reference_ids = clean_ids(target_cpu[index].tolist(), eos, pad)
            hypothesis_ids = clean_ids(prediction[index].tolist(), eos, pad)
            source_text = sp.decode(source_ids)
            reference_text = sp.decode(reference_ids)
            hypothesis_text = sp.decode(hypothesis_ids)
            references.append(reference_text)
            hypotheses.append(hypothesis_text)
            rows.append({
                "source": source_text,
                "reference": reference_text,
                "hypothesis": hypothesis_text,
            })
            for n in repeated_ngram_sentences:
                grams = [
                    tuple(hypothesis_ids[start : start + n])
                    for start in range(max(0, len(hypothesis_ids) - n + 1))
                ]
                repeated_ngram_sentences[n] += int(len(grams) != len(set(grams)))

    prediction_path = output / f"predictions_{arm}.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    result = {
        "rows": len(rows),
        "prediction_path": str(prediction_path),
        "prediction_sha256": digest_file(prediction_path),
        "mean_length_ratio": statistics.fmean(
            len(hypothesis) / max(1, len(reference))
            for hypothesis, reference in zip(hypotheses, references)
        ),
        "repeated_ngram_sentence_rate": {
            str(n): count / max(1, len(rows))
            for n, count in repeated_ngram_sentences.items()
        },
    }
    try:
        import sacrebleu
        result["sacrebleu"] = sacrebleu.corpus_bleu(
            hypotheses, [references], tokenize="zh",
        ).score
        result["chrf2"] = sacrebleu.corpus_chrf(
            hypotheses, [references], word_order=2,
        ).score
        result["ter"] = sacrebleu.corpus_ter(
            hypotheses, [references],
        ).score
    except ImportError:
        result["standard_metrics_error"] = "sacrebleu is not installed"
    return result


def arm_targets(records, arm):
    targets = []
    weights = []
    for record in records:
        gold = record["gold"]
        candidates = record["candidates"]
        q = record["weights"]
        if arm == "gold":
            row_targets, row_weights = [gold], [1.0]
        elif arm == "top1":
            row_targets, row_weights = [gold, candidates[0]], [0.5, 0.5]
        elif arm == "topk":
            row_targets = [gold, *candidates]
            row_weights = [0.5, *[0.5 * value for value in q]]
        elif arm == "shuffled":
            row_targets = [gold, *candidates]
            row_weights = [0.5, *[0.5 * value for value in reversed(q)]]
        else:
            raise ValueError(arm)
        targets.extend(row_targets)
        weights.extend(row_weights)
    return targets, weights, len(row_targets)


def load_student(args, decoder_checkpoint, vocab, pad):
    model = c06.load_model(args, vocab, pad)
    payload = torch.load(
        decoder_checkpoint, map_location="cpu", weights_only=False,
    )
    model.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def train_arm(
    arm, records, valid_loader, test_loader, args, cli, sp,
    pieces, pad, bos, eos, vocab, decoder_checkpoint, output,
):
    c01.set_seed(cli.model_seed)
    model = load_student(args, decoder_checkpoint, vocab, pad).to(args.device)
    encoder_before = c05.tensor_digest(model.encoder)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cli.encoder_lr},
        {"params": model.decoder.parameters(), "lr": cli.decoder_lr},
    ])
    order = list(range(len(records)))
    random.Random(cli.model_seed).shuffle(order)
    trace = []
    finite = True
    encoder_grad_nonzero = encoder_grad_observations = 0
    started = time.time()
    model.train()
    for step, offset in enumerate(
        range(0, len(order), args.batch_size), start=1,
    ):
        indices = order[offset : offset + args.batch_size]
        batch_records = [records[index] for index in indices]
        source_rows = [
            (record["source"], record["gold"]) for record in batch_records
        ]
        source, length, _, _ = base.collate(pad)(source_rows)
        targets, weights, variants = arm_targets(batch_records, arm)
        source = source.to(args.device).repeat_interleave(variants, dim=0)
        length = length.to(args.device).repeat_interleave(variants, dim=0)
        target = pad_targets(targets, pad, args.device)
        weight = torch.tensor(
            weights, dtype=torch.float32, device=args.device,
        )
        fixed, visible_length = c08.fixed_source(
            source, length, "eos_tail", args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        logits, _ = model.teacher(
            fixed, visible_length, target, bos, route_mode="depth_floor",
        )
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad,
            reduction="none",
        ).reshape(target.shape)
        sequence_loss = (
            token_loss.sum(1) / target.ne(pad).sum(1).clamp_min(1)
        )
        loss = (sequence_loss * weight).sum() / len(batch_records)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and bool(torch.isfinite(loss))
        maximum = 0.0
        for parameter in model.encoder.parameters():
            if parameter.grad is not None:
                maximum = max(maximum, float(parameter.grad.detach().abs().max()))
        encoder_grad_observations += 1
        encoder_grad_nonzero += int(maximum > 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % cli.eval_interval == 0 or offset + args.batch_size >= len(order):
            valid = c09.evaluate(
                model, valid_loader, args, pad, bos, eos, pieces, sp,
            )
            row = {
                "arm": arm,
                "step": step,
                "train_loss": float(loss.detach()),
                "valid_nll": valid["nll"],
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)

    final_test = c09.evaluate(
        model, test_loader, args, pad, bos, eos, pieces, sp, generate=True,
    )
    detail = []
    for depth in range(model.encoder.depths):
        damaged = c09.evaluate(
            model, test_loader, args, pad, bos, eos, pieces, sp,
            intervention=f"detail_shuffle_{depth}",
        )
        detail.append({
            "depth": depth,
            "damage_nll": damaged["nll"] - final_test["nll"],
        })
    result = {
        "arm": arm,
        "final_test": final_test,
        "trace": trace,
        "encoder_grad_nonzero_fraction": (
            encoder_grad_nonzero / max(1, encoder_grad_observations)
        ),
        "encoder_changed": encoder_before != c05.tensor_digest(model.encoder),
        "finite": finite,
        "detail_shuffle": detail,
        "max_detail_shuffle_damage_nll": max(
            item["damage_nll"] for item in detail
        ),
        "seconds": time.time() - started,
    }
    if cli.save_checkpoints:
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"treeheap_{arm}.pt"
        torch.save({
            "experiment_id": "s3_stone2_teacher_uncertainty",
            "arm": arm,
            "model_state_dict": model.state_dict(),
            "student_args": vars(args),
            "tokens": {"pad": pad, "bos": bos, "eos": eos, "vocab": vocab},
            "teacher": cli.teacher,
            "teacher_temperature": cli.teacher_temperature,
        }, checkpoint)
        result["checkpoint"] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": digest_file(checkpoint),
        }
        result["product_evaluation"] = product_evaluate(
            model, test_loader, args, sp, pieces, pad, bos, eos, arm, output,
        )
    del model
    torch.cuda.empty_cache()
    return result


def main():
    cli = parse_args()
    output = Path(cli.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    args, contract = make_args(cli, output)
    temperature_tag = str(cli.teacher_temperature).replace(".", "p")
    cache = Path(cli.cache) if cli.cache else (
        output / (
            f"opus_top{cli.teacher_beams}_{args.train_samples}"
            f"_temp{temperature_tag}.jsonl.gz"
        )
    )
    if cli.select_from_summary:
        prior = json.loads(
            Path(cli.select_from_summary).read_text(encoding="utf-8")
        )
        legal = ("gold", "top1", "topk")
        selected_arms = [min(
            legal,
            key=lambda arm: (
                prior["results"][arm]["final_test"]["nll"],
                -prior["results"][arm]["final_test"]["token_bleu4"],
            ),
        )]
    else:
        selected_arms = [
            arm.strip() for arm in cli.arms.split(",") if arm.strip()
        ]
    if not selected_arms or any(arm not in ARMS for arm in selected_arms):
        raise ValueError(f"invalid arms: {selected_arms}")
    config = {
        **vars(cli),
        "selected_arms": selected_arms,
        "resolved_student": vars(args),
        "cache": str(cache),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )
    if cli.save_checkpoints:
        shutil.copy2(args.spm_model, output / "treeheap_sp.model")

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    if cache.exists():
        records = load_records(cache)
        teacher_train = {
            **summarize_records(records),
            "loaded_cache": True,
        }
    else:
        records, teacher_train = generate_records(
            rows, sp, cli, cache, eos, pad, args.device, "train",
        )
    if len(records) != len(rows):
        raise RuntimeError("teacher cache and frozen training split differ")
    _, teacher_test = generate_records(
        test, sp, cli, None, eos, pad, args.device, "test",
    )
    teacher_manifest = {
        "model": cli.teacher,
        "revision": cli.teacher_revision,
        "beams": cli.teacher_beams,
        "temperature": cli.teacher_temperature,
        "cache": str(cache),
        "cache_bytes": cache.stat().st_size,
        "cache_sha256": digest_file(cache),
        "train": teacher_train,
        "test": teacher_test,
    }
    (output / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2), encoding="utf-8",
    )

    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)
    decoder_checkpoint = (
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_c09_replication/checkpoints/decoder_eos_seed71902.pt"
    )
    results = {}
    for arm in selected_arms:
        results[arm] = train_arm(
            arm, records, valid_loader, test_loader, args, cli, sp,
            pieces, pad, bos, eos, vocab, decoder_checkpoint, output,
        )
        (output / "partial_summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    if tuple(selected_arms) != ARMS:
        summary = {
            "experiment_id": "s3_stone2_teacher_uncertainty_materialized",
            "claim": "S3-STONE2-PRODUCT-C01",
            "status": "selected_arm_materialized",
            "host": socket.gethostname(),
            "device": torch.cuda.get_device_name(0),
            "selection_source": cli.select_from_summary or None,
            "selected_arms": selected_arms,
            "dataset": manifest,
            "teacher": teacher_manifest,
            "results": results,
            "boundary": (
                "This materializes a checkpoint selected by frozen D01 test "
                "metrics; it is a product artifact, not independent evidence."
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({
            "status": summary["status"],
            "selected_arms": selected_arms,
            "metrics": {
                arm: {
                    "nll": result["final_test"]["nll"],
                    "bleu4": result["final_test"]["token_bleu4"],
                }
                for arm, result in results.items()
            },
        }, ensure_ascii=False), flush=True)
        return

    gold = results["gold"]["final_test"]
    top1 = results["top1"]["final_test"]
    topk = results["topk"]["final_test"]
    shuffled = results["shuffled"]["final_test"]
    gates = {
        "U1_topk_nll_beats_top1_by_0_02": (
            top1["nll"] - topk["nll"] >= 0.02
        ),
        "U2_topk_bleu_beats_top1_by_0_20": (
            topk["token_bleu4"] - top1["token_bleu4"] >= 0.20
        ),
        "U3_topk_nll_beats_shuffled_by_0_02": (
            shuffled["nll"] - topk["nll"] >= 0.02
        ),
        "D1_top1_improves_gold": (
            gold["nll"] - top1["nll"] >= 0.02
            or top1["token_bleu4"] - gold["token_bleu4"] >= 0.20
        ),
        "D2_generation_valid": all(
            result["final_test"]["nonempty"] == 1.0
            and result["final_test"]["severe_repetition_rate"] <= 0.10
            for result in results.values()
        ),
        "D3_teacher_distribution_non_degenerate": (
            teacher_test["mean_top1_weight"] >= 0.30
            and teacher_test["mean_unique_candidates"] >= 2.0
            and teacher_test["all_candidates_identical_fraction"] <= 0.05
        ),
        "S1_encoder_gradient": all(
            result["encoder_grad_nonzero_fraction"] > 0.0
            and result["finite"] for result in results.values()
        ),
        "S2_encoder_changed": all(
            result["encoder_changed"] for result in results.values()
        ),
        "S3_topk_detail_damage": (
            results["topk"]["max_detail_shuffle_damage_nll"] >= 0.10
        ),
        "S4_topk_depth_floor": (
            min(topk["route_mass_by_level"]) >= 0.019
        ),
    }
    uncertainty = all(gates[key] for key in (
        "U1_topk_nll_beats_top1_by_0_02",
        "U2_topk_bleu_beats_top1_by_0_20",
        "U3_topk_nll_beats_shuffled_by_0_02",
        "D3_teacher_distribution_non_degenerate",
    ))
    status = (
        "teacher_uncertainty_supported_pilot"
        if uncertainty
        else "deterministic_teacher_only_supported"
        if gates["D1_top1_improves_gold"]
        else "teacher_distillation_not_supported_under_recipe"
    )
    summary = {
        "experiment_id": "s3_stone2_teacher_uncertainty",
        "claim": "S3-STONE2-TEACHER-UNCERTAINTY-D01",
        "status": "smoke_" + status if cli.smoke else status,
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(0),
        "dataset": manifest,
        "teacher": teacher_manifest,
        "results": results,
        "gates": gates,
        "boundary": (
            "Teacher probabilities are model beliefs, not real-world truth. "
            "This measures behavioral transfer under a gold anchor."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "teacher_test": teacher_test,
        "metrics": {
            arm: {
                "nll": result["final_test"]["nll"],
                "bleu4": result["final_test"]["token_bleu4"],
            }
            for arm, result in results.items()
        },
        "gates": gates,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
