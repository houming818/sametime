import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).resolve().parents[1] / "src" / "s3_multiresolution_language_curriculum_a01.py"
SPEC = importlib.util.spec_from_file_location("multires_a01", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scale_depths_are_strict_and_cover_root_leaf():
    for count in (6, 7, 8, 9):
        depths = MODULE.scale_depths(count)
        values = [depths[scale] for scale in MODULE.SCALES]
        assert values[0] == 0
        assert values[-1] == count - 1
        assert all(left < right for left, right in zip(values, values[1:]))


def test_number_and_negation_audit():
    rows = [{
        "target_zh_full": "甲不能购买3个苹果",
        "target_zh_core": "甲不能购买3个苹果",
        "target_zh_short": "甲不能购买苹果",
        "target_zh_reduced": "甲不能购买3个苹果",
    }]
    result = MODULE.invariant_audit(rows)
    assert result["core"]["number_recall"] == 1.0
    assert result["short"]["number_recall"] == 0.0
    assert result["short"]["negation_recall"] == 1.0


def test_stable_split_and_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    import json
    rows = []
    for index in range(100):
        rows.append({
            "source_line": index,
            "source_en": f"source {index}",
            "target_zh_core": "甲",
            "target_zh_short": "甲做事",
            "target_zh_reduced": "甲正在做一件事",
            "target_zh_full": "甲正在认真地做一件重要的事情",
        })
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    first = MODULE.load_rows(path, 20261013, 96)
    second = MODULE.load_rows(path, 20261013, 96)
    assert [(row["source_line"], row["split"]) for row in first] == [
        (row["source_line"], row["split"]) for row in second
    ]
    assert len(first) == 96


def test_finite_token_ce_avoids_float32_reduction_overflow():
    logits = torch.tensor([[[4.0e37, -4.0e37], [4.0e37, -4.0e37]]])
    target = torch.tensor([[1, 1]])
    token_loss = MODULE.finite_token_ce(logits, target, pad=2)
    assert torch.isfinite(token_loss).all()
    assert torch.isfinite(token_loss.double().mean())


def test_prefix_control_matches_reference_lengths_and_ends_with_eos():
    pad, eos = 99, 2
    targets = {
        "core": torch.tensor([[7, 2, 99], [8, 9, 2]]),
        "full": torch.tensor([[10, 11, 12, 2], [20, 21, 22, 2]]),
    }
    result = MODULE.prefix_control_target(targets, "core", pad, eos)
    assert result.tolist() == [[10, 2, 99], [20, 21, 2]]


if __name__ == "__main__":
    import tempfile

    test_scale_depths_are_strict_and_cover_root_leaf()
    test_number_and_negation_audit()
    test_finite_token_ce_avoids_float32_reduction_overflow()
    test_prefix_control_matches_reference_lengths_and_ends_with_eos()
    with tempfile.TemporaryDirectory() as directory:
        test_stable_split_and_limit(Path(directory))
    print("5 direct tests passed")
