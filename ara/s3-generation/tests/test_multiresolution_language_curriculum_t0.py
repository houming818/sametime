import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "s3_multiresolution_language_curriculum_t0.py"
SPEC = importlib.util.spec_from_file_location("curriculum_t0", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CurriculumT0Tests(unittest.TestCase):
    def test_json_object_accepts_fenced_json(self):
        value = MODULE.json_object('```json\n{"core":"甲","short":"甲做事","reduced":"甲完成了任务"}\n```')
        self.assertEqual(value, {"core": "甲", "short": "甲做事", "reduced": "甲完成了任务"})

    def test_json_field_accepts_single_stage(self):
        value = MODULE.json_field('```json\n{"short":"人员完成实验"}\n```', "short")
        self.assertEqual(value, "人员完成实验")

    def test_validate_candidate_accepts_ordered_scales(self):
        full = "研究人员在反复检查数据以后完成了实验，并向团队报告了主要结果。"
        candidate = {"core": "人员实验", "short": "人员完成实验", "reduced": "研究人员检查数据后完成实验"}
        valid, reasons, audit = MODULE.validate_candidate(candidate, full)
        self.assertTrue(valid, reasons)
        self.assertLessEqual(audit["lengths"]["core"], audit["lengths"]["short"])

    def test_validate_candidate_rejects_reversed_lengths(self):
        full = "研究人员在反复检查数据以后完成了实验，并向团队报告了主要结果。"
        candidate = {"core": "研究人员完成实验并报告主要结果", "short": "完成实验", "reduced": "研究人员检查后完成实验"}
        valid, reasons, _ = MODULE.validate_candidate(candidate, full)
        self.assertFalse(valid)
        self.assertIn("non_monotonic_length", reasons)

    def test_suggested_range_is_observation_not_rejection(self):
        full = "研究人员在反复检查数据以后完成了实验，并向团队报告了主要结果。"
        candidate = {"core": "实验", "short": "人员完成实验", "reduced": "研究人员检查后完成了实验"}
        valid, reasons, audit = MODULE.validate_candidate(candidate, full)
        self.assertTrue(valid, reasons)
        self.assertIsInstance(audit["range_observations"], list)

    def test_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "pairs.tsv"
            rows = []
            for index in range(30):
                zh = f"第{index}位研究人员认真检查全部实验数据以后完成了报告并向团队说明主要结果。"
                en = f"Researcher number {index} carefully checked all experimental data and then completed the report for the whole team."
                rows.append(f"{zh}\t{en}")
            data.write_text("\n".join(rows) + "\n", encoding="utf-8")
            first, _ = MODULE.select_material(data, 8, 20261012)
            second, _ = MODULE.select_material(data, 8, 20261012)
            self.assertEqual(
                json.dumps(first, ensure_ascii=False, sort_keys=True),
                json.dumps(second, ensure_ascii=False, sort_keys=True),
            )


if __name__ == "__main__":
    unittest.main()

