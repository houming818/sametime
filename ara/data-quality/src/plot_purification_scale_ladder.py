#!/usr/bin/env python3
"""Aggregate and plot the preregistered purification scale ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(
        args.evidence_dir.glob("*/summary.json"),
        key=lambda path: int(path.parent.name),
    ):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        generation = summary["generation"]
        standard = generation["standard"]
        nll = float(summary["test_nll"])
        rows.append(
            {
                "train_rows": int(summary["rows"]["train"]),
                "steps": int(summary["config"]["steps"]),
                "test_nll": nll,
                "ppl": math.exp(nll),
                "adjacent_repetition_rate": float(
                    generation["adjacent_repetition_rate"]
                ),
                "en2zh_bleu": float(standard["en2zh"]["sacrebleu"]),
                "zh2en_bleu": float(standard["zh2en"]["sacrebleu"]),
                "source_shuffle_delta": float(summary["source_shuffle_delta"]),
                "pair_break_depth_0_delta": float(
                    summary["pair_break_depth_0_delta"]
                ),
                "runtime_identity_delta": float(
                    summary["runtime_identity_delta"]
                ),
                "seconds": float(summary["seconds"]),
            }
        )

    gains = [
        rows[index - 1]["test_nll"] - rows[index]["test_nll"]
        for index in range(1, len(rows))
    ]
    for index, row in enumerate(rows):
        row["marginal_nll_gain"] = None if index == 0 else gains[index - 1]

    comparison = {
        "experiment": "purification_scale_ladder",
        "seed": 14108,
        "plateau_threshold": 0.05,
        "plateau_requires_consecutive_increments": 2,
        "plateau_reached": len(gains) >= 2 and all(gain < 0.05 for gain in gains[-2:]),
        "rows": rows,
    }
    (args.evidence_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )

    image = Image.new("RGB", (1600, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    draw.text((40, 25), "NioClean purification scale ladder (seed 14108)", fill="black", font=font)

    def plot_box(x0: int, title: str, values: list[float], minimum: float, maximum: float) -> None:
        left, top, right, bottom = x0 + 80, 110, x0 + 700, 590
        draw.rectangle((left, top, right, bottom), outline="#555555", width=2)
        draw.text((x0 + 220, 70), title, fill="black", font=small)
        for index, value in enumerate(values):
            x = left + index * (right - left) / max(1, len(values) - 1)
            y = bottom - (value - minimum) / (maximum - minimum) * (bottom - top)
            if index:
                previous = values[index - 1]
                px = left + (index - 1) * (right - left) / max(1, len(values) - 1)
                py = bottom - (previous - minimum) / (maximum - minimum) * (bottom - top)
                draw.line((px, py, x, y), fill="#1769aa", width=4)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#1769aa")
            draw.text((x - 18, bottom + 12), f"{rows[index]['train_rows'] // 1000}K", fill="black", font=small)
            draw.text((x - 28, y - 32), f"{value:.3f}", fill="black", font=small)

    plot_box(0, "Test NLL (lower is better)", [row["test_nll"] for row in rows], 6.0, 7.4)
    plot_box(800, "Marginal NLL gain", [0.0] + gains, 0.0, 0.52)
    threshold_y = 590 - 0.05 / 0.52 * (590 - 110)
    draw.line((880, threshold_y, 1500, threshold_y), fill="crimson", width=2)
    draw.text((1190, threshold_y - 28), "plateau threshold = 0.05", fill="crimson", font=small)
    image.save(args.evidence_dir / "scale_ladder.png")


if __name__ == "__main__":
    main()
