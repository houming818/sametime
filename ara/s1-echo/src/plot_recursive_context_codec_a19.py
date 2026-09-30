#!/usr/bin/env python3
"""Plot the A19 dimension ladder and recursive READ-depth curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_with_pillow(results: dict, audit: dict, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 23)
        small = ImageFont.truetype("arial.ttf", 19)
        title = ImageFont.truetype("arialbd.ttf", 29)
    except OSError:
        font = small = title = ImageFont.load_default()

    colors = {2: "#777777", 4: "#0072B2", 8: "#009E73", 16: "#D55E00"}
    y_min, y_max = 5.0, 7.0

    def panel(box, heading, x_labels, series, baselines=()):
        x0, y0, x1, y1 = box
        left, top, right, bottom = x0 + 95, y0 + 65, x1 - 25, y1 - 75
        draw.text(((x0 + x1) / 2, y0 + 12), heading, fill="#111111",
                  font=font, anchor="ma")
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        for tick in (5.0, 5.5, 6.0, 6.5, 7.0):
            py = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
            draw.line((left, py, right, py), fill="#DDDDDD", width=1)
            draw.text((left - 12, py), f"{tick:.1f}", fill="#333333",
                      font=small, anchor="rm")
        for value, color, label in baselines:
            py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
            for px in range(int(left), int(right), 18):
                draw.line((px, py, min(px + 10, right), py), fill=color, width=3)
            draw.text((right - 5, py - 7), label, fill=color, font=small, anchor="rs")
        count = len(x_labels)
        x_positions = [left + index * (right - left) / max(count - 1, 1)
                       for index in range(count)]
        for px, label in zip(x_positions, x_labels):
            draw.text((px, bottom + 12), str(label), fill="#333333",
                      font=small, anchor="ma")
        for label, values, color in series:
            points = []
            for px, value in zip(x_positions, values):
                py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                points.append((px, py))
            draw.line(points, fill=color, width=4, joint="curve")
            for px, py in points:
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
            legend_x = left + 8
            legend_y = top + 10 + 28 * list(series).index((label, values, color))
            draw.line((legend_x, legend_y, legend_x + 35, legend_y), fill=color, width=4)
            draw.text((legend_x + 45, legend_y), label, fill=color,
                      font=small, anchor="lm")
        draw.text(((left + right) / 2, y1 - 24), "Dimension" if count == 4 else "READ depth",
                  fill="#222222", font=small, anchor="ma")

    dimensions = [2, 4, 8, 16]
    baselines = (
        (audit["global_context_prior_test_nll"], "#CC79A7", "global prior"),
        (audit["direct_row_probability_field_test_nll"], "#56B4E9", "direct row field"),
    )
    panel(
        (20, 70, 890, 700),
        "Capacity ladder: sealed-test NLL",
        dimensions,
        [("recursive codec", [results[d]["final"]["test"]["nll"] for d in dimensions],
          "#222222")],
        baselines,
    )
    depth_series = []
    for d in (4, 8, 16):
        depth_series.append((f"d={d}",
                             [point["test"]["nll"] for point in results[d]["depth_curve"]],
                             colors[d]))
    panel(
        (910, 70, 1780, 700),
        "Resolution revealed by recursive READ",
        list(range(11)),
        depth_series,
        ((audit["global_context_prior_test_nll"], "#CC79A7", "global prior"),),
    )
    draw.text((width / 2, 25),
              "A19 shared recursive FOLD/READ: full-corpus dimension search",
              fill="#111111", font=title, anchor="ma")
    image.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    formal = Path(args.formal)
    audit = json.loads((formal / "posthoc_baselines.json").read_text(encoding="utf-8"))
    dimensions = [2, 4, 8, 16]
    results = {
        d: json.loads((formal / f"dimension_{d}.json").read_text(encoding="utf-8"))
        for d in dimensions
    }

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plot_with_pillow(results, audit, Path(args.out))
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = {2: "#777777", 4: "#0072B2", 8: "#009E73", 16: "#D55E00"}

    ax = axes[0]
    nll = [results[d]["final"]["test"]["nll"] for d in dimensions]
    ax.plot(dimensions, nll, color="#222222", marker="o", linewidth=2.0,
            label="Recursive codec")
    ax.axhline(audit["global_context_prior_test_nll"], color="#CC79A7",
               linestyle="--", linewidth=1.8, label="Global context prior")
    ax.axhline(audit["direct_row_probability_field_test_nll"], color="#56B4E9",
               linestyle=":", linewidth=1.8, label="Direct row field")
    for d, value in zip(dimensions, nll):
        ax.annotate(f"{value:.3f}", (d, value), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(dimensions)
    ax.set_xlabel("TreeHeap state dimension")
    ax.set_ylabel("Sealed-test NLL (lower is better)")
    ax.set_title("Capacity ladder")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for d in dimensions[1:]:
        curve = results[d]["depth_curve"]
        ax.plot([point["read_depth"] for point in curve],
                [point["test"]["nll"] for point in curve],
                color=colors[d], marker="o", markersize=3.5, linewidth=1.8,
                label=f"d={d}")
    ax.axhline(audit["global_context_prior_test_nll"], color="#CC79A7",
               linestyle="--", linewidth=1.5, label="Global prior")
    ax.set_xticks(range(11))
    ax.set_xlabel("Recursive READ depth")
    ax.set_ylabel("Sealed-test NLL")
    ax.set_title("Resolution revealed by READ")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("A19 shared recursive FOLD/READ: full-corpus dimension search",
                 fontsize=12, fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
