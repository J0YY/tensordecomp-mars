from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

from scripts.search_stroke_mask_residual import Reg, example_style_plot, fit_variant
from scripts.validate_best_combo import train_base


def make_specs(rank: int):
    base = dict(rank=rank, topk=1, lr=0.04, residual_scale=0.35)
    return [
        dict(
            **base,
            name="cp_res16_pen020",
            kind="cp",
            residual_rank=16,
            seed=11,
            seed_offset=0,
            reg=Reg(distill=0.20, residual_branch=0.20),
        ),
        dict(
            **base,
            name="cp_res24_pen015",
            kind="cp",
            residual_rank=24,
            seed=12,
            seed_offset=8,
            reg=Reg(distill=0.20, residual_branch=0.15),
        ),
        dict(
            **base,
            name="cp_res32_pen012",
            kind="cp",
            residual_rank=32,
            seed=13,
            seed_offset=16,
            reg=Reg(distill=0.20, residual_branch=0.12),
        ),
        dict(
            **base,
            name="cp_res32_pen006",
            kind="cp",
            residual_rank=32,
            seed=14,
            seed_offset=24,
            reg=Reg(distill=0.18, residual_branch=0.06),
        ),
        dict(
            **base,
            name="cp_res48_pen010",
            kind="cp",
            residual_rank=48,
            seed=15,
            seed_offset=32,
            reg=Reg(distill=0.18, residual_branch=0.10),
        ),
        dict(
            **base,
            name="cp_res48_pen004",
            kind="cp",
            residual_rank=48,
            seed=16,
            seed_offset=40,
            reg=Reg(distill=0.16, residual_branch=0.04),
        ),
        dict(
            **base,
            name="split_res24_pen015",
            kind="split",
            residual_rank=24,
            seed=17,
            seed_offset=48,
            reg=Reg(distill=0.20, residual_branch=0.15),
        ),
        dict(
            **base,
            name="split_res32_pen010",
            kind="split",
            residual_rank=32,
            seed=18,
            seed_offset=56,
            reg=Reg(distill=0.18, residual_branch=0.10),
        ),
        dict(
            **{**base, "residual_scale": 0.45},
            name="cp_res32_scale045_pen010",
            kind="cp",
            residual_rank=32,
            seed=19,
            seed_offset=64,
            reg=Reg(distill=0.18, residual_branch=0.10),
        ),
        dict(
            **{**base, "topk": 2},
            name="cp_top2_res32_pen008",
            kind="cp",
            residual_rank=32,
            seed=20,
            seed_offset=72,
            reg=Reg(distill=0.18, residual_branch=0.08),
        ),
    ]


def score(row):
    return (
        0.42 * row["similarity"]
        + 0.18 * row["main_similarity"]
        + 0.25 * row["test_acc"]
        + 0.20 * row["class_selectivity"]
        + 0.10 * row["locality_7x7"]
        + 0.10 * row["pattern_gini"]
    )


def plot_frontier(rows, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for row in rows:
        color = "#4C78A8" if row["topk"] == 1 else "#E45756"
        marker = "o" if row["kind"] == "cp" else "s"
        ax.scatter(
            row["residual_fraction"],
            row["similarity"],
            s=32 + 1.8 * row["residual_rank"],
            color=color,
            marker=marker,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        label = row["name"].replace("_pen", "\npen")
        ax.annotate(label, (row["residual_fraction"], row["similarity"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("residual fraction (lower is more dictionary-owned)")
    ax.set_ylabel("tensor cosine")
    ax.set_title("Residual capacity frontier", fontstyle="italic")
    ax.grid(True, color="#e8e8e8", linewidth=0.7)
    ax.set_xlim(0.35, 0.73)
    ax.set_ylim(0.89, 0.965)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("figures/residual_capacity_frontier"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, test, base, target, train_x, train_logits, base_acc = train_base(device, args.epochs)

    rows, models = [], {}
    for spec in make_specs(args.rank):
        row, model = fit_variant(spec, target, train_x, train_logits, test, args.steps)
        row["residual_fraction"] = 1.0 - row["main_similarity"] / max(row["similarity"], 1e-8)
        row["frontier_score_2"] = score(row)
        print(row)
        rows.append(row)
        models[row["name"]] = model

    rows.sort(key=lambda r: (r["topk"] == 1, r["similarity"]), reverse=True)
    plot_frontier(rows, args.outdir / "residual_capacity_frontier.png")
    fieldnames = list(rows[0].keys())
    with (args.outdir / "residual_capacity_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    onehot = [r for r in rows if r["topk"] == 1 and r["class_selectivity"] >= 0.999]
    best_fidelity = max(onehot, key=lambda r: r["similarity"])
    best_frontier = max(onehot, key=lambda r: r["frontier_score_2"])
    best_any = max(rows, key=lambda r: r["similarity"])

    for tag, row in [("best_onehot_fidelity", best_fidelity), ("best_onehot_frontier", best_frontier), ("best_any_fidelity", best_any)]:
        example_style_plot(
            models[row["name"]],
            args.outdir / f"{tag}_{row['name']}_denoised.png",
            f"Residual frontier {tag}: {row['name']} (top pixels)",
            denoise_quantile=0.70,
        )
        example_style_plot(
            models[row["name"]],
            args.outdir / f"{tag}_{row['name']}_raw.png",
            f"Residual frontier {tag}: {row['name']} (raw)",
        )

    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={device}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"best_onehot_fidelity={best_fidelity}\n")
        f.write(f"best_onehot_frontier={best_frontier}\n")
        f.write(f"best_any_fidelity={best_any}\n")


if __name__ == "__main__":
    main()
