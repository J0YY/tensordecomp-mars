from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.search_stroke_mask_residual import Reg, example_style_plot, fit_variant
from scripts.validate_best_combo import train_base


def variant_spec(name: str, rank: int):
    specs = {
        "combo_cp_top1_res16": dict(
            name="combo_cp_top1_res16",
            kind="cp",
            rank=rank,
            residual_rank=16,
            topk=1,
            residual_scale=0.35,
            seed=3,
            seed_offset=12,
            lr=0.04,
            reg=Reg(distill=0.20, residual_branch=0.20),
        ),
        "combo_split_top1_res16": dict(
            name="combo_split_top1_res16",
            kind="split",
            rank=rank,
            residual_rank=16,
            topk=1,
            residual_scale=0.35,
            seed=7,
            seed_offset=36,
            lr=0.04,
            reg=Reg(distill=0.20, residual_branch=0.20),
        ),
    }
    if name not in specs:
        raise ValueError(f"unknown variant {name}; choose one of {sorted(specs)}")
    return specs[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="combo_cp_top1_res16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=340)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("figures/high_fidelity_onehot"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, test, base, target, train_x, train_logits, base_acc = train_base(device, args.epochs)
    spec = variant_spec(args.variant, args.rank)
    row, model = fit_variant(spec, target, train_x, train_logits, test, args.steps)
    row["base_acc"] = base_acc
    row["device"] = device
    print(row)

    with (args.outdir / f"{args.variant}_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    example_style_plot(
        model,
        args.outdir / f"{args.variant}_denoised.png",
        f"High-fidelity one-hot variant: {args.variant} (top pixels)",
        denoise_quantile=0.70,
    )
    example_style_plot(
        model,
        args.outdir / f"{args.variant}_raw.png",
        f"High-fidelity one-hot variant: {args.variant} (raw)",
    )


if __name__ == "__main__":
    main()
