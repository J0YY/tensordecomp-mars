from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from kornia.augmentation import RandomGaussianNoise

from image import MNIST, Model
from scripts.search_stroke_mask_residual import Reg, StrokeMaskResidual, fit_variant
from scripts.search_stroke_templates import interaction_tensor, normalize_columns, patch_locality


def device_name():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_base(device, epochs):
    train = MNIST(train=True, device=device)
    test = MNIST(train=False, device=device)
    set_seed(2027)
    base = Model.from_config(epochs=epochs).to(device)
    base.fit(train, test, RandomGaussianNoise(std=0.55).to(device))
    with torch.no_grad():
        target = interaction_tensor(base).detach()
        idx = torch.randperm(train.x.shape[0], device=train.x.device)[:8192]
        train_x = train.x[idx]
        train_logits = base(train_x).detach()
        base_acc = (base(test.x).argmax(-1) == test.y).float().mean().item()
    return train, test, base, target, train_x, train_logits, base_acc


def best_spec(seed=1, rank=64, steps=260):
    return dict(
        name=f"combo_cp_top1_res8_seed{seed}",
        kind="cp",
        rank=rank,
        residual_rank=8,
        topk=1,
        residual_scale=0.30,
        seed=seed,
        seed_offset=0,
        lr=0.04,
        reg=Reg(distill=0.20, residual_branch=0.30),
        steps=steps,
    )


@torch.no_grad()
def component_patterns(model, k=16):
    plus, minus, down, sigma = model.decompose(visual=True)
    patterns = torch.cat([plus[:, :k], minus[:, :k]], dim=1)
    return normalize_columns(patterns), plus, minus, down, sigma


@torch.no_grad()
def greedy_match_score(reference, candidate):
    sim = (reference.T @ candidate).abs().detach().cpu()
    used_ref, used_cand, vals = set(), set(), []
    for flat in sim.flatten().argsort(descending=True).tolist():
        i, j = divmod(flat, sim.shape[1])
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        vals.append(float(sim[i, j]))
        if len(vals) == min(sim.shape):
            break
    return sum(vals) / max(1, len(vals)), vals


@torch.no_grad()
def activation_gallery(model, train, out_path, k=8, topn=8):
    plus, minus, down, sigma = model.decompose(visual=False)
    a, b, _, _ = model.main_factors()
    acts = (train.x @ a) * (train.x @ b)
    k = min(k, plus.shape[1])
    fig, axes = plt.subplots(k, topn + 2, figsize=(1.15 * (topn + 2), 1.1 * k))
    vmax = max(float(plus[:, :k].abs().max()), float(minus[:, :k].abs().max()))
    for row in range(k):
        axes[row, 0].imshow(plus[:, row].detach().cpu().reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 1].imshow(minus[:, row].detach().cpu().reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        top = acts[:, row].abs().topk(topn).indices
        for col, idx in enumerate(top.tolist(), start=2):
            axes[row, col].imshow(train.x[idx].detach().cpu().reshape(28, 28), cmap="gray_r")
        for col in range(topn + 2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        cls = int(down[:, row].abs().argmax().item())
        sign = "+" if down[cls, row] >= 0 else "-"
        axes[row, 0].set_ylabel(f"c{row}\n{sign}{cls}", rotation=0, labelpad=20, va="center")
    axes[0, 0].set_title("pos")
    axes[0, 1].set_title("neg")
    for i in range(topn):
        axes[0, i + 2].set_title(f"ex{i+1}")
    fig.suptitle("Activation gallery for best displayed components", y=1.01, fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def class_consensus_panel(models, seeds, out_path):
    rows = 10
    cols = len(seeds) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(1.05 * cols, 1.05 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row_cls in range(10):
        row_images = []
        selections = []
        for seed in seeds:
            model = models[seed]
            plus, minus, down, sigma = model.decompose(visual=True)
            strength = down[row_cls].abs() * sigma / sigma.max().clamp_min(1e-8)
            idx = int(strength.argmax().item())
            row_images.extend([plus[:, idx], minus[:, idx]])
            sign = "+" if down[row_cls, idx] >= 0 else "-"
            selections.append((idx, sign))
        vmax = max(float(torch.stack(row_images, dim=1).abs().max()), 1e-6)
        for seed_pos, seed in enumerate(seeds):
            idx, sign = selections[seed_pos]
            model = models[seed]
            plus, minus, down, sigma = model.decompose(visual=True)
            for offset, image in enumerate([plus[:, idx], minus[:, idx]]):
                ax = axes[row_cls, 2 * seed_pos + offset]
                ax.imshow(image.detach().cpu().reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_cls == 0:
                    ax.set_title(f"s{seed} {'pos' if offset == 0 else 'neg'}", fontsize=8)
            axes[row_cls, 2 * seed_pos].set_ylabel(f"{row_cls} {sign}", rotation=0, labelpad=15, va="center")
    fig.suptitle("Best displayed component per digit across seeds", y=1.01, fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("figures/best_combo_validation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = device_name()
    train, test, base, target, train_x, train_logits, base_acc = train_base(device, args.epochs)

    rows, models = [], {}
    for seed in args.seeds:
        spec = best_spec(seed=seed, rank=args.rank, steps=args.steps)
        row, model = fit_variant(spec, target, train_x, train_logits, test, args.steps)
        row["seed"] = seed
        print(row)
        rows.append(row)
        models[seed] = model

    ref_patterns, *_ = component_patterns(models[args.seeds[0]], k=12)
    for row in rows:
        pats, *_ = component_patterns(models[row["seed"]], k=12)
        score, vals = greedy_match_score(ref_patterns, pats)
        row["stability_vs_seed0"] = score
        row["top12_min_match"] = min(vals) if vals else 0.0

    with (args.outdir / "best_combo_seed_stability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = sorted(rows, key=lambda r: r["frontier_score"], reverse=True)[0]
    activation_gallery(models[best["seed"]], train, args.outdir / "best_combo_activation_gallery.png")
    class_consensus_panel(models, args.seeds, args.outdir / "best_combo_class_consensus.png")
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={device}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"best={best}\n")
        f.write(f"mean_stability={sum(r['stability_vs_seed0'] for r in rows[1:]) / max(1, len(rows)-1):.6f}\n")


if __name__ == "__main__":
    main()
