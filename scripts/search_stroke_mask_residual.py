from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from kornia.augmentation import RandomGaussianNoise

from image import MNIST, Model
from scripts.search_stroke_templates import (
    class_selectivity,
    gini_1d,
    head_top1_frac,
    interaction_tensor,
    laplacian_loss,
    make_template_bank,
    mean_gini,
    normalize_columns,
    patch_locality,
    set_seed,
    total_variation,
)


def device_name():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def avg_smooth(x, passes=1):
    img = x.T.reshape(-1, 1, 28, 28)
    for _ in range(passes):
        img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    return img.reshape(img.shape[0], -1).T


def init_centers(rank, device):
    side = math.ceil(math.sqrt(rank))
    coords = torch.linspace(-0.72, 0.72, side, device=device)
    centers = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)[:rank]
    return centers


def gaussian_masks(center_raw, log_sigma, min_sigma=0.18, max_sigma=0.65):
    centers = 0.9 * torch.tanh(center_raw)
    sigma = min_sigma + (max_sigma - min_sigma) * torch.sigmoid(log_sigma)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 28, device=center_raw.device),
        torch.linspace(-1, 1, 28, device=center_raw.device),
        indexing="ij",
    )
    grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    dist = (grid[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
    masks = torch.exp(-dist / (2 * sigma[None, :].square()))
    return masks / masks.max(dim=0, keepdim=True).values.clamp_min(1e-8), sigma


@dataclass
class Reg:
    tensor: float = 1.0
    distill: float = 0.16
    anchor: float = 0.04
    residual_l2: float = 0.01
    mask_residual_l2: float = 0.01
    raw_tv: float = 0.015
    raw_lap: float = 0.004
    d_l1: float = 0.0002
    sigma: float = 0.002
    residual_branch: float = 0.25


class StrokeMaskResidual(nn.Module):
    def __init__(
        self,
        kind: str,
        rank: int,
        residual_rank: int,
        topk_head: int,
        residual_scale: float,
        device: str,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.kind = kind
        self.rank = rank
        self.residual_rank = residual_rank
        self.topk_head = topk_head
        self.residual_scale = residual_scale
        bank = make_template_bank(rank + seed_offset + 10, device)
        self.register_buffer("anchor_a", bank[:, seed_offset : seed_offset + rank])
        self.register_buffer("anchor_b", bank[:, seed_offset + 4 : seed_offset + 4 + rank])
        if self.anchor_b.shape[1] < rank:
            self.anchor_b = bank[:, :rank]

        self.delta_a = nn.Parameter(torch.zeros(784, rank, device=device))
        self.delta_b = nn.Parameter(torch.zeros(784, rank, device=device))
        self.log_amp_a = nn.Parameter(torch.zeros(rank, device=device))
        self.log_amp_b = nn.Parameter(torch.zeros(rank, device=device))

        centers = init_centers(rank, device)
        self.center_a = nn.Parameter(torch.atanh((centers / 0.9).clamp(-0.95, 0.95)))
        self.center_b = nn.Parameter(torch.atanh((torch.roll(centers, 3, 0) / 0.9).clamp(-0.95, 0.95)))
        self.log_sigma_a = nn.Parameter(torch.zeros(rank, device=device))
        self.log_sigma_b = nn.Parameter(torch.zeros(rank, device=device))

        self.raw_d = nn.Parameter(0.03 * torch.randn(10, rank, device=device))

        self.res_a = nn.Parameter(0.015 * torch.randn(784, residual_rank, device=device))
        self.res_b = nn.Parameter(0.015 * torch.randn(784, residual_rank, device=device))
        self.res_d = nn.Parameter(0.015 * torch.randn(10, residual_rank, device=device))

    def main_factors(self):
        mask_a, sig_a = gaussian_masks(self.center_a, self.log_sigma_a)
        mask_b, sig_b = gaussian_masks(self.center_b, self.log_sigma_b)
        a = (self.anchor_a + self.residual_scale * avg_smooth(self.delta_a, 1)) * mask_a
        b = (self.anchor_b + self.residual_scale * avg_smooth(self.delta_b, 1)) * mask_b
        a = a * self.log_amp_a.exp().unsqueeze(0)
        b = b * self.log_amp_b.exp().unsqueeze(0)
        return a, b, sig_a, sig_b

    def down(self):
        idx = self.raw_d.detach().abs().topk(self.topk_head, dim=0).indices
        mask = torch.zeros_like(self.raw_d)
        mask.scatter_(0, idx, 1.0)
        return self.raw_d * mask

    def main_tensor(self):
        a, b, _, _ = self.main_factors()
        d = self.down()
        if self.kind == "cp":
            t = einsum(d, a, b, "c r, i r, j r -> c i j")
            return 0.5 * (t + t.mT)
        return einsum(d, a, a, "c r, i r, j r -> c i j") - einsum(d, b, b, "c r, i r, j r -> c i j")

    def residual_tensor(self):
        t = einsum(self.res_d, self.res_a, self.res_b, "c r, i r, j r -> c i j")
        return 0.5 * (t + t.mT)

    def tensor(self):
        return self.main_tensor() + self.residual_tensor()

    def forward(self, x):
        a, b, _, _ = self.main_factors()
        if self.kind == "cp":
            main = (x @ a) * (x @ b)
        else:
            main = (x @ a).square() - (x @ b).square()
        res = (x @ self.res_a) * (x @ self.res_b)
        return main @ self.down().T + res @ self.res_d.T

    def decompose(self, visual=True):
        a, b, _, _ = self.main_factors()
        if self.kind == "cp":
            plus = normalize_columns(a + b)
            minus = normalize_columns(a - b)
            sigma = a.norm(dim=0) * b.norm(dim=0) * self.down().norm(dim=0)
        else:
            plus = normalize_columns(a)
            minus = normalize_columns(b)
            sigma = (a.norm(dim=0).square() + b.norm(dim=0).square()) * self.down().norm(dim=0)
        down = self.down()
        if visual:
            loc = torch.stack([patch_locality(torch.stack([plus[:, i], minus[:, i]], dim=1)) for i in range(self.rank)])
            gin = torch.stack([gini_1d(plus[:, i]) + gini_1d(minus[:, i]) for i in range(self.rank)])
            head = down.abs().max(dim=0).values / down.abs().sum(dim=0).clamp_min(1e-8)
            score = sigma / sigma.max().clamp_min(1e-8) + 0.85 * loc + 0.65 * gin + 0.8 * head
            order = score.argsort(descending=True)
        else:
            order = sigma.argsort(descending=True)
        return plus[:, order], minus[:, order], down[:, order], sigma[order]

    def regularizer(self, reg: Reg):
        a, b, sig_a, sig_b = self.main_factors()
        anchor = F.mse_loss(normalize_columns(a), normalize_columns(self.anchor_a)) + F.mse_loss(
            normalize_columns(b), normalize_columns(self.anchor_b)
        )
        residual = self.delta_a.square().mean() + self.delta_b.square().mean()
        smooth = total_variation(torch.cat([a, b], dim=1)) + 0.5 * laplacian_loss(torch.cat([a, b], dim=1))
        res_branch = self.res_a.square().mean() + self.res_b.square().mean() + self.res_d.square().mean()
        sig_penalty = (sig_a.mean() + sig_b.mean())
        return (
            reg.anchor * anchor
            + reg.mask_residual_l2 * residual
            + reg.raw_tv * smooth
            + reg.d_l1 * self.down().abs().mean()
            + reg.residual_branch * res_branch
            + reg.sigma * sig_penalty
        )


def fit_variant(spec, target, train_x, train_logits, test, steps):
    set_seed(spec["seed"])
    model = StrokeMaskResidual(
        kind=spec["kind"],
        rank=spec["rank"],
        residual_rank=spec["residual_rank"],
        topk_head=spec["topk"],
        residual_scale=spec["residual_scale"],
        device=target.device,
        seed_offset=spec["seed_offset"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    reg: Reg = spec["reg"]
    logit_scale = train_logits.std().detach().clamp_min(1e-4)
    for _ in range(steps):
        tensor_loss = 1 - F.cosine_similarity(model.tensor().flatten(), target.flatten(), dim=0)
        distill = F.mse_loss(model(train_x) / logit_scale, train_logits / logit_scale)
        loss = reg.tensor * tensor_loss + reg.distill * distill + model.regularizer(reg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        plus, minus, down, sigma = model.decompose(visual=True)
        display = torch.cat([plus, minus], dim=1)
        logits = model(test.x)
        sim = F.cosine_similarity(model.tensor().flatten(), target.flatten(), dim=0)
        main_sim = F.cosine_similarity(model.main_tensor().flatten(), target.flatten(), dim=0)
        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            "rank": spec["rank"],
            "residual_rank": spec["residual_rank"],
            "topk": spec["topk"],
            "similarity": sim.item(),
            "main_similarity": main_sim.item(),
            "test_acc": (logits.argmax(-1) == test.y).float().mean().item(),
            "pattern_gini": mean_gini(display).item(),
            "locality_7x7": patch_locality(display).item(),
            "class_selectivity": class_selectivity(down).item(),
            "head_top1_frac": head_top1_frac(down).item(),
            "display_tv": total_variation(display).item(),
            "top_sigma_frac": (sigma[:8].sum() / sigma.sum().clamp_min(1e-8)).item(),
        }
        row["frontier_score"] = (
            0.40 * row["similarity"]
            + 0.25 * row["main_similarity"]
            + 0.30 * row["test_acc"]
            + 0.45 * row["pattern_gini"]
            + 0.65 * row["locality_7x7"]
            + 0.60 * row["class_selectivity"]
            + 0.15 * row["head_top1_frac"]
        )
    return row, model


def example_style_plot(model, out_path, title, denoise_quantile=None):
    plus, minus, down, _ = model.decompose(visual=True)
    plus, minus, down = plus.detach().cpu(), minus.detach().cpu(), down.detach().cpu()
    if denoise_quantile is not None:
        for pats in (plus, minus):
            thresh = pats.abs().quantile(denoise_quantile, dim=0, keepdim=True)
            pats *= pats.abs() >= thresh
    k = min(10, plus.shape[1])
    fig, axes = plt.subplots(3, k, figsize=(11.5, 3.8), gridspec_kw={"height_ratios": [1, 1, 0.6], "hspace": 0.08, "wspace": 0.22})
    vmax = max(float(plus[:, :k].abs().max()), float(minus[:, :k].abs().max()))
    for i in range(k):
        axes[0, i].imshow(plus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, i].imshow(minus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        colors = ["#4c78a8" if v >= 0 else "#d62728" for v in down[:, i].tolist()]
        axes[2, i].bar(range(10), down[:, i], color=colors, width=0.75)
        axes[2, i].axhline(0, color="#dddddd", linewidth=0.6)
        axes[2, i].set_xticks([])
        axes[2, i].tick_params(left=False, labelleft=False)
        for r in [0, 1]:
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
        for spine in axes[2, i].spines.values():
            spine.set_visible(False)
    for label, row in [("L+R / pos", 0), ("L-R / neg", 1), ("head", 2)]:
        axes[row, -1].set_ylabel(label, rotation=270, labelpad=18, fontsize=11)
        axes[row, -1].yaxis.set_label_position("right")
    fig.suptitle(title, y=0.99, fontsize=13, fontstyle="italic")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("figures/stroke_mask_residual"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = device_name()

    set_seed(911)
    train = MNIST(train=True, device=dev)
    test = MNIST(train=False, device=dev)
    base = Model.from_config(epochs=args.epochs).to(dev)
    base.fit(train, test, RandomGaussianNoise(std=0.55).to(dev))
    with torch.no_grad():
        target = interaction_tensor(base).detach()
        idx = torch.randperm(train.x.shape[0], device=train.x.device)[:8192]
        train_x = train.x[idx]
        train_logits = base(train_x).detach()
        base_acc = (base(test.x).argmax(-1) == test.y).float().mean().item()

    specs = [
        dict(name="combo_cp_top1_res8", kind="cp", rank=args.rank, residual_rank=8, topk=1, residual_scale=0.30, seed=1, seed_offset=0, lr=0.04, reg=Reg(distill=0.20, residual_branch=0.30)),
        dict(name="combo_cp_top2_res8", kind="cp", rank=args.rank, residual_rank=8, topk=2, residual_scale=0.30, seed=2, seed_offset=6, lr=0.04, reg=Reg(distill=0.18, residual_branch=0.30)),
        dict(name="combo_cp_top1_res16", kind="cp", rank=args.rank, residual_rank=16, topk=1, residual_scale=0.35, seed=3, seed_offset=12, lr=0.04, reg=Reg(distill=0.20, residual_branch=0.20)),
        dict(name="combo_cp_top2_res16", kind="cp", rank=args.rank, residual_rank=16, topk=2, residual_scale=0.35, seed=4, seed_offset=18, lr=0.04, reg=Reg(distill=0.18, residual_branch=0.20)),
        dict(name="combo_split_top1_res8", kind="split", rank=args.rank, residual_rank=8, topk=1, residual_scale=0.30, seed=5, seed_offset=24, lr=0.04, reg=Reg(distill=0.20, residual_branch=0.30)),
        dict(name="combo_split_top2_res8", kind="split", rank=args.rank, residual_rank=8, topk=2, residual_scale=0.30, seed=6, seed_offset=30, lr=0.04, reg=Reg(distill=0.18, residual_branch=0.30)),
        dict(name="combo_split_top1_res16", kind="split", rank=args.rank, residual_rank=16, topk=1, residual_scale=0.35, seed=7, seed_offset=36, lr=0.04, reg=Reg(distill=0.20, residual_branch=0.20)),
        dict(name="combo_split_top2_res16", kind="split", rank=args.rank, residual_rank=16, topk=2, residual_scale=0.35, seed=8, seed_offset=42, lr=0.04, reg=Reg(distill=0.18, residual_branch=0.20)),
    ]

    rows, models = [], {}
    for spec in specs:
        row, model = fit_variant(spec, target, train_x, train_logits, test, args.steps)
        print(row)
        rows.append(row)
        models[row["name"]] = model

    rows = sorted(rows, key=lambda r: r["frontier_score"], reverse=True)
    with (args.outdir / "stroke_mask_residual_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_model = models[best["name"]]
    example_style_plot(best_model, args.outdir / "best_stroke_mask_residual_raw.png", f"Stroke templates + masks + residual: {best['name']} (raw)")
    example_style_plot(best_model, args.outdir / "best_stroke_mask_residual_denoised.png", f"Stroke templates + masks + residual: {best['name']} (top pixels)", denoise_quantile=0.70)
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={dev}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"best={best}\n")


if __name__ == "__main__":
    main()

