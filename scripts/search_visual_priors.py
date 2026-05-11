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


def normalize_columns(x, eps=1e-8):
    return x / x.norm(dim=0, keepdim=True).clamp_min(eps)


def images(x):
    return x.T.reshape(-1, 1, 28, 28)


def flatten_images(x):
    return x.reshape(x.shape[0], -1).T


def avg_smooth(x, passes=1):
    if passes <= 0:
        return x
    img = images(x)
    for _ in range(passes):
        img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    return flatten_images(img)


def total_variation(patterns):
    img = images(patterns)
    return (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean() + (
        img[:, :, :, 1:] - img[:, :, :, :-1]
    ).abs().mean()


def laplacian_loss(patterns):
    img = images(patterns)
    kern = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=patterns.device)
    kern = kern.reshape(1, 1, 3, 3)
    return F.conv2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), kern).abs().mean()


def gini_1d(x, eps=1e-8):
    x = x.detach().abs().flatten()
    if x.numel() == 0 or x.sum() < eps:
        return x.new_tensor(0.0)
    x = x.sort().values
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return ((2 * idx - n - 1) * x).sum() / (n * x.sum().clamp_min(eps))


def mean_gini(patterns):
    return torch.stack([gini_1d(col) for col in patterns.T]).mean()


def patch_locality(patterns, patch=7, eps=1e-8):
    img = images(patterns)
    energy = img.square()
    pooled = F.avg_pool2d(energy, patch, stride=1) * patch * patch
    best = pooled.flatten(1).max(dim=1).values
    total = energy.flatten(1).sum(dim=1).clamp_min(eps)
    return (best / total).mean()


def class_selectivity(down, eps=1e-8):
    p = down.detach().abs()
    p = p / p.sum(dim=0, keepdim=True).clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum(dim=0) / math.log(p.shape[0])
    return 1 - entropy.mean()


def head_top1_frac(down, eps=1e-8):
    p = down.detach().abs()
    return (p.max(dim=0).values / p.sum(dim=0).clamp_min(eps)).mean()


def make_masks(rank, sigma=0.38, device="cpu"):
    side = math.ceil(math.sqrt(rank))
    coords = torch.linspace(-0.75, 0.75, side, device=device)
    centers = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)[:rank]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 28, device=device),
        torch.linspace(-1, 1, 28, device=device),
        indexing="ij",
    )
    grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    dist = (grid[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
    masks = torch.exp(-dist / (2 * sigma**2))
    return masks / masks.max(dim=0, keepdim=True).values.clamp_min(1e-8)


@torch.no_grad()
def interaction_tensor(model):
    l, r = model.w_lr[0].unbind()
    b = einsum(
        model.w_u,
        l,
        r,
        model.w_e,
        model.w_e,
        "cls out, out hid1, out hid2, hid1 pix1, hid2 pix2 -> cls pix1 pix2",
    )
    return 0.5 * (b + b.mT)


@dataclass
class Reg:
    raw_l1: float = 0.0
    raw_tv: float = 0.0
    raw_lap: float = 0.0
    display_l1: float = 0.0
    display_tv: float = 0.0
    d_l1: float = 0.0
    d_entropy: float = 0.0
    duplicate: float = 0.0
    distill: float = 0.0


class VisualFactor(nn.Module):
    def __init__(
        self,
        kind: str,
        rank: int,
        smooth_passes: int = 0,
        topk_head: int | None = None,
        mask_sigma: float | None = None,
        device: str = "cpu",
        scale: float = 0.02,
    ):
        super().__init__()
        self.kind = kind
        self.rank = rank
        self.smooth_passes = smooth_passes
        self.topk_head = topk_head
        self.raw_a = nn.Parameter(scale * torch.randn(784, rank))
        self.raw_b = nn.Parameter(scale * torch.randn(784, rank))
        self.raw_d = nn.Parameter(scale * torch.randn(10, rank))
        self.register_buffer("masks", make_masks(rank, mask_sigma, device) if mask_sigma else torch.ones(784, rank, device=device))

    def factors(self):
        a = avg_smooth(self.raw_a, self.smooth_passes) * self.masks
        b = avg_smooth(self.raw_b, self.smooth_passes) * self.masks
        return a, b

    def down(self):
        if self.topk_head is None:
            return self.raw_d
        k = min(self.topk_head, self.raw_d.shape[0])
        idx = self.raw_d.detach().abs().topk(k, dim=0).indices
        mask = torch.zeros_like(self.raw_d)
        mask.scatter_(0, idx, 1.0)
        return self.raw_d * mask

    def tensor(self):
        a, b = self.factors()
        d = self.down()
        if self.kind == "cp":
            t = einsum(d, a, b, "c r, i r, j r -> c i j")
            return 0.5 * (t + t.mT)
        return einsum(d, a, a, "c r, i r, j r -> c i j") - einsum(d, b, b, "c r, i r, j r -> c i j")

    def activations(self, x):
        a, b = self.factors()
        if self.kind == "cp":
            return (x @ a) * (x @ b)
        return (x @ a).square() - (x @ b).square()

    def forward(self, x):
        return self.activations(x) @ self.down().T

    def strengths(self):
        a, b = self.factors()
        if self.kind == "cp":
            return a.norm(dim=0) * b.norm(dim=0) * self.down().norm(dim=0)
        return (a.norm(dim=0).square() + b.norm(dim=0).square()) * self.down().norm(dim=0)

    def decompose(self, order="sigma"):
        a, b = self.factors()
        if self.kind == "cp":
            plus = normalize_columns(a + b)
            minus = normalize_columns(a - b)
        else:
            plus = normalize_columns(a)
            minus = normalize_columns(b)
        sigma = self.strengths()
        down = self.down()
        if order == "visual":
            pats = torch.cat([plus, minus], dim=1)
            g = torch.stack([gini_1d(plus[:, i]) + gini_1d(minus[:, i]) for i in range(self.rank)])
            loc = torch.stack([patch_locality(torch.stack([plus[:, i], minus[:, i]], dim=1)) for i in range(self.rank)])
            sel = down.abs().max(dim=0).values / down.abs().sum(dim=0).clamp_min(1e-8)
            score = sigma / sigma.max().clamp_min(1e-8) + 0.45 * g + 0.65 * loc + 0.8 * sel
            idx = score.argsort(descending=True)
        else:
            idx = sigma.argsort(descending=True)
        return plus[:, idx], minus[:, idx], down[:, idx], sigma[idx]

    def regularizer(self, reg: Reg):
        a, b = self.factors()
        plus, minus, down, _ = self.decompose()
        raw = torch.cat([a, b], dim=1)
        display = torch.cat([plus, minus], dim=1)
        loss = raw.new_tensor(0.0)
        if reg.raw_l1:
            loss = loss + reg.raw_l1 * raw.abs().mean()
        if reg.raw_tv:
            loss = loss + reg.raw_tv * total_variation(raw)
        if reg.raw_lap:
            loss = loss + reg.raw_lap * laplacian_loss(raw)
        if reg.display_l1:
            loss = loss + reg.display_l1 * display.abs().mean()
        if reg.display_tv:
            loss = loss + reg.display_tv * total_variation(display)
        if reg.d_l1:
            loss = loss + reg.d_l1 * down.abs().mean()
        if reg.d_entropy:
            p = down.abs() / down.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)
            ent = -(p * (p + 1e-8).log()).sum(dim=0).mean() / math.log(10)
            loss = loss + reg.d_entropy * ent
        if reg.duplicate:
            n = normalize_columns(display)
            gram = n.T @ n - torch.eye(n.shape[1], device=n.device)
            loss = loss + reg.duplicate * gram.square().mean()
        return loss


def fit_variant(spec, target, base, train_x, train_logits, test, steps):
    set_seed(spec["seed"])
    model = VisualFactor(
        spec["kind"],
        spec["rank"],
        smooth_passes=spec.get("smooth", 0),
        topk_head=spec.get("topk"),
        mask_sigma=spec.get("mask_sigma"),
        device=target.device,
    ).to(target.device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("wd", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    reg: Reg = spec["reg"]
    logit_scale = train_logits.std().detach().clamp_min(1e-4)
    for step in range(steps):
        approx = model.tensor()
        recon = 1 - F.cosine_similarity(approx.flatten(), target.flatten(), dim=0)
        loss = recon + model.regularizer(reg)
        if reg.distill:
            pred = model(train_x)
            loss = loss + reg.distill * F.mse_loss(pred / logit_scale, train_logits / logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        plus, minus, down, sigma = model.decompose()
        display = torch.cat([plus, minus], dim=1)
        logits = model(test.x)
        sim = F.cosine_similarity(model.tensor().flatten(), target.flatten(), dim=0)
        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            "rank": spec["rank"],
            "smooth": spec.get("smooth", 0),
            "topk": spec.get("topk") or 0,
            "mask_sigma": spec.get("mask_sigma") or 0.0,
            "similarity": sim.item(),
            "test_acc": (logits.argmax(-1) == test.y).float().mean().item(),
            "pattern_gini": mean_gini(display).item(),
            "locality_7x7": patch_locality(display).item(),
            "class_selectivity": class_selectivity(down).item(),
            "head_top1_frac": head_top1_frac(down).item(),
            "display_tv": total_variation(display).item(),
            "top_sigma_frac": (sigma[:8].sum() / sigma.sum().clamp_min(1e-8)).item(),
        }
        row["human_score"] = (
            0.35 * row["similarity"]
            + 0.30 * row["test_acc"]
            + 0.65 * row["pattern_gini"]
            + 0.85 * row["locality_7x7"]
            + 0.90 * row["class_selectivity"]
            + 0.25 * row["head_top1_frac"]
        )
    return row, model


def example_style_plot(model, out_path, title, order="visual", k=10, denoise_quantile=None):
    plus, minus, down, sigma = model.decompose(order=order)
    plus, minus, down = plus.detach().cpu(), minus.detach().cpu(), down.detach().cpu()
    if denoise_quantile is not None:
        for pats in (plus, minus):
            thresh = pats.abs().quantile(denoise_quantile, dim=0, keepdim=True)
            pats *= (pats.abs() >= thresh)
    k = min(k, plus.shape[1])
    fig, axes = plt.subplots(3, k, figsize=(1.15 * k, 3.75), gridspec_kw={"height_ratios": [1, 1, 0.6], "hspace": 0.08, "wspace": 0.22})
    vmax = max(float(plus[:, :k].abs().max()), float(minus[:, :k].abs().max()))
    for i in range(k):
        axes[0, i].imshow(plus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, i].imshow(minus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        colors = ["#4c78a8" if v >= 0 else "#d62728" for v in down[:, i].tolist()]
        axes[2, i].bar(range(10), down[:, i], color=colors, width=0.75)
        axes[2, i].axhline(0, color="#dddddd", linewidth=0.6)
        axes[2, i].set_xticks([])
        axes[2, i].tick_params(left=False, labelleft=False)
        for r in range(2):
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
    parser.add_argument("--outdir", type=Path, default=Path("figures/visual_priors"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = device_name()

    set_seed(321)
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
        dict(name="raw_smooth_cp_top3", kind="cp", rank=args.rank, smooth=1, topk=3, lr=0.035, seed=1, reg=Reg(raw_l1=2e-4, raw_tv=5e-2, raw_lap=1e-2, d_l1=2e-4, distill=0.08)),
        dict(name="raw_smooth_cp_top2", kind="cp", rank=args.rank, smooth=1, topk=2, lr=0.035, seed=2, reg=Reg(raw_l1=2e-4, raw_tv=5e-2, raw_lap=1e-2, d_l1=2e-4, distill=0.08)),
        dict(name="smooth2_cp_top3", kind="cp", rank=args.rank, smooth=2, topk=3, lr=0.035, seed=3, reg=Reg(raw_l1=2e-4, raw_tv=4e-2, raw_lap=8e-3, d_l1=2e-4, distill=0.10)),
        dict(name="masked_cp_wide_top3", kind="cp", rank=args.rank, smooth=1, topk=3, mask_sigma=0.50, lr=0.04, seed=4, reg=Reg(raw_l1=2e-4, raw_tv=2e-2, raw_lap=6e-3, distill=0.10)),
        dict(name="masked_cp_tight_top3", kind="cp", rank=args.rank, smooth=1, topk=3, mask_sigma=0.34, lr=0.04, seed=5, reg=Reg(raw_l1=2e-4, raw_tv=2e-2, raw_lap=6e-3, distill=0.12)),
        dict(name="raw_smooth_split_top3", kind="split", rank=args.rank, smooth=1, topk=3, lr=0.035, seed=6, reg=Reg(raw_l1=2e-4, raw_tv=5e-2, raw_lap=1e-2, d_l1=2e-4, distill=0.08)),
        dict(name="raw_smooth_split_top2", kind="split", rank=args.rank, smooth=1, topk=2, lr=0.035, seed=7, reg=Reg(raw_l1=2e-4, raw_tv=5e-2, raw_lap=1e-2, d_l1=2e-4, distill=0.08)),
        dict(name="smooth2_split_top3", kind="split", rank=args.rank, smooth=2, topk=3, lr=0.035, seed=8, reg=Reg(raw_l1=2e-4, raw_tv=4e-2, raw_lap=8e-3, d_l1=2e-4, distill=0.10)),
        dict(name="masked_split_wide_top3", kind="split", rank=args.rank, smooth=1, topk=3, mask_sigma=0.50, lr=0.04, seed=9, reg=Reg(raw_l1=2e-4, raw_tv=2e-2, raw_lap=6e-3, distill=0.10)),
        dict(name="masked_split_tight_top3", kind="split", rank=args.rank, smooth=1, topk=3, mask_sigma=0.34, lr=0.04, seed=10, reg=Reg(raw_l1=2e-4, raw_tv=2e-2, raw_lap=6e-3, distill=0.12)),
    ]

    rows, models = [], {}
    for spec in specs:
        row, model = fit_variant(spec, target, base, train_x, train_logits, test, args.steps)
        print(row)
        rows.append(row)
        models[row["name"]] = model

    rows = sorted(rows, key=lambda r: r["human_score"], reverse=True)
    with (args.outdir / "visual_prior_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_model = models[best["name"]]
    example_style_plot(best_model, args.outdir / "best_visual_raw.png", f"Best visual-prior variant: {best['name']} (raw)", order="visual")
    example_style_plot(best_model, args.outdir / "best_visual_denoised.png", f"Best visual-prior variant: {best['name']} (top pixels)", order="visual", denoise_quantile=0.72)
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={dev}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"best={best}\n")


if __name__ == "__main__":
    main()

