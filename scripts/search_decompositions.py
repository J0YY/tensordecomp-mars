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


def total_variation(patterns):
    imgs = patterns.T.reshape(-1, 1, 28, 28)
    return (imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).abs().mean() + (
        imgs[:, :, :, 1:] - imgs[:, :, :, :-1]
    ).abs().mean()


def patch_locality(patterns, patch=7, eps=1e-8):
    imgs = patterns.T.reshape(-1, 1, 28, 28)
    energy = imgs.square()
    pooled = F.avg_pool2d(energy, patch, stride=1) * patch * patch
    best = pooled.flatten(1).max(dim=1).values
    total = energy.flatten(1).sum(dim=1).clamp_min(eps)
    return (best / total).mean()


def class_selectivity(down, eps=1e-8):
    p = down.detach().abs()
    p = p / p.sum(dim=0, keepdim=True).clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum(dim=0) / math.log(p.shape[0])
    return 1 - entropy.mean()


def head_entropy(down, eps=1e-8):
    p = down.abs()
    p = p / p.sum(dim=0, keepdim=True).clamp_min(eps)
    return (-(p * (p + eps).log()).sum(dim=0) / math.log(p.shape[0])).mean()


def border_penalty(patterns):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 28, device=patterns.device),
        torch.linspace(-1, 1, 28, device=patterns.device),
        indexing="ij",
    )
    dist = (xx.square() + yy.square()).reshape(-1, 1)
    return (patterns.square() * dist).mean()


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
    l1: float = 0.0
    tv: float = 0.0
    d_l1: float = 0.0
    entropy: float = 0.0
    duplicate: float = 0.0
    border: float = 0.0
    symmetry: float = 0.0


class Factor(nn.Module):
    def __init__(self, kind: str, rank: int, scale: float = 0.02, nonnegative: bool = False):
        super().__init__()
        self.kind = kind
        self.rank = rank
        self.nonnegative = nonnegative
        if kind == "cp":
            self.a = nn.Parameter(scale * torch.randn(784, rank))
            self.b = nn.Parameter(scale * torch.randn(784, rank))
        elif kind == "sym":
            self.a = nn.Parameter(scale * torch.randn(784, rank))
            self.b = None
        elif kind == "split":
            shift = -4.0 if nonnegative else 0.0
            self.a = nn.Parameter(scale * torch.randn(784, rank) + shift)
            self.b = nn.Parameter(scale * torch.randn(784, rank) + shift)
        else:
            raise ValueError(kind)
        self.d = nn.Parameter(scale * torch.randn(10, rank))

    def pos(self):
        return F.softplus(self.a) if self.nonnegative else self.a

    def neg_or_right(self):
        if self.b is None:
            return None
        return F.softplus(self.b) if self.nonnegative else self.b

    def tensor(self):
        d = self.d
        if self.kind == "cp":
            t = einsum(d, self.a, self.b, "c r, i r, j r -> c i j")
            return 0.5 * (t + t.mT)
        if self.kind == "sym":
            v = self.a
            return einsum(d, v, v, "c r, i r, j r -> c i j")
        p, n = self.pos(), self.neg_or_right()
        return einsum(d, p, p, "c r, i r, j r -> c i j") - einsum(
            d, n, n, "c r, i r, j r -> c i j"
        )

    def activations(self, x):
        if self.kind == "cp":
            return (x @ self.a) * (x @ self.b)
        if self.kind == "sym":
            return (x @ self.a).square()
        return (x @ self.pos()).square() - (x @ self.neg_or_right()).square()

    def forward(self, x):
        return self.activations(x) @ self.d.T

    def decompose(self):
        if self.kind == "cp":
            plus = normalize_columns(self.a + self.b)
            minus = normalize_columns(self.a - self.b)
            sigma = self.a.norm(dim=0) * self.b.norm(dim=0) * self.d.norm(dim=0)
        elif self.kind == "sym":
            plus = normalize_columns(self.a)
            minus = torch.zeros_like(plus)
            sigma = self.a.norm(dim=0).square() * self.d.norm(dim=0)
        else:
            p, n = self.pos(), self.neg_or_right()
            plus = normalize_columns(p)
            minus = normalize_columns(n)
            sigma = (p.norm(dim=0).square() + n.norm(dim=0).square()) * self.d.norm(dim=0)
        order = sigma.argsort(descending=True)
        return plus[:, order], minus[:, order], self.d[:, order], sigma[order]

    def regularizer(self, reg: Reg):
        plus, minus, down, _ = self.decompose()
        pats = torch.cat([plus, minus], dim=1)
        loss = pats.new_tensor(0.0)
        if reg.l1:
            loss = loss + reg.l1 * pats.abs().mean()
        if reg.tv:
            loss = loss + reg.tv * total_variation(pats)
        if reg.d_l1:
            loss = loss + reg.d_l1 * down.abs().mean()
        if reg.entropy:
            loss = loss + reg.entropy * head_entropy(down)
        if reg.duplicate:
            n = normalize_columns(pats)
            gram = n.T @ n
            gram = gram - torch.eye(gram.shape[0], device=gram.device)
            loss = loss + reg.duplicate * gram.square().mean()
        if reg.border:
            loss = loss + reg.border * border_penalty(pats)
        if reg.symmetry and self.kind == "cp":
            loss = loss + reg.symmetry * (self.a - self.b).square().mean()
        return loss


def fit_variant(name, kind, rank, reg, target, test, steps, lr, seed, nonnegative=False):
    set_seed(seed)
    model = Factor(kind, rank, nonnegative=nonnegative).to(target.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        approx = model.tensor()
        recon = 1 - F.cosine_similarity(approx.flatten(), target.flatten(), dim=0)
        loss = recon + model.regularizer(reg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        plus, minus, down, sigma = model.decompose()
        pats = torch.cat([plus, minus], dim=1)
        logits = model(test.x)
        sim = F.cosine_similarity(model.tensor().flatten(), target.flatten(), dim=0)
        row = {
            "name": name,
            "kind": kind,
            "rank": rank,
            "steps": steps,
            "similarity": sim.item(),
            "test_acc": (logits.argmax(-1) == test.y).float().mean().item(),
            "pattern_gini": mean_gini(pats).item(),
            "locality_7x7": patch_locality(pats).item(),
            "class_selectivity": class_selectivity(down).item(),
            "tv": total_variation(pats).item(),
            "top_sigma_frac": (sigma[:8].sum() / sigma.sum().clamp_min(1e-8)).item(),
        }
        row["visual_score"] = (
            row["similarity"]
            + 0.5 * row["test_acc"]
            + 0.35 * row["pattern_gini"]
            + 0.25 * row["locality_7x7"]
            + 0.25 * row["class_selectivity"]
        )
    return row, model


def example_style_plot(model, out_path, title, k=10, denoise=False):
    plus, minus, down, sigma = model.decompose()
    plus, minus, down = plus.detach().cpu(), minus.detach().cpu(), down.detach().cpu()
    if denoise:
        # Visual-only denoising: preserve signs, hide the lowest-magnitude 65% pixels per component.
        for pats in (plus, minus):
            thresh = pats.abs().quantile(0.65, dim=0, keepdim=True)
            pats *= (pats.abs() >= thresh)
    k = min(k, plus.shape[1])
    fig, axes = plt.subplots(
        3,
        k,
        figsize=(1.2 * k, 4.0),
        gridspec_kw={"height_ratios": [1, 1, 0.65], "hspace": 0.12, "wspace": 0.28},
    )
    vmax = max(float(plus[:, :k].abs().max()), float(minus[:, :k].abs().max()))
    for i in range(k):
        axes[0, i].imshow(plus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, i].imshow(minus[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        colors = ["#4c78a8" if v >= 0 else "#d62728" for v in down[:, i].tolist()]
        axes[2, i].bar(range(10), down[:, i], color=colors, width=0.75)
        axes[2, i].set_xticks([])
        axes[2, i].tick_params(left=False, labelleft=False)
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
        for spine in axes[2, i].spines.values():
            spine.set_visible(False)
        axes[2, i].axhline(0, color="#dddddd", linewidth=0.6)
    axes[0, -1].set_ylabel("L+R / pos", rotation=270, labelpad=18, fontsize=11)
    axes[1, -1].set_ylabel("L-R / neg", rotation=270, labelpad=18, fontsize=11)
    axes[2, -1].set_ylabel("head", rotation=270, labelpad=18, fontsize=11)
    for row in range(3):
        axes[row, -1].yaxis.set_label_position("right")
    fig.suptitle(title, y=0.99, fontsize=13, fontstyle="italic")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--outdir", type=Path, default=Path("figures/search"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dev = device_name()
    set_seed(123)
    train = MNIST(train=True, device=dev)
    test = MNIST(train=False, device=dev)
    base = Model.from_config(epochs=args.epochs).to(dev)
    base.fit(train, test, RandomGaussianNoise(std=0.4).to(dev))
    with torch.no_grad():
        base_acc = (base(test.x).argmax(-1) == test.y).float().mean().item()
        target = interaction_tensor(base).detach()

    specs = [
        ("cp_plain", "cp", args.rank, Reg(), 0.03, 1, False),
        ("cp_soft_sym_l1tv", "cp", args.rank, Reg(symmetry=0.05, l1=2e-4, tv=1e-3), 0.03, 2, False),
        ("split_plain", "split", args.rank, Reg(), 0.035, 3, False),
        ("split_l1_tv", "split", args.rank, Reg(l1=2e-4, tv=2e-3, d_l1=2e-4), 0.035, 4, False),
        ("split_strong_tv", "split", args.rank, Reg(l1=2e-4, tv=8e-3, d_l1=2e-4), 0.035, 5, False),
        ("split_entropy_head", "split", args.rank, Reg(l1=2e-4, tv=2e-3, entropy=2e-3), 0.035, 6, False),
        ("split_border_center", "split", args.rank, Reg(l1=2e-4, tv=2e-3, border=2e-2), 0.035, 7, False),
        ("split_diverse", "split", args.rank, Reg(l1=2e-4, tv=2e-3, duplicate=1e-2), 0.035, 8, False),
        ("sym_l1_tv", "sym", args.rank, Reg(l1=2e-4, tv=2e-3, d_l1=2e-4), 0.04, 9, False),
        ("nonnegative_split", "split", args.rank + 16, Reg(l1=3e-4, tv=2e-3, entropy=1e-3), 0.03, 10, True),
    ]

    rows = []
    models = {}
    for spec in specs:
        name, kind, rank, reg, lr, seed, nonneg = spec
        row, model = fit_variant(
            name,
            kind,
            rank,
            reg,
            target,
            test,
            steps=args.steps,
            lr=lr,
            seed=seed,
            nonnegative=nonneg,
        )
        rows.append(row)
        models[name] = model
        print(row)

    rows = sorted(rows, key=lambda r: r["visual_score"], reverse=True)
    with (args.outdir / "variant_search_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    example_style_plot(
        models[best["name"]],
        args.outdir / "example_style_best_raw.png",
        f"Our best search variant: {best['name']} (raw)",
        denoise=False,
    )
    example_style_plot(
        models[best["name"]],
        args.outdir / "example_style_best_denoised.png",
        f"Our best search variant: {best['name']} (denoised view)",
        denoise=True,
    )
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={dev}\\n")
        f.write(f"base_acc={base_acc:.6f}\\n")
        f.write(f"best={best}\\n")


if __name__ == "__main__":
    main()
