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
    img = images(x)
    for _ in range(passes):
        img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    return flatten_images(img)


def total_variation(patterns):
    img = images(patterns)
    return (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean() + (
        img[:, :, :, 1:] - img[:, :, :, :-1]
    ).abs().mean()


def laplacian_loss(patterns):
    img = images(patterns)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=patterns.device)
    kernel = kernel.reshape(1, 1, 3, 3)
    return F.conv2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), kernel).abs().mean()


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


def grid(device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 28, device=device),
        torch.linspace(-1, 1, 28, device=device),
        indexing="ij",
    )
    return yy, xx


def line_template(yy, xx, cy, cx, theta, length=0.65, width=0.06):
    dy, dx = yy - cy, xx - cx
    par = dx * math.cos(theta) + dy * math.sin(theta)
    perp = -dx * math.sin(theta) + dy * math.cos(theta)
    return torch.exp(-(perp.square() / (2 * width**2) + par.square() / (2 * length**2)))


def arc_template(yy, xx, cy, cx, radius=0.35, theta0=0.0, span=math.pi, width=0.055):
    dy, dx = yy - cy, xx - cx
    rr = torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6)
    ang = torch.atan2(dy, dx)
    # Circular distance to the arc center angle.
    dang = torch.atan2(torch.sin(ang - theta0), torch.cos(ang - theta0)).abs()
    radial = torch.exp(-((rr - radius).square()) / (2 * width**2))
    angular = torch.exp(-(dang.square()) / (2 * (span / 2.4) ** 2))
    return radial * angular


def blob_template(yy, xx, cy, cx, sy=0.12, sx=0.12):
    return torch.exp(-(((yy - cy) / sy).square() + ((xx - cx) / sx).square()) / 2)


def make_template_bank(rank, device):
    yy, xx = grid(device)
    templates = []
    # Stroke-like bars at digit-relevant locations.
    centers_y = [-0.55, -0.25, 0.05, 0.35, 0.62]
    centers_x = [-0.45, -0.18, 0.12, 0.42]
    angles = [0, math.pi / 2, math.pi / 4, -math.pi / 4]
    for cy in centers_y:
        for cx in centers_x:
            for theta in angles:
                templates.append(line_template(yy, xx, cy, cx, theta))
    # Longer horizontal/vertical bars that often distinguish 1, 4, 7.
    for cy in [-0.62, -0.2, 0.18, 0.55]:
        templates.append(line_template(yy, xx, cy, 0.0, 0, length=0.9, width=0.055))
    for cx in [-0.42, -0.1, 0.18, 0.45]:
        templates.append(line_template(yy, xx, 0.0, cx, math.pi / 2, length=0.95, width=0.055))
    # Arcs and loops for 0, 3, 5, 6, 8, 9.
    for cy in [-0.28, 0.18]:
        for cx in [-0.22, 0.22]:
            for theta0 in [0, math.pi / 2, math.pi, -math.pi / 2]:
                templates.append(arc_template(yy, xx, cy, cx, radius=0.34, theta0=theta0, span=math.pi))
            templates.append(arc_template(yy, xx, cy, cx, radius=0.34, theta0=0, span=2 * math.pi))
    # Small endpoints/corners.
    for cy in [-0.62, -0.25, 0.15, 0.55]:
        for cx in [-0.48, 0.0, 0.48]:
            templates.append(blob_template(yy, xx, cy, cx))

    bank = torch.stack(templates, dim=0).reshape(len(templates), -1).T
    bank = bank - bank.mean(dim=0, keepdim=True)
    bank = normalize_columns(bank)
    # Prefer diverse early columns, then repeat if rank is larger.
    if bank.shape[1] < rank:
        repeats = math.ceil(rank / bank.shape[1])
        bank = bank.repeat(1, repeats)
    return bank[:, :rank]


@dataclass
class Reg:
    tensor: float = 1.0
    distill: float = 0.1
    anchor: float = 0.03
    residual_l2: float = 0.01
    raw_tv: float = 0.02
    raw_lap: float = 0.005
    d_l1: float = 0.0002


class StrokeTemplateFactor(nn.Module):
    def __init__(
        self,
        kind: str,
        rank: int,
        topk_head: int | None,
        residual_scale: float,
        device: str,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.kind = kind
        self.rank = rank
        self.topk_head = topk_head
        self.residual_scale = residual_scale
        bank = make_template_bank(rank + seed_offset + 8, device)
        anchor_a = bank[:, seed_offset : seed_offset + rank]
        anchor_b = bank[:, seed_offset + 3 : seed_offset + 3 + rank]
        if anchor_b.shape[1] < rank:
            anchor_b = bank[:, :rank]
        self.register_buffer("anchor_a", anchor_a)
        self.register_buffer("anchor_b", anchor_b)
        self.delta_a = nn.Parameter(torch.zeros(784, rank, device=device))
        self.delta_b = nn.Parameter(torch.zeros(784, rank, device=device))
        self.log_amp_a = nn.Parameter(torch.zeros(rank, device=device))
        self.log_amp_b = nn.Parameter(torch.zeros(rank, device=device))
        self.raw_d = nn.Parameter(0.03 * torch.randn(10, rank, device=device))

    def factors(self):
        da = avg_smooth(self.delta_a, 1)
        db = avg_smooth(self.delta_b, 1)
        a = (self.anchor_a + self.residual_scale * da) * self.log_amp_a.exp().unsqueeze(0)
        b = (self.anchor_b + self.residual_scale * db) * self.log_amp_b.exp().unsqueeze(0)
        return a, b

    def down(self):
        if self.topk_head is None:
            return self.raw_d
        idx = self.raw_d.detach().abs().topk(self.topk_head, dim=0).indices
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

    def decompose(self, visual=False):
        a, b = self.factors()
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
            score = sigma / sigma.max().clamp_min(1e-8) + 0.8 * loc + 0.6 * gin + 0.7 * head
            order = score.argsort(descending=True)
        else:
            order = sigma.argsort(descending=True)
        return plus[:, order], minus[:, order], down[:, order], sigma[order]

    def regularizer(self, reg: Reg):
        a, b = self.factors()
        anchor = F.mse_loss(normalize_columns(a), self.anchor_a) + F.mse_loss(normalize_columns(b), self.anchor_b)
        residual = self.delta_a.square().mean() + self.delta_b.square().mean()
        smooth = total_variation(torch.cat([a, b], dim=1)) + 0.5 * laplacian_loss(torch.cat([a, b], dim=1))
        return reg.anchor * anchor + reg.residual_l2 * residual + reg.raw_tv * smooth + reg.d_l1 * self.down().abs().mean()


def fit_variant(spec, target, train_x, train_logits, test, steps):
    set_seed(spec["seed"])
    model = StrokeTemplateFactor(
        kind=spec["kind"],
        rank=spec["rank"],
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
        approx = model.tensor()
        tensor_loss = 1 - F.cosine_similarity(approx.flatten(), target.flatten(), dim=0)
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
        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            "rank": spec["rank"],
            "topk": spec["topk"],
            "residual_scale": spec["residual_scale"],
            "similarity": sim.item(),
            "test_acc": (logits.argmax(-1) == test.y).float().mean().item(),
            "pattern_gini": mean_gini(display).item(),
            "locality_7x7": patch_locality(display).item(),
            "class_selectivity": class_selectivity(down).item(),
            "head_top1_frac": head_top1_frac(down).item(),
            "display_tv": total_variation(display).item(),
            "top_sigma_frac": (sigma[:8].sum() / sigma.sum().clamp_min(1e-8)).item(),
        }
        row["submission_score"] = (
            0.35 * row["similarity"]
            + 0.35 * row["test_acc"]
            + 0.55 * row["pattern_gini"]
            + 0.75 * row["locality_7x7"]
            + 0.70 * row["class_selectivity"]
            + 0.20 * row["head_top1_frac"]
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
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--rank", type=int, default=72)
    parser.add_argument("--outdir", type=Path, default=Path("figures/stroke_templates"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = device_name()

    set_seed(777)
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
        dict(name="stroke_cp_top1_resid03", kind="cp", rank=args.rank, topk=1, residual_scale=0.30, seed=1, seed_offset=0, lr=0.045, reg=Reg(distill=0.20, anchor=0.05, residual_l2=0.015)),
        dict(name="stroke_cp_top2_resid03", kind="cp", rank=args.rank, topk=2, residual_scale=0.30, seed=2, seed_offset=5, lr=0.045, reg=Reg(distill=0.18, anchor=0.05, residual_l2=0.015)),
        dict(name="stroke_cp_top3_resid04", kind="cp", rank=args.rank, topk=3, residual_scale=0.40, seed=3, seed_offset=10, lr=0.045, reg=Reg(distill=0.16, anchor=0.04, residual_l2=0.012)),
        dict(name="stroke_cp_top2_resid06", kind="cp", rank=args.rank, topk=2, residual_scale=0.60, seed=4, seed_offset=15, lr=0.04, reg=Reg(distill=0.16, anchor=0.03, residual_l2=0.010)),
        dict(name="stroke_split_top1_resid03", kind="split", rank=args.rank, topk=1, residual_scale=0.30, seed=5, seed_offset=20, lr=0.045, reg=Reg(distill=0.20, anchor=0.05, residual_l2=0.015)),
        dict(name="stroke_split_top2_resid03", kind="split", rank=args.rank, topk=2, residual_scale=0.30, seed=6, seed_offset=25, lr=0.045, reg=Reg(distill=0.18, anchor=0.05, residual_l2=0.015)),
        dict(name="stroke_split_top3_resid04", kind="split", rank=args.rank, topk=3, residual_scale=0.40, seed=7, seed_offset=30, lr=0.045, reg=Reg(distill=0.16, anchor=0.04, residual_l2=0.012)),
        dict(name="stroke_split_top2_resid06", kind="split", rank=args.rank, topk=2, residual_scale=0.60, seed=8, seed_offset=35, lr=0.04, reg=Reg(distill=0.16, anchor=0.03, residual_l2=0.010)),
    ]

    rows, models = [], {}
    for spec in specs:
        row, model = fit_variant(spec, target, train_x, train_logits, test, args.steps)
        print(row)
        rows.append(row)
        models[row["name"]] = model

    rows = sorted(rows, key=lambda r: r["submission_score"], reverse=True)
    with (args.outdir / "stroke_template_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_model = models[best["name"]]
    example_style_plot(best_model, args.outdir / "best_stroke_template_raw.png", f"Stroke-template dictionary: {best['name']} (raw)")
    example_style_plot(best_model, args.outdir / "best_stroke_template_denoised.png", f"Stroke-template dictionary: {best['name']} (top pixels)", denoise_quantile=0.70)
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={dev}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"best={best}\n")


if __name__ == "__main__":
    main()

