from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomGaussianNoise

from image import MNIST, Model
from scripts.search_stroke_mask_residual import Reg, StrokeMaskResidual, example_style_plot
from scripts.search_stroke_templates import (
    class_selectivity,
    gini_1d,
    head_top1_frac,
    interaction_tensor,
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
        act_idx = torch.randperm(train.x.shape[0], device=train.x.device)[:4096]
        act_x, act_y = train.x[act_idx], train.y[act_idx]
        base_acc = (base(test.x).argmax(-1) == test.y).float().mean().item()
    return train, test, target, train_x, train_logits, act_x, act_y, base_acc


class AdditiveBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def tensor(self):
        return sum(branch.main_tensor() for branch in self.branches)

    def main_tensor(self):
        return self.tensor()

    def forward(self, x):
        return sum(branch(x) for branch in self.branches)

    def regularizer(self, reg: Reg):
        return sum(branch.regularizer(reg) for branch in self.branches)

    def down(self):
        return torch.cat([branch.down() for branch in self.branches], dim=1)

    def component_activations(self, x):
        return torch.cat([branch_activations(branch, x) for branch in self.branches], dim=1)

    def decompose(self, visual=True):
        parts = [branch.decompose(visual=False) for branch in self.branches]
        plus = torch.cat([p[0] for p in parts], dim=1)
        minus = torch.cat([p[1] for p in parts], dim=1)
        down = torch.cat([p[2] for p in parts], dim=1)
        sigma = torch.cat([p[3] for p in parts], dim=0)
        if visual:
            loc = torch.stack([patch_locality(torch.stack([plus[:, i], minus[:, i]], dim=1)) for i in range(plus.shape[1])])
            gin = torch.stack([gini_1d(plus[:, i]) + gini_1d(minus[:, i]) for i in range(plus.shape[1])])
            head = down.abs().max(dim=0).values / down.abs().sum(dim=0).clamp_min(1e-8)
            score = sigma / sigma.max().clamp_min(1e-8) + 0.85 * loc + 0.65 * gin + 0.8 * head
            order = score.argsort(descending=True)
        else:
            order = sigma.argsort(descending=True)
        return plus[:, order], minus[:, order], down[:, order], sigma[order]


def branch_activations(model, x):
    a, b, _, _ = model.main_factors()
    if model.kind == "cp":
        return (x @ a) * (x @ b)
    return (x @ a).square() - (x @ b).square()


def component_activations(model, x):
    if hasattr(model, "component_activations"):
        return model.component_activations(x)
    return branch_activations(model, x)


def set_data_anchors(model, bank, shift=5):
    bank = bank.to(model.anchor_a.device)
    if bank.shape[1] < model.rank + shift:
        repeats = math.ceil((model.rank + shift) / bank.shape[1])
        bank = bank.repeat(1, repeats)
    with torch.no_grad():
        model.anchor_a.copy_(bank[:, : model.rank])
        model.anchor_b.copy_(bank[:, shift : shift + model.rank])
    return model


@torch.no_grad()
def make_data_bank(train_x, train_y, rank):
    pieces = []
    means = []
    for digit in range(10):
        mean = train_x[train_y == digit].mean(dim=0)
        means.append(mean)
        pieces.append(mean)
    means = torch.stack(means)
    for i in range(10):
        for j in range(i + 1, 10):
            pieces.append(means[i] - means[j])
            pieces.append(means[j] - means[i])
    sample = train_x[torch.randperm(train_x.shape[0], device=train_x.device)[:6000]]
    centered = sample - sample.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=min(32, centered.shape[0] - 1, centered.shape[1] - 1), center=False)
    pieces.extend(v.T)
    bank = torch.stack(pieces, dim=1)
    bank = bank - bank.mean(dim=0, keepdim=True)
    bank = normalize_columns(bank)
    if bank.shape[1] < rank:
        bank = bank.repeat(1, math.ceil(rank / bank.shape[1]))
    return bank[:, :rank]


def activation_consistency_loss(model, x, y):
    acts = component_activations(model, x).square()
    down = model.down()
    heads = down.detach().abs().argmax(dim=0)
    same = y[:, None] == heads[None, :]
    total = acts.mean(dim=0).clamp_min(1e-8)
    off = (acts * (~same)).mean(dim=0)
    return (off / total).mean()


def fit_model(
    model,
    target,
    train_x,
    train_logits,
    test,
    steps,
    reg,
    lr=0.04,
    act_x=None,
    act_y=None,
    activation_weight=0.0,
    residual_schedule=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    logit_scale = train_logits.std().detach().clamp_min(1e-4)
    for step in range(steps):
        if residual_schedule is not None:
            start, end = residual_schedule
            reg.residual_branch = start + (end - start) * (step / max(1, steps - 1))
        tensor_loss = 1 - F.cosine_similarity(model.tensor().flatten(), target.flatten(), dim=0)
        distill = F.mse_loss(model(train_x) / logit_scale, train_logits / logit_scale)
        loss = reg.tensor * tensor_loss + reg.distill * distill + model.regularizer(reg)
        if activation_weight and act_x is not None and act_y is not None:
            loss = loss + activation_weight * activation_consistency_loss(model, act_x, act_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return evaluate(model, target, test)


@torch.no_grad()
def evaluate(model, target, test):
    plus, minus, down, sigma = model.decompose(visual=True)
    display = torch.cat([plus, minus], dim=1)
    tensor = model.tensor()
    main_tensor = model.main_tensor() if hasattr(model, "main_tensor") else tensor
    sim = F.cosine_similarity(tensor.flatten(), target.flatten(), dim=0).item()
    main_sim = F.cosine_similarity(main_tensor.flatten(), target.flatten(), dim=0).item()
    return {
        "similarity": sim,
        "main_similarity": main_sim,
        "test_acc": (model(test.x).argmax(-1) == test.y).float().mean().item(),
        "pattern_gini": mean_gini(display).item(),
        "locality_7x7": patch_locality(display).item(),
        "class_selectivity": class_selectivity(down).item(),
        "head_top1_frac": head_top1_frac(down).item(),
        "display_tv": total_variation(display).item(),
        "top_sigma_frac": (sigma[:8].sum() / sigma.sum().clamp_min(1e-8)).item(),
        "residual_fraction": 1.0 - main_sim / max(sim, 1e-8),
    }


def make_branch(kind, rank, residual_rank, topk, residual_scale, seed, seed_offset, device, data_bank=None):
    set_seed(seed)
    branch = StrokeMaskResidual(
        kind=kind,
        rank=rank,
        residual_rank=residual_rank,
        topk_head=topk,
        residual_scale=residual_scale,
        device=device,
        seed_offset=seed_offset,
    )
    if data_bank is not None:
        set_data_anchors(branch, data_bank, shift=5 + seed_offset % 7)
    return branch


def plot_summary(rows, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        ax.scatter(row["residual_fraction"], row["similarity"], s=90, alpha=0.85)
        ax.annotate(row["name"], (row["residual_fraction"], row["similarity"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("residual fraction")
    ax.set_ylabel("tensor cosine")
    ax.set_title("Next-step interpretable decomposition attempts", fontstyle="italic")
    ax.grid(True, color="#e8e8e8")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_boosted(target, train_x, train_logits, test, device, steps):
    first = make_branch("cp", 64, 0, 1, 0.30, 501, 0, device)
    reg = Reg(distill=0.18, residual_branch=0.0)
    row1 = fit_model(first, target, train_x, train_logits, test, steps, reg)
    with torch.no_grad():
        residual_target = target - first.tensor()
        residual_logits = train_logits - first(train_x)
    second = make_branch("cp", 64, 0, 1, 0.38, 502, 18, device)
    row2 = fit_model(second, residual_target, train_x, residual_logits, test, steps, Reg(distill=0.14, residual_branch=0.0))
    combo = AdditiveBranches([first, second])
    row = evaluate(combo, target, test)
    row.update(name="boosted_two_stage_residual", note=f"stage1={row1['similarity']:.4f}; stage2_residual={row2['similarity']:.4f}")
    return row, combo


def run_all(args):
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    device = device_name()
    train, test, target, train_x, train_logits, act_x, act_y, base_acc = train_base(device, args.epochs)
    data_bank = make_data_bank(train.x, train.y, 96)

    rows, models = [], {}

    row, model = run_boosted(target, train_x, train_logits, test, device, args.steps)
    rows.append(row)
    models[row["name"]] = model
    print(row)

    annealed = make_branch("cp", 64, 48, 1, 0.35, 511, 0, device)
    row = fit_model(
        annealed,
        target,
        train_x,
        train_logits,
        test,
        args.steps + 80,
        Reg(distill=0.18, residual_branch=0.04),
        residual_schedule=(0.04, 0.42),
    )
    row.update(name="residual_anneal_res48")
    rows.append(row)
    models[row["name"]] = annealed
    print(row)

    primary = make_branch("cp", 64, 0, 1, 0.30, 521, 0, device)
    secondary = make_branch("cp", 64, 0, 1, 0.48, 522, 28, device)
    two_bank = AdditiveBranches([primary, secondary])
    row = fit_model(two_bank, target, train_x, train_logits, test, args.steps + 80, Reg(distill=0.18, residual_branch=0.0))
    row.update(name="two_bank_interpretable")
    rows.append(row)
    models[row["name"]] = two_bank
    print(row)

    act_model = make_branch("cp", 64, 16, 1, 0.35, 531, 12, device)
    row = fit_model(
        act_model,
        target,
        train_x,
        train_logits,
        test,
        args.steps,
        Reg(distill=0.20, residual_branch=0.20),
        act_x=act_x,
        act_y=act_y,
        activation_weight=0.08,
    )
    row.update(name="activation_consistency")
    rows.append(row)
    models[row["name"]] = act_model
    print(row)

    data_model = make_branch("cp", 64, 16, 1, 0.35, 541, 0, device, data_bank=data_bank)
    row = fit_model(data_model, target, train_x, train_logits, test, args.steps, Reg(distill=0.20, residual_branch=0.18))
    row.update(name="data_derived_priors")
    rows.append(row)
    models[row["name"]] = data_model
    print(row)

    rows.sort(key=lambda r: (r["similarity"], r["main_similarity"]), reverse=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (outdir / "interpretable_next_steps_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "summary.txt").open("w") as f:
        f.write(f"device={device}\nbase_acc={base_acc:.6f}\n")
        for row in rows:
            f.write(f"{row}\n")

    plot_summary(rows, outdir / "interpretable_next_steps_frontier.png")
    for row in rows[:3]:
        example_style_plot(
            models[row["name"]],
            outdir / f"{row['name']}_denoised.png",
            f"Next-step experiment: {row['name']} (top pixels)",
            denoise_quantile=0.70,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--outdir", type=Path, default=Path("figures/interpretable_next_steps"))
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
