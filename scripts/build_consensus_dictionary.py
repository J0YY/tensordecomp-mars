from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from scripts.search_stroke_templates import gini_1d, normalize_columns, patch_locality
from scripts.validate_best_combo import best_spec, train_base
from scripts.search_stroke_mask_residual import fit_variant


@dataclass
class Component:
    seed: int
    idx: int
    digit: int
    sign: int
    score: float
    plus: torch.Tensor
    minus: torch.Tensor
    feature: torch.Tensor


@dataclass
class Cluster:
    digit: int
    sign: int
    members: list[Component] = field(default_factory=list)
    orientations: list[float] = field(default_factory=list)

    def centroid(self):
        feats = [orient * c.feature for orient, c in zip(self.orientations, self.members)]
        return torch.stack(feats, dim=1).mean(dim=1)

    def add(self, comp: Component, orient: float):
        self.members.append(comp)
        self.orientations.append(orient)

    def seeds(self):
        return {c.seed for c in self.members}

    def mean_score(self):
        return sum(c.score for c in self.members) / max(1, len(self.members))

    def mean_similarity(self):
        if len(self.members) < 2:
            return 1.0
        feats = torch.stack([orient * c.feature for orient, c in zip(self.orientations, self.members)], dim=1)
        sims = feats.T @ feats
        vals = []
        for i in range(sims.shape[0]):
            for j in range(i + 1, sims.shape[1]):
                vals.append(float(sims[i, j].item()))
        return sum(vals) / max(1, len(vals))

    def images(self):
        plus = torch.stack([orient * c.plus for orient, c in zip(self.orientations, self.members)], dim=1).mean(dim=1)
        minus = torch.stack([orient * c.minus for orient, c in zip(self.orientations, self.members)], dim=1).mean(dim=1)
        return plus, minus


@torch.no_grad()
def collect_components(models, seeds, per_seed=28):
    comps = []
    for seed in seeds:
        plus, minus, down, sigma = models[seed].decompose(visual=True)
        display = torch.cat([plus, minus], dim=1)
        loc = torch.stack([patch_locality(torch.stack([plus[:, i], minus[:, i]], dim=1)) for i in range(plus.shape[1])])
        gin = torch.stack([gini_1d(plus[:, i]) + gini_1d(minus[:, i]) for i in range(plus.shape[1])])
        head = down.abs().max(dim=0).values / down.abs().sum(dim=0).clamp_min(1e-8)
        strength = sigma / sigma.max().clamp_min(1e-8)
        score = 0.35 * strength + 0.35 * loc + 0.20 * gin + 0.45 * head
        for idx in score.argsort(descending=True)[:per_seed].tolist():
            digit = int(down[:, idx].abs().argmax().item())
            sign = 1 if float(down[digit, idx].item()) >= 0 else -1
            feat = normalize_columns(torch.cat([plus[:, idx], minus[:, idx]], dim=0).reshape(-1, 1)).flatten()
            comps.append(
                Component(
                    seed=seed,
                    idx=idx,
                    digit=digit,
                    sign=sign,
                    score=float(score[idx].item()),
                    plus=plus[:, idx].detach(),
                    minus=minus[:, idx].detach(),
                    feature=feat.detach(),
                )
            )
    return sorted(comps, key=lambda c: c.score, reverse=True)


@torch.no_grad()
def cluster_components(comps, min_abs_cos=0.23):
    clusters: list[Cluster] = []
    for comp in comps:
        best_cluster = None
        best_orient = 1.0
        best_sim = -1.0
        for cluster in clusters:
            if cluster.digit != comp.digit:
                continue
            centroid = normalize_columns(cluster.centroid().reshape(-1, 1)).flatten()
            sim = float((centroid @ comp.feature).item())
            orient = 1.0 if sim >= 0 else -1.0
            abs_sim = abs(sim)
            if abs_sim > best_sim:
                best_sim = abs_sim
                best_cluster = cluster
                best_orient = orient
        if best_cluster is not None and best_sim >= min_abs_cos:
            best_cluster.add(comp, best_orient)
        else:
            cluster = Cluster(digit=comp.digit, sign=comp.sign)
            cluster.add(comp, 1.0)
            clusters.append(cluster)
    clusters.sort(key=lambda c: (len(c.seeds()), c.mean_similarity(), c.mean_score()), reverse=True)
    return clusters


def select_clusters(clusters, max_clusters=12, max_per_digit=2):
    chosen = []
    digit_counts: dict[int, int] = {}
    for cluster in clusters:
        if len(cluster.seeds()) < 2:
            continue
        if digit_counts.get(cluster.digit, 0) >= max_per_digit:
            continue
        chosen.append(cluster)
        digit_counts[cluster.digit] = digit_counts.get(cluster.digit, 0) + 1
        if len(chosen) == max_clusters:
            break
    if len(chosen) < max_clusters:
        for cluster in clusters:
            if cluster in chosen:
                continue
            chosen.append(cluster)
            if len(chosen) == max_clusters:
                break
    return chosen


def denoise_columns(images, quantile=0.68):
    stacked = torch.stack(images, dim=1).clone()
    thresh = stacked.abs().quantile(quantile, dim=0, keepdim=True)
    stacked *= stacked.abs() >= thresh
    return [stacked[:, i] for i in range(stacked.shape[1])]


def plot_consensus(clusters, out_path, max_clusters=12, denoise_quantile=0.68):
    chosen = select_clusters(clusters, max_clusters=max_clusters)
    fig, axes = plt.subplots(3, len(chosen), figsize=(1.25 * len(chosen), 3.4))
    if len(chosen) == 1:
        axes = axes[:, None]
    images = []
    for cluster in chosen:
        images.extend(cluster.images())
    if denoise_quantile is not None:
        images = denoise_columns(images, denoise_quantile)
    vmax = max(float(torch.stack(images, dim=1).abs().max()), 1e-6)
    for col, cluster in enumerate(chosen):
        plus, minus = images[2 * col], images[2 * col + 1]
        axes[0, col].imshow(plus.cpu().reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, col].imshow(minus.cpu().reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2, col].bar([0, 1], [len(cluster.seeds()) / 5, cluster.mean_similarity()], color=["#4C78A8", "#E45756"])
        axes[2, col].set_ylim(0, 1.0)
        axes[0, col].set_title(
            f"{'+' if cluster.sign > 0 else '-'}{cluster.digit}\n{len(cluster.seeds())} seeds",
            fontsize=9,
        )
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, -1].set_ylabel("L+R / pos", rotation=270, labelpad=18)
    axes[1, -1].set_ylabel("L-R / neg", rotation=270, labelpad=18)
    axes[2, -1].set_ylabel("support/sim", rotation=270, labelpad=18)
    fig.suptitle("Consensus dictionary: recurring components across seeds", y=1.02, fontsize=14, fontstyle="italic")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--per-seed", type=int, default=28)
    parser.add_argument("--min-cos", type=float, default=0.23)
    parser.add_argument("--outdir", type=Path, default=Path("figures/consensus_dictionary"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, test, base, target, train_x, train_logits, base_acc = train_base(device, args.epochs)
    rows, models = [], {}
    for seed in args.seeds:
        row, model = fit_variant(best_spec(seed=seed, rank=args.rank, steps=args.steps), target, train_x, train_logits, test, args.steps)
        row["seed"] = seed
        print(row)
        rows.append(row)
        models[seed] = model

    comps = collect_components(models, args.seeds, per_seed=args.per_seed)
    clusters = cluster_components(comps, min_abs_cos=args.min_cos)
    chosen = plot_consensus(clusters, args.outdir / "consensus_dictionary.png")

    with (args.outdir / "consensus_clusters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "digit", "sign", "members", "seeds", "mean_similarity", "mean_score", "member_ids"],
        )
        writer.writeheader()
        for rank, cluster in enumerate(clusters, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "digit": cluster.digit,
                    "sign": "+" if cluster.sign > 0 else "-",
                    "members": len(cluster.members),
                    "seeds": len(cluster.seeds()),
                    "mean_similarity": cluster.mean_similarity(),
                    "mean_score": cluster.mean_score(),
                    "member_ids": " ".join(f"s{c.seed}:c{c.idx}" for c in cluster.members),
                }
            )
    with (args.outdir / "summary.txt").open("w") as f:
        f.write(f"device={device}\n")
        f.write(f"base_acc={base_acc:.6f}\n")
        f.write(f"seeds={args.seeds}\n")
        f.write(f"components={len(comps)}\n")
        f.write(f"clusters={len(clusters)}\n")
        f.write(f"chosen={len(chosen)}\n")
        f.write(f"chosen_mean_seed_support={sum(len(c.seeds()) for c in chosen) / max(1, len(chosen)):.6f}\n")
        f.write(f"chosen_mean_similarity={sum(c.mean_similarity() for c in chosen) / max(1, len(chosen)):.6f}\n")


if __name__ == "__main__":
    main()
