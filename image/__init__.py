from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from einops import einsum
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


@dataclass
class Config:
    n_inputs: int = 28 * 28
    n_embed: int = 96
    n_hidden: int = 256
    n_outputs: int = 10
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-4


class MNIST:
    def __init__(self, train: bool = True, device: str | torch.device = "cpu", root: str = "data"):
        ds = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
        x = ds.data.float().reshape(-1, 28 * 28) / 255.0
        y = ds.targets.long()
        self.x = x.to(device)
        self.y = y.to(device)

    def loader(self, batch_size: int = 512, shuffle: bool = True):
        return DataLoader(TensorDataset(self.x, self.y), batch_size=batch_size, shuffle=shuffle)


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.w_e = nn.Parameter(torch.randn(config.n_embed, config.n_inputs) / config.n_inputs**0.5)
        self.w_lr_param = nn.Parameter(torch.randn(2, config.n_hidden, config.n_embed) / config.n_embed**0.5)
        self.w_u = nn.Parameter(torch.randn(config.n_outputs, config.n_hidden) / config.n_hidden**0.5)
        self.bias = nn.Parameter(torch.zeros(config.n_outputs))

    @classmethod
    def from_config(cls, **kwargs):
        cfg = Config(**kwargs)
        return cls(cfg)

    @property
    def w_lr(self):
        return self.w_lr_param.unsqueeze(0)

    @property
    def w_l(self):
        return self.w_lr_param[0].unsqueeze(0)

    @property
    def w_r(self):
        return self.w_lr_param[1].unsqueeze(0)

    def forward(self, x):
        h = x @ self.w_e.T
        l = h @ self.w_lr_param[0].T
        r = h @ self.w_lr_param[1].T
        return (l * r) @ self.w_u.T + self.bias

    def fit(self, train: MNIST, test: MNIST | None = None, augmentation=None):
        cfg = self.config
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        metrics = []
        for epoch in range(cfg.epochs):
            self.train()
            total_loss = 0.0
            total = 0
            for x, y in train.loader(cfg.batch_size, shuffle=True):
                xb = augmentation(x.reshape(-1, 1, 28, 28)).reshape(-1, 28 * 28) if augmentation else x
                logits = self(xb)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.shape[0]
                total += x.shape[0]
            row = {"epoch": epoch, "train_loss": total_loss / max(1, total)}
            if test is not None:
                row["test_acc"] = self.accuracy(test)
            metrics.append(row)
            if test is not None:
                print(f"epoch {epoch + 1:02d}/{cfg.epochs} loss={row['train_loss']:.4f} acc={row['test_acc']:.4f}")
            else:
                print(f"epoch {epoch + 1:02d}/{cfg.epochs} loss={row['train_loss']:.4f}")
        self.eval()
        return metrics

    @torch.no_grad()
    def accuracy(self, data: MNIST, batch_size: int = 2048):
        correct = 0
        total = 0
        for x, y in data.loader(batch_size, shuffle=False):
            correct += (self(x).argmax(-1) == y).sum().item()
            total += y.numel()
        return correct / max(1, total)

    @torch.no_grad()
    def decompose(self):
        l, r = self.w_lr_param
        b = einsum(self.w_u, l, r, "cls out, out emb1, out emb2 -> cls emb1 emb2")
        b = 0.5 * (b + b.mT)
        vals, vecs = torch.linalg.eigh(b)
        vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
        return vals, vecs


def plot_eigenspectrum(model: Model, digit: int = 0, k: int = 8):
    vals, vecs = model.decompose()
    order = vals[digit].abs().argsort(descending=True)[:k]
    fig = px.imshow(
        vecs[digit, order].reshape(-1, 28, 28).detach().cpu(),
        facet_col=0,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    return fig


def plot_explanation(model: Model, x):
    vals, vecs = model.decompose()
    logits = model(x.unsqueeze(0)).squeeze(0)
    cls = logits.argsort(descending=True)[:3]
    rows = []
    for c in cls.tolist():
        acts = (vecs[c] @ x).square() * vals[c]
        rows.append(acts.detach().cpu())
    return px.imshow(torch.stack(rows), color_continuous_midpoint=0, color_continuous_scale="RdBu")

