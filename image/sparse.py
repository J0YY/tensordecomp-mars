from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


@dataclass
class Config:
    rank: int = 64
    n_inputs: int = 28 * 28
    n_outputs: int = 10
    scale: float = 0.02


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.left = nn.Parameter(config.scale * torch.randn(config.n_inputs, config.rank))
        self.right = nn.Parameter(config.scale * torch.randn(config.n_inputs, config.rank))
        self.down_param = nn.Parameter(config.scale * torch.randn(config.n_outputs, config.rank))

    @classmethod
    def from_config(cls, **kwargs):
        return cls(Config(**kwargs))

    def tensor(self):
        t = einsum(self.down_param, self.left, self.right, "c r, i r, j r -> c i j")
        return 0.5 * (t + t.mT)

    def _target_tensor(self, model):
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

    def similarity(self, model):
        target = self._target_tensor(model)
        return F.cosine_similarity(self.tensor().flatten(), target.flatten(), dim=0)

    def component_activations(self, x):
        return (x @ self.left) * (x @ self.right)

    def forward(self, x):
        return self.component_activations(x) @ self.down_param.T

    @torch.no_grad()
    def decompose(self):
        plus_raw = self.left + self.right
        minus_raw = self.left - self.right
        plus = plus_raw / plus_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        minus = minus_raw / minus_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        sigma = self.left.norm(dim=0) * self.right.norm(dim=0) * self.down_param.norm(dim=0)
        order = sigma.argsort(descending=True)
        return plus[:, order], minus[:, order], self.down_param[:, order], sigma[order]

