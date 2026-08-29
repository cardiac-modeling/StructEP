"""2D ligand branch for Tri-Channel PoseMIL.

Morgan fingerprint + RDKit/Mordred 2D descriptors -> 2-layer MLP -> z_2d.
This branch is the dependency-light "safety mat" path: it lets the model
fall back on cheminformatics features when docking poses are noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Ligand2DConfig:
    morgan_bits: int = 2048
    descriptor_dim: int = 0
    hidden_dim_1: int = 1024
    hidden_dim_2: int = 512
    out_dim: int = 128
    dropout: float = 0.1
    # LayerNorm by default: BatchNorm1d crashes when a batch has size 1 (happens
    # when skip_bad_samples=True filters all but one sample). LN doesn't depend
    # on batch dim and is the standard choice in modern MLPs.
    use_batchnorm: bool = False


class Ligand2DBranch(nn.Module):
    def __init__(self, cfg: Ligand2DConfig) -> None:
        super().__init__()
        in_dim = int(cfg.morgan_bits) + int(cfg.descriptor_dim)
        if in_dim <= 0:
            raise ValueError("Ligand2DBranch requires at least one of morgan_bits / descriptor_dim > 0")
        self.cfg = cfg
        norm_cls = nn.BatchNorm1d if cfg.use_batchnorm else nn.LayerNorm
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim_1),
            norm_cls(cfg.hidden_dim_1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2),
            norm_cls(cfg.hidden_dim_2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim_2, cfg.out_dim),
            nn.LayerNorm(cfg.out_dim),
        )

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        return self.net(x_2d)
