"""Prediction heads for Tri-Channel PoseMIL.

The standard checkpoint path returns four outputs from the fused bag embedding:
``mu_pic50``, ``log_var_pic50``, ``blocker_logit``, and monotonic cumulative
``ordinal_logits``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadsConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    head_type: str = "standard"  # 'standard' | 'ordinal_residual'
    ordinal_thresholds: Tuple[float, ...] = (5.0, 6.0, 7.0, 8.0)
    ordinal_value_clip: Tuple[float, float] = (3.0, 10.0)
    ordinal_residual_scale: float = 1.0
    log_var_clip: Tuple[float, float] = (-4.0, 2.0)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Linear(hidden // 2, out_dim),
    )


class CumulativeOrdinalHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, thresholds: Sequence[float], dropout: float) -> None:
        super().__init__()
        thresholds = tuple(float(x) for x in thresholds)
        if not thresholds:
            raise ValueError("ordinal_thresholds must be non-empty")
        self.thresholds = thresholds
        self.net = _mlp(in_dim, hidden, len(thresholds), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if raw.size(-1) == 1:
            return raw
        base = raw[..., :1]
        gaps = F.softplus(raw[..., 1:])
        return torch.cat([base, base - torch.cumsum(gaps, dim=-1)], dim=-1)


class OrdinalResidualHead(nn.Module):
    """Ordinal-bin plus within-bin residual pIC50 head."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        thresholds: Sequence[float],
        value_clip: Sequence[float],
        dropout: float,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        thresholds = tuple(float(x) for x in thresholds)
        if not thresholds:
            raise ValueError("ordinal_thresholds must be non-empty")
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError(f"ordinal_thresholds must be sorted ascending: {thresholds!r}")
        if len(value_clip) != 2:
            raise ValueError(f"ordinal_value_clip must contain exactly two values, got {value_clip!r}")
        clip_min, clip_max = float(value_clip[0]), float(value_clip[1])
        if not (clip_min < thresholds[0] and thresholds[-1] < clip_max):
            raise ValueError(
                "ordinal_value_clip must bracket all thresholds: "
                f"clip=({clip_min}, {clip_max}), thresholds={thresholds}"
            )

        boundaries = torch.tensor([clip_min, *thresholds, clip_max], dtype=torch.float32)
        centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        half_widths = 0.5 * (boundaries[1:] - boundaries[:-1])
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer("centers", centers)
        self.register_buffer("half_widths", half_widths)
        self.register_buffer("clip_min", torch.tensor(clip_min, dtype=torch.float32))
        self.register_buffer("clip_max", torch.tensor(clip_max, dtype=torch.float32))
        self.residual_scale = float(residual_scale)

        self.ordinal_raw = _mlp(in_dim, hidden, len(thresholds), dropout)
        self.bin_embedding = nn.Parameter(torch.zeros(len(thresholds) + 1, in_dim))
        self.residual = _mlp(in_dim, hidden, 1, dropout)
        nn.init.normal_(self.bin_embedding, std=0.02)

    def _monotonic_logits(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.size(-1) == 1:
            return raw
        base = raw[..., :1]
        gaps = F.softplus(raw[..., 1:])
        return torch.cat([base, base - torch.cumsum(gaps, dim=-1)], dim=-1)

    def clamp_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.clamp(min=float(self.clip_min.item()), max=float(self.clip_max.item()))

    def ordinal_targets(self, labels: torch.Tensor) -> torch.Tensor:
        y = self.clamp_labels(labels)
        return (y.unsqueeze(-1) >= self.thresholds.unsqueeze(0)).to(dtype=labels.dtype)

    def bin_indices(self, labels: torch.Tensor) -> torch.Tensor:
        y = self.clamp_labels(labels)
        return torch.bucketize(y, self.thresholds, right=True).clamp(max=self.centers.numel() - 1)

    def residual_targets(self, labels: torch.Tensor) -> torch.Tensor:
        y = self.clamp_labels(labels)
        bin_idx = self.bin_indices(y)
        return y - self.centers[bin_idx]

    def forward(self, x: torch.Tensor, fine_x: torch.Tensor | None = None) -> dict:
        if fine_x is None:
            fine_x = x
        ordinal_logits = self._monotonic_logits(self.ordinal_raw(x))
        survival_probs = torch.sigmoid(ordinal_logits)
        bin_probs = torch.cat(
            [
                1.0 - survival_probs[:, :1],
                survival_probs[:, :-1] - survival_probs[:, 1:],
                survival_probs[:, -1:],
            ],
            dim=-1,
        )
        bin_probs = bin_probs.clamp_min(0.0)
        bin_probs = bin_probs / bin_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        residual_in = fine_x.unsqueeze(1) + self.bin_embedding.unsqueeze(0)
        raw_residuals = self.residual(residual_in).squeeze(-1)
        residuals = torch.tanh(raw_residuals) * self.half_widths.unsqueeze(0) * self.residual_scale
        bin_values = self.centers.unsqueeze(0) + residuals
        pred = torch.sum(bin_probs * bin_values, dim=-1)
        return {
            "mu_pic50": pred,
            "ordinal_logits": ordinal_logits,
            "ordinal_survival_probs": survival_probs,
            "ordinal_bin_probs": bin_probs,
            "ordinal_residuals": residuals,
            "ordinal_bin_values": bin_values,
            "blocker_logit": ordinal_logits[:, 0],
            "log_var_pic50": pred.new_zeros(pred.shape),
        }


class PoseMILHeads(nn.Module):
    def __init__(self, cfg: HeadsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_type = str(cfg.head_type).lower()
        if self.head_type not in {"standard", "ordinal_residual"}:
            raise ValueError(f"Unsupported PoseMIL head_type: {cfg.head_type!r}")
        if self.head_type == "standard":
            self.mu_head = _mlp(cfg.input_dim, cfg.hidden_dim, 1, cfg.dropout)
            self.log_var_head = _mlp(cfg.input_dim, cfg.hidden_dim, 1, cfg.dropout)
            self.blocker_head = _mlp(cfg.input_dim, cfg.hidden_dim, 1, cfg.dropout)
            self.ordinal_head = CumulativeOrdinalHead(cfg.input_dim, cfg.hidden_dim, cfg.ordinal_thresholds, cfg.dropout)
        else:
            self.ordinal_residual_head = OrdinalResidualHead(
                cfg.input_dim,
                cfg.hidden_dim,
                cfg.ordinal_thresholds,
                cfg.ordinal_value_clip,
                cfg.dropout,
                residual_scale=cfg.ordinal_residual_scale,
            )
        self.rank_head = _mlp(cfg.input_dim * 3, cfg.hidden_dim, 1, cfg.dropout)

    def forward(self, z: torch.Tensor, fine_z: torch.Tensor | None = None) -> dict:
        if self.head_type == "ordinal_residual":
            return self.ordinal_residual_head(z, fine_x=fine_z)
        log_var = self.log_var_head(z).squeeze(-1).clamp(*self.cfg.log_var_clip)
        return {
            "mu_pic50": self.mu_head(z).squeeze(-1),
            "log_var_pic50": log_var,
            "blocker_logit": self.blocker_head(z).squeeze(-1),
            "ordinal_logits": self.ordinal_head(z),
        }

    def rank_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left, right, torch.abs(left - right)], dim=-1)
        return self.rank_head(pair).squeeze(-1)
