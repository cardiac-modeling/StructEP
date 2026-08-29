"""Attention-based multiple-instance pooling over docked poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class PoseMILConfig:
    d_model: int = 128
    quality_dim: int = 5
    quality_hidden: int = 32
    pooling_mode: str = "attention"  # attention | mean
    use_ifp: bool = False
    ifp_input_dim: int = 0
    ifp_hidden_dim: int = 128
    ifp_dropout: float = 0.1
    ifp_scale_init: float = 0.1


def _segment_softmax(scores: torch.Tensor, seg_index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    orig_dtype = scores.dtype
    scores_f = scores.float()
    max_per_seg = scores_f.new_full((num_segments,), float("-inf"))
    max_per_seg = max_per_seg.scatter_reduce(0, seg_index, scores_f, reduce="amax", include_self=True)
    max_per_seg = torch.where(torch.isinf(max_per_seg), torch.zeros_like(max_per_seg), max_per_seg)
    shifted = scores_f - max_per_seg[seg_index]
    exp = shifted.exp()
    sum_per_seg = scores_f.new_zeros((num_segments,))
    sum_per_seg.index_add_(0, seg_index, exp)
    denom = sum_per_seg[seg_index].clamp_min(1e-8)
    return (exp / denom).to(dtype=orig_dtype)


class PoseAttentionMIL(nn.Module):
    def __init__(self, cfg: PoseMILConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pooling_mode = str(cfg.pooling_mode or "attention").lower()
        if self.pooling_mode not in {"attention", "mean"}:
            raise ValueError(f"pooling_mode must be 'attention' or 'mean', got {cfg.pooling_mode!r}")
        self.content_score = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Tanh(),
            nn.Linear(cfg.d_model, 1),
        )
        self.quality_norm = nn.LayerNorm(cfg.quality_dim)
        self.quality_score = nn.Sequential(
            nn.Linear(cfg.quality_dim, cfg.quality_hidden),
            nn.GELU(),
            nn.Linear(cfg.quality_hidden, 1),
        )
        self.ifp_encoder = None
        self.ifp_score = None
        self.ifp_scale = None
        if cfg.use_ifp:
            if int(cfg.ifp_input_dim) <= 0:
                raise ValueError("ifp_input_dim must be positive when use_ifp=True")
            self.ifp_encoder = nn.Sequential(
                nn.LayerNorm(cfg.ifp_input_dim),
                nn.Linear(cfg.ifp_input_dim, cfg.ifp_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.ifp_dropout),
                nn.Linear(cfg.ifp_hidden_dim, cfg.ifp_hidden_dim),
                nn.GELU(),
            )
            self.ifp_score = nn.Sequential(
                nn.LayerNorm(cfg.ifp_hidden_dim),
                nn.Linear(cfg.ifp_hidden_dim, 1),
            )
            self.ifp_scale = nn.Parameter(torch.tensor(float(cfg.ifp_scale_init)))

    def forward(
        self,
        inst_emb: torch.Tensor,
        bag_index: torch.Tensor,
        num_bags: int,
        pose_quality: torch.Tensor | None = None,
        ifp_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool instance embeddings and return attention diagnostics."""

        content = self.content_score(inst_emb).squeeze(-1)
        if pose_quality is None:
            quality = inst_emb.new_zeros((inst_emb.size(0),))
        else:
            quality = self.quality_score(self.quality_norm(pose_quality)).squeeze(-1)
        ifp_bias = inst_emb.new_zeros((inst_emb.size(0),))
        if self.ifp_encoder is not None and self.ifp_score is not None and ifp_features is not None:
            ifp = ifp_features.to(device=inst_emb.device, dtype=inst_emb.dtype)
            target_dim = int(self.cfg.ifp_input_dim)
            if ifp.size(-1) < target_dim:
                pad = inst_emb.new_zeros((ifp.size(0), target_dim - ifp.size(-1)))
                ifp = torch.cat([ifp, pad], dim=-1)
            elif ifp.size(-1) > target_dim:
                ifp = ifp[:, :target_dim]
            ifp_bias = self.ifp_score(self.ifp_encoder(ifp)).squeeze(-1)
            ifp_bias = ifp_bias * self.ifp_scale.to(device=inst_emb.device, dtype=inst_emb.dtype)
        scores = content + quality + ifp_bias
        if self.pooling_mode == "mean":
            counts = inst_emb.new_zeros((num_bags,))
            counts.index_add_(0, bag_index, torch.ones_like(content, dtype=counts.dtype))
            attn = 1.0 / counts[bag_index].clamp_min(1.0)
        else:
            attn = _segment_softmax(scores, bag_index, num_bags)
        bag_emb = inst_emb.new_zeros((num_bags, inst_emb.size(-1)))
        bag_emb.index_add_(0, bag_index, (attn.unsqueeze(-1) * inst_emb).to(dtype=bag_emb.dtype))
        return bag_emb, attn, content, quality, ifp_bias
