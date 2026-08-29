from __future__ import annotations

import torch
import torch.nn as nn


class DistanceAwareCrossAttention(nn.Module):
    def __init__(self, d_model: int = 128, num_rbfs: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        centers = torch.linspace(0.0, 12.0, num_rbfs)
        self.register_buffer('rbf_centers', centers)
        self.rbf_width = 1.0
        self.dist_proj = nn.Linear(num_rbfs, 1)
        self.scale = d_model ** -0.5

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [B, Nq, Nk]
        centers = self.rbf_centers.view(1, 1, 1, -1)
        x = dist.unsqueeze(-1)
        return torch.exp(-((x - centers) ** 2) / (2 * self.rbf_width ** 2))

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        q_xyz: torch.Tensor,
        kv_xyz: torch.Tensor,
        q_mask: torch.Tensor,
        kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(q_tokens)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dist = torch.cdist(q_xyz, kv_xyz)
        logits = logits + self.dist_proj(self._rbf(dist)).squeeze(-1)

        valid = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)
        logits = logits.masked_fill(~valid, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = attn.masked_fill(~valid, 0.0)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        return q_tokens + out


class ComplexFusionBlock(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.lig_to_prot = DistanceAwareCrossAttention(d_model=d_model)
        self.prot_to_lig = DistanceAwareCrossAttention(d_model=d_model)
        self.ffn_p = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_l = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        prot_tokens: torch.Tensor,
        lig_tokens: torch.Tensor,
        prot_xyz: torch.Tensor,
        lig_xyz: torch.Tensor,
        prot_mask: torch.Tensor,
        lig_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prot_tokens = self.lig_to_prot(prot_tokens, lig_tokens, prot_xyz, lig_xyz, prot_mask, lig_mask)
        lig_tokens = self.prot_to_lig(lig_tokens, prot_tokens, lig_xyz, prot_xyz, lig_mask, prot_mask)
        prot_tokens = prot_tokens + self.ffn_p(prot_tokens)
        lig_tokens = lig_tokens + self.ffn_l(lig_tokens)
        return prot_tokens, lig_tokens
