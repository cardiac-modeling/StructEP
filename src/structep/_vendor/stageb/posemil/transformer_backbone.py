"""Transformer and distance-aware cross-attention 3D backbone.

The input and output contract matches ``ProteinLigandEGNN`` so the surrounding
multiple-instance model can switch backbones without changing its batch shape.
The stack contains separate protein-residue and ligand-atom Transformers,
distance-aware cross-attention fusion blocks, masked attention pooling, and a
final instance-level projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..models.encoders import LigandAtomEncoder, ProteinResidueEncoder
from ..models.fusion import ComplexFusionBlock


@dataclass
class TransformerBackboneConfig:
    d_model: int = 128
    protein_layers: int = 3
    ligand_layers: int = 2
    fusion_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1


class _MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1)
        has_any = mask.any(dim=1, keepdim=True)
        safe_mask = torch.where(has_any, mask, torch.ones_like(mask))
        logits = logits.masked_fill(~safe_mask, -1e9)
        attn = torch.softmax(logits, dim=1)
        attn = attn * mask.float()
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attn = attn / denom
        pooled = (attn.unsqueeze(-1) * tokens).sum(dim=1)
        pooled = pooled * has_any.float()
        return pooled


class ProteinLigandTransformer(nn.Module):
    """Encode one flattened set of protein-ligand pose instances."""

    def __init__(self, cfg: TransformerBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.protein_encoder = ProteinResidueEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.protein_layers,
            dropout=cfg.dropout,
        )
        self.ligand_encoder = LigandAtomEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.ligand_layers,
            dropout=cfg.dropout,
        )
        self.fusion = nn.ModuleList([
            ComplexFusionBlock(d_model=cfg.d_model, dropout=cfg.dropout)
            for _ in range(cfg.fusion_layers)
        ])
        self.prot_pool = _MaskedAttentionPool(cfg.d_model)
        self.lig_pool = _MaskedAttentionPool(cfg.d_model)
        self.combine = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prot_tokens = self.protein_encoder(batch["protein_aa"], batch["protein_xyz"], batch["protein_mask"])
        lig_tokens = self.ligand_encoder(batch["ligand_atom"], batch["ligand_xyz"], batch["ligand_mask"])
        for block in self.fusion:
            prot_tokens, lig_tokens = block(
                prot_tokens, lig_tokens,
                batch["protein_xyz"], batch["ligand_xyz"],
                batch["protein_mask"], batch["ligand_mask"],
            )
        z_prot = self.prot_pool(prot_tokens, batch["protein_mask"])
        z_lig = self.lig_pool(lig_tokens, batch["ligand_mask"])
        z_inst = self.combine(torch.cat([z_lig, z_prot], dim=-1))
        return z_inst, {"z_lig_pool": z_lig, "z_prot_pool": z_prot}
