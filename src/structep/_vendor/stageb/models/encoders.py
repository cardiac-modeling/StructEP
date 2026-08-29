from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalCoordEncoder(nn.Module):
    def __init__(self, out_dim: int, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        in_dim = 3 * (2 * num_freqs + 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 3]
        feats = [coords]
        for i in range(self.num_freqs):
            f = 2.0 ** i
            feats.append(torch.sin(coords * f))
            feats.append(torch.cos(coords * f))
        x = torch.cat(feats, dim=-1)
        return self.proj(x)


class TokenTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask True for valid positions; transformer expects True for padding
        key_padding_mask = ~mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class ProteinResidueEncoder(nn.Module):
    def __init__(self, vocab_size: int = 21, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.coord_emb = SinusoidalCoordEncoder(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.encoder = TokenTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

    def forward(self, aa_idx: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        center = (coords * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        x = self.token_emb(aa_idx) + self.coord_emb(coords - center)
        x = self.norm(x)
        return self.encoder(x, mask)


class LigandAtomEncoder(nn.Module):
    def __init__(self, atom_vocab_size: int = 12, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.atom_emb = nn.Embedding(atom_vocab_size, d_model)
        self.coord_emb = SinusoidalCoordEncoder(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.encoder = TokenTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

    def forward(self, atom_idx: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        center = (coords * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        x = self.atom_emb(atom_idx) + self.coord_emb(coords - center)
        x = self.norm(x)
        return self.encoder(x, mask)


class SmilesSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 97,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 160,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.encoder = TokenTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if token_ids.size(1) > self.max_len:
            raise ValueError(f'SMILES length {token_ids.size(1)} exceeds configured max_len={self.max_len}')
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.norm(x)
        x = self.encoder(x, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
