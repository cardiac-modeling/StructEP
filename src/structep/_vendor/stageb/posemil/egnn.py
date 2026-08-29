"""Compatibility types for the unused EGNN architecture branch.

StructEP checkpoints are registered against the Transformer backbone. The
original model module imports these names while defining its optional EGNN
branch, so compact stubs are retained to keep that module importable without
shipping an unrelated implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class EGNNConfig:
    d_model: int = 128
    num_layers: int = 3
    edge_hidden: int = 64
    rbf_centers: int = 16
    rbf_max_dist: float = 12.0
    inter_radius: float = 8.0
    intra_lig_radius: float = 2.5
    protein_knn: int = 8
    update_coords: bool = False
    dropout: float = 0.1
    residue_vocab: int = 22
    atom_vocab: int = 13
    num_channels: int = 0
    n_shared_layers: int = -1
    adapter_rank: int = 16


class ProteinLigandEGNN(nn.Module):
    """Guard against selecting an architecture not used by registered models."""

    def __init__(self, _cfg: EGNNConfig) -> None:
        super().__init__()
        raise ValueError(
            "StructEP registered checkpoints require model.backbone_type='transformer'"
        )
