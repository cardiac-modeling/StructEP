"""Strict safetensors loading for StructEP inference checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file

from .contract import EXPECTED_STATE_DICT_NUMEL, EXPECTED_TRAINABLE_PARAMETERS


EXPECTED_STATE_DICT_TENSORS = 223


def _validate_state_dict(state: Mapping[str, torch.Tensor]) -> None:
    if len(state) != EXPECTED_STATE_DICT_TENSORS:
        raise ValueError(
            f"checkpoint contains {len(state)} tensors; "
            f"expected {EXPECTED_STATE_DICT_TENSORS}"
        )
    numel = sum(value.numel() for value in state.values())
    if numel != EXPECTED_STATE_DICT_NUMEL:
        raise ValueError(
            f"checkpoint contains {numel:,} tensor elements; "
            f"expected {EXPECTED_STATE_DICT_NUMEL:,}"
        )
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("checkpoints must contain only float32 tensors")


def load_weights_strict(model: torch.nn.Module, path: str | Path) -> None:
    """Load a checkpoint after validating format, inventory, shape, and dtype."""

    weight_path = Path(path).expanduser().resolve(strict=True)
    if not weight_path.is_file() or weight_path.suffix != ".safetensors":
        raise ValueError("checkpoint must be an existing .safetensors file")
    state: Mapping[str, torch.Tensor] = load_file(str(weight_path), device="cpu")
    _validate_state_dict(state)
    model.load_state_dict(state, strict=True)
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"loaded model has {trainable:,} trainable parameters; "
            f"expected {EXPECTED_TRAINABLE_PARAMETERS:,}"
        )


__all__ = ["EXPECTED_STATE_DICT_TENSORS", "load_weights_strict"]
