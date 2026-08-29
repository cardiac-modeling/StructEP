"""Safe serialization and validation for model-ready StructEP batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .errors import InputBatchError


FLOAT_KEYS = (
    "x_2d",
    "protein_xyz",
    "ligand_xyz",
    "pose_quality",
    "state_features",
    "state_role",
)
LONG_KEYS = (
    "protein_aa",
    "ligand_atom",
    "bag_index",
    "state_index",
    "state_to_bag",
    "state_type_idx",
    "channel_idx",
)
BOOL_KEYS = (
    "protein_mask",
    "ligand_mask",
)
SCALAR_KEYS = (
    "num_bags",
    "num_states",
)
REQUIRED_KEYS = frozenset(FLOAT_KEYS + LONG_KEYS + BOOL_KEYS + SCALAR_KEYS)
ALLOWED_ARCHIVE_KEYS = REQUIRED_KEYS | {"sample_ids", "schema_version"}

BatchValue = torch.Tensor | int
ModelBatch = dict[str, BatchValue]


def make_smoke_batch(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> ModelBatch:
    """Create a deterministic structural batch for runtime validation."""

    dataset = config.get("dataset")
    model = config.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise InputBatchError("config must contain dataset and model mappings")

    x_2d_dim = int(dataset.get("morgan_bits", 2048)) + int(
        model.get("descriptor_dim", 0)
    )
    quality_dim = int(model.get("quality_dim", 5))
    state_feature_dim = int(model.get("state_feature_dim", 0))
    target = _resolve_device(device)

    batch: ModelBatch = {
        "x_2d": torch.zeros((1, x_2d_dim), dtype=torch.float32, device=target),
        "protein_aa": torch.zeros((2, 4), dtype=torch.long, device=target),
        "protein_xyz": torch.zeros((2, 4, 3), dtype=torch.float32, device=target),
        "protein_mask": torch.ones((2, 4), dtype=torch.bool, device=target),
        "ligand_atom": torch.zeros((2, 3), dtype=torch.long, device=target),
        "ligand_xyz": torch.zeros((2, 3, 3), dtype=torch.float32, device=target),
        "ligand_mask": torch.ones((2, 3), dtype=torch.bool, device=target),
        "pose_quality": torch.zeros((2, quality_dim), dtype=torch.float32, device=target),
        "bag_index": torch.zeros(2, dtype=torch.long, device=target),
        "state_index": torch.tensor([0, 1], dtype=torch.long, device=target),
        "state_to_bag": torch.zeros(2, dtype=torch.long, device=target),
        "state_features": torch.empty(
            (2, state_feature_dim), dtype=torch.float32, device=target
        ),
        "state_type_idx": torch.tensor([0, 1], dtype=torch.long, device=target),
        "state_role": torch.ones(2, dtype=torch.float32, device=target),
        "channel_idx": torch.zeros(1, dtype=torch.long, device=target),
        "num_bags": 1,
        "num_states": 2,
    }
    validate_batch(batch, config)
    return batch


def save_npz_batch(
    path: str | Path,
    batch: Mapping[str, BatchValue],
    config: Mapping[str, Any],
    *,
    sample_ids: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Write a validated batch in the portable, non-pickle NPZ format."""

    validate_batch(batch, config, sample_ids=sample_ids)
    output = Path(path).expanduser()
    if output.suffix.lower() != ".npz":
        raise InputBatchError("StructEP batch files must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)

    num_bags = int(batch["num_bags"])
    names = list(
        sample_ids
        if sample_ids is not None
        else [f"sample_{index:06d}" for index in range(num_bags)]
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "sample_ids": np.asarray(names, dtype=np.str_),
    }
    for key in FLOAT_KEYS + LONG_KEYS + BOOL_KEYS:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise InputBatchError(f"{key} must be a tensor")
        arrays[key] = value.detach().cpu().numpy()
    for key in SCALAR_KEYS:
        arrays[key] = np.asarray(int(batch[key]), dtype=np.int64)
    np.savez_compressed(output, **arrays)
    return output.resolve()


def load_npz_batch(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[ModelBatch, tuple[str, ...]]:
    """Load a model-ready batch without enabling pickle deserialization."""

    try:
        return _load_npz_batch(path, config, device=device)
    except InputBatchError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise InputBatchError(f"could not read StructEP NPZ input: {exc}") from exc


def _load_npz_batch(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[ModelBatch, tuple[str, ...]]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".npz":
        raise InputBatchError(f"input must be an existing .npz file: {source}")

    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            missing = sorted(REQUIRED_KEYS - names)
            unknown = sorted(names - ALLOWED_ARCHIVE_KEYS)
            if missing:
                raise InputBatchError(
                    f"input archive is missing keys: {', '.join(missing)}"
                )
            if unknown:
                raise InputBatchError(
                    f"input archive has unknown keys: {', '.join(unknown)}"
                )
            if "schema_version" in archive:
                schema_version = _read_scalar(
                    archive["schema_version"],
                    "schema_version",
                )
                if schema_version != 1:
                    raise InputBatchError(
                        f"unsupported input schema version: {schema_version}"
                    )

            batch: ModelBatch = {}
            for key in FLOAT_KEYS:
                batch[key] = torch.as_tensor(
                    np.asarray(archive[key]),
                    dtype=torch.float32,
                )
            for key in LONG_KEYS:
                batch[key] = torch.as_tensor(
                    np.asarray(archive[key]),
                    dtype=torch.long,
                )
            for key in BOOL_KEYS:
                batch[key] = torch.as_tensor(
                    np.asarray(archive[key]),
                    dtype=torch.bool,
                )
            for key in SCALAR_KEYS:
                batch[key] = _read_scalar(archive[key], key)

            if "sample_ids" in archive:
                raw_ids = np.asarray(archive["sample_ids"])
                if raw_ids.ndim != 1 or raw_ids.dtype.kind not in {"U", "S"}:
                    raise InputBatchError(
                        "sample_ids must be a one-dimensional string array"
                    )
                sample_ids = tuple(str(value) for value in raw_ids.tolist())
            else:
                sample_ids = tuple(
                    f"sample_{index:06d}"
                    for index in range(int(batch["num_bags"]))
                )
    except InputBatchError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise InputBatchError(f"unable to read input archive {source}: {exc}") from exc

    validate_batch(batch, config, sample_ids=sample_ids)
    return move_batch_to_device(batch, device), sample_ids


def validate_batch(
    batch: Mapping[str, BatchValue],
    config: Mapping[str, Any],
    *,
    sample_ids: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Validate dimensions, dtypes, index ranges, and hierarchy consistency."""

    missing = sorted(REQUIRED_KEYS - set(batch))
    if missing:
        raise InputBatchError(f"batch is missing keys: {', '.join(missing)}")

    num_bags = _positive_int(batch["num_bags"], "num_bags")
    num_states = _positive_int(batch["num_states"], "num_states")

    tensors: dict[str, torch.Tensor] = {}
    for key in FLOAT_KEYS + LONG_KEYS + BOOL_KEYS:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise InputBatchError(f"{key} must be a torch.Tensor")
        tensors[key] = value

    for key in FLOAT_KEYS:
        if not tensors[key].is_floating_point():
            raise InputBatchError(f"{key} must use a floating-point dtype")
        if not bool(torch.isfinite(tensors[key]).all()):
            raise InputBatchError(f"{key} contains non-finite values")
    for key in LONG_KEYS:
        if tensors[key].dtype != torch.long:
            raise InputBatchError(f"{key} must use torch.long")
    for key in BOOL_KEYS:
        if tensors[key].dtype != torch.bool:
            raise InputBatchError(f"{key} must use torch.bool")

    dataset = config.get("dataset")
    model = config.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise InputBatchError("config must contain dataset and model mappings")
    x_2d_dim = int(dataset.get("morgan_bits", 2048)) + int(
        model.get("descriptor_dim", 0)
    )
    quality_dim = int(model.get("quality_dim", 5))
    state_feature_dim = int(model.get("state_feature_dim", 0))

    x_2d = tensors["x_2d"]
    protein_aa = tensors["protein_aa"]
    protein_xyz = tensors["protein_xyz"]
    protein_mask = tensors["protein_mask"]
    ligand_atom = tensors["ligand_atom"]
    ligand_xyz = tensors["ligand_xyz"]
    ligand_mask = tensors["ligand_mask"]
    pose_quality = tensors["pose_quality"]
    bag_index = tensors["bag_index"]
    state_index = tensors["state_index"]
    state_to_bag = tensors["state_to_bag"]
    state_features = tensors["state_features"]
    state_type_idx = tensors["state_type_idx"]
    state_role = tensors["state_role"]
    channel_idx = tensors["channel_idx"]

    if x_2d.shape != (num_bags, x_2d_dim):
        raise InputBatchError(
            f"x_2d must have shape ({num_bags}, {x_2d_dim}), found {tuple(x_2d.shape)}"
        )
    if protein_aa.ndim != 2:
        raise InputBatchError("protein_aa must have shape [poses, residues]")
    num_poses, num_residues = protein_aa.shape
    if num_poses < 1 or num_residues < 1:
        raise InputBatchError("protein_aa must contain at least one pose and residue")
    if protein_xyz.shape != (num_poses, num_residues, 3):
        raise InputBatchError("protein_xyz must align with protein_aa and end in xyz")
    if protein_mask.shape != protein_aa.shape:
        raise InputBatchError("protein_mask must match protein_aa")
    if ligand_atom.ndim != 2 or ligand_atom.shape[0] != num_poses:
        raise InputBatchError("ligand_atom must have shape [poses, atoms]")
    num_atoms = ligand_atom.shape[1]
    if num_atoms < 1:
        raise InputBatchError("ligand_atom must contain at least one atom")
    if ligand_xyz.shape != (num_poses, num_atoms, 3):
        raise InputBatchError("ligand_xyz must align with ligand_atom and end in xyz")
    if ligand_mask.shape != ligand_atom.shape:
        raise InputBatchError("ligand_mask must match ligand_atom")
    if pose_quality.shape != (num_poses, quality_dim):
        raise InputBatchError(
            f"pose_quality must have shape ({num_poses}, {quality_dim})"
        )
    if bag_index.shape != (num_poses,) or state_index.shape != (num_poses,):
        raise InputBatchError("bag_index and state_index must have one value per pose")
    if state_to_bag.shape != (num_states,):
        raise InputBatchError("state_to_bag must have one value per state")
    if state_features.shape != (num_states, state_feature_dim):
        raise InputBatchError(
            f"state_features must have shape ({num_states}, {state_feature_dim})"
        )
    if state_type_idx.shape != (num_states,) or state_role.shape != (num_states,):
        raise InputBatchError("state_type_idx and state_role must have one value per state")
    if channel_idx.shape != (num_bags,):
        raise InputBatchError("channel_idx must have one value per bag")

    if not bool(protein_mask.any(dim=1).all()):
        raise InputBatchError("each pose must contain at least one protein residue")
    if not bool(ligand_mask.any(dim=1).all()):
        raise InputBatchError("each pose must contain at least one ligand atom")
    _check_index_range(protein_aa, 21, "protein_aa")
    _check_index_range(ligand_atom, 12, "ligand_atom")
    _check_index_range(bag_index, num_bags, "bag_index")
    _check_index_range(state_index, num_states, "state_index")
    _check_index_range(state_to_bag, num_bags, "state_to_bag")
    _check_index_range(state_type_idx, 4, "state_type_idx")
    if not bool((channel_idx == 0).all()):
        raise InputBatchError("channel_idx must be zero for channel-specific checkpoints")
    if not bool((state_to_bag[state_index] == bag_index).all()):
        raise InputBatchError("state_index and state_to_bag disagree with bag_index")

    bag_counts = torch.bincount(bag_index, minlength=num_bags)
    state_counts = torch.bincount(state_index, minlength=num_states)
    if not bool((bag_counts > 0).all()):
        raise InputBatchError("every bag must contain at least one pose")
    if not bool((state_counts > 0).all()):
        raise InputBatchError("every state must contain at least one pose")

    if sample_ids is not None:
        if len(sample_ids) != num_bags:
            raise InputBatchError("sample_ids must have one value per bag")
        if any(not str(value).strip() for value in sample_ids):
            raise InputBatchError("sample_ids cannot contain empty values")
        if len(set(str(value) for value in sample_ids)) != len(sample_ids):
            raise InputBatchError("sample_ids must be unique within a batch")


def move_batch_to_device(
    batch: Mapping[str, BatchValue],
    device: str | torch.device,
) -> ModelBatch:
    """Copy all tensors to one validated runtime device."""

    target = _resolve_device(device)
    return {
        key: value.to(target) if isinstance(value, torch.Tensor) else int(value)
        for key, value in batch.items()
        if key in REQUIRED_KEYS
    }


def _read_scalar(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    if array.shape not in {(), (1,)}:
        raise InputBatchError(f"{label} must be a scalar integer")
    try:
        integer = int(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputBatchError(f"{label} must be a scalar integer") from exc
    if float(array.reshape(-1)[0]) != float(integer):
        raise InputBatchError(f"{label} must be a scalar integer")
    return integer


def _positive_int(value: BatchValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputBatchError(f"{label} must be a positive integer")
    if value < 1:
        raise InputBatchError(f"{label} must be a positive integer")
    return value


def _check_index_range(values: torch.Tensor, upper: int, label: str) -> None:
    if values.numel() == 0:
        raise InputBatchError(f"{label} cannot be empty")
    if int(values.min().item()) < 0 or int(values.max().item()) >= upper:
        raise InputBatchError(f"{label} contains an out-of-range value")


def _resolve_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise InputBatchError("CUDA was requested but is not available")
    if target.type not in {"cpu", "cuda", "mps"}:
        raise InputBatchError(f"unsupported device type: {target.type}")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise InputBatchError("MPS was requested but is not available")
    return target


__all__ = [
    "ModelBatch",
    "load_npz_batch",
    "make_smoke_batch",
    "move_batch_to_device",
    "save_npz_batch",
    "validate_batch",
]
