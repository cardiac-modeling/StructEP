from __future__ import annotations

import numpy as np
import pytest
import torch

from structep import (
    InputBatchError,
    load_model_config,
    load_npz_batch,
    make_smoke_batch,
    save_npz_batch,
    validate_batch,
)
from structep.batch import BOOL_KEYS, FLOAT_KEYS, LONG_KEYS, SCALAR_KEYS


def _herg_config():
    return load_model_config("herg")


def _clone_batch(batch):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def test_npz_round_trip_preserves_batch_and_sample_ids(tmp_path) -> None:
    config = _herg_config()
    batch = make_smoke_batch(config)
    destination = save_npz_batch(
        tmp_path / "batch.npz",
        batch,
        config,
        sample_ids=["compound_001"],
    )

    loaded, sample_ids = load_npz_batch(destination, config)
    validate_batch(loaded, config, sample_ids=sample_ids)

    assert sample_ids == ("compound_001",)
    assert loaded["num_bags"] == 1
    assert loaded["num_states"] == 2
    assert loaded["protein_aa"].shape == (2, 4)
    assert loaded["state_features"].shape == (2, 0)
    assert loaded["protein_mask"].dtype == torch.bool
    for key in FLOAT_KEYS + LONG_KEYS + BOOL_KEYS:
        assert torch.equal(loaded[key], batch[key].cpu())


def test_loader_rejects_object_arrays_without_pickle(tmp_path) -> None:
    config = _herg_config()
    batch = make_smoke_batch(config)
    payload = {
        key: batch[key].cpu().numpy()
        for key in FLOAT_KEYS + LONG_KEYS + BOOL_KEYS
    }
    payload.update(
        {
            key: np.asarray(batch[key], dtype=np.int64)
            for key in SCALAR_KEYS
        }
    )
    payload["schema_version"] = np.asarray(1, dtype=np.int64)
    payload["sample_ids"] = np.asarray(["unsafe"], dtype=np.str_)
    payload["x_2d"] = np.asarray([[object()] * 2048], dtype=object)
    path = tmp_path / "object-array.npz"
    np.savez(path, **payload)

    with pytest.raises(InputBatchError, match="allow_pickle=False"):
        load_npz_batch(path, config)


def test_validator_rejects_out_of_range_hierarchy_index() -> None:
    config = _herg_config()
    batch = _clone_batch(make_smoke_batch(config))
    batch["state_index"][0] = 2

    with pytest.raises(InputBatchError, match="state_index"):
        validate_batch(batch, config)


def test_validator_rejects_out_of_range_token_index() -> None:
    config = _herg_config()
    batch = _clone_batch(make_smoke_batch(config))
    batch["ligand_atom"][0, 0] = 12

    with pytest.raises(InputBatchError, match="ligand_atom"):
        validate_batch(batch, config)


def test_validator_rejects_duplicate_sample_ids() -> None:
    config = _herg_config()
    batch = make_smoke_batch(config)

    with pytest.raises(InputBatchError, match="one value per bag"):
        validate_batch(batch, config, sample_ids=["same", "same"])


def test_loader_rejects_unknown_archive_key(tmp_path) -> None:
    config = _herg_config()
    batch = make_smoke_batch(config)
    path = save_npz_batch(tmp_path / "batch.npz", batch, config)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["unexpected"] = np.asarray([1], dtype=np.int64)
    np.savez(path, **payload)

    with pytest.raises(InputBatchError, match="unknown keys"):
        load_npz_batch(path, config)
