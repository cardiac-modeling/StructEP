from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from structep import (
    ModelContractError,
    build_model,
    load_model_config,
    normalize_channel,
    read_model_registry,
    select_model_specs,
)


EXPECTED_STATE_LAYOUT_SHA256 = (
    "16cf09b99b2aa89efd5f2d5e7af2e8ad3d94e5a8f87f8a3d7e400ed4ef40186d"
)


def test_registry_is_complete() -> None:
    entries = read_model_registry()
    assert len(entries) == 30
    assert {
        (entry.channel_key, entry.ensemble_member_index) for entry in entries
    } == {
        (channel, member)
        for channel in ("cav1d2", "herg", "nav1d5")
        for member in range(10)
    }
    assert len({entry.model_id for entry in entries}) == 30
    assert len({entry.weight_file for entry in entries}) == 30


def test_channel_aliases_and_canonical_keys() -> None:
    assert normalize_channel("hERG") == "herg"
    assert normalize_channel("Kv11") == "herg"
    assert normalize_channel("NaV1.5") == "nav1d5"
    assert normalize_channel("nav1d5") == "nav1d5"
    assert normalize_channel("SCN5A") == "nav1d5"
    assert normalize_channel("CaV1.2") == "cav1d2"
    assert normalize_channel("cav1d2") == "cav1d2"
    assert normalize_channel("CACNA1C") == "cav1d2"
    assert len(select_model_specs(channel="herg")) == 10


def test_all_channel_configs_build_the_checkpoint_layout() -> None:
    for channel in ("cav1d2", "herg", "nav1d5"):
        config = load_model_config(channel)
        model = build_model(config)
        state = model.state_dict()
        layout = "\n".join(
            f"{key}|{tuple(value.shape)}|{value.dtype}"
            for key, value in state.items()
        )
        assert hashlib.sha256(layout.encode()).hexdigest() == EXPECTED_STATE_LAYOUT_SHA256
        assert len(state) == 223
        assert sum(value.numel() for value in state.values()) == 5_180_904
        assert sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ) == 5_180_840


def test_model_contract_rejects_architecture_and_schema_drift() -> None:
    architecture_drift = deepcopy(load_model_config("herg"))
    architecture_drift["model"]["d_model"] = 256
    with pytest.raises(ModelContractError, match="model.d_model"):
        build_model(architecture_drift)

    schema_drift = deepcopy(load_model_config("herg"))
    schema_drift["unexpected"] = True
    with pytest.raises(ModelContractError, match="unknown top-level keys"):
        build_model(schema_drift)
