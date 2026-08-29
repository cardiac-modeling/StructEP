"""Strict construction contract for the checkpoint-compatible StructEP model."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch

from structep._vendor.stageb.posemil.model import (
    TriChannelPoseMIL,
    TriChannelPoseMILConfig,
)


EXPECTED_TRAINABLE_PARAMETERS = 5_180_840
EXPECTED_STATE_DICT_NUMEL = 5_180_904

REQUIRED_MODEL_VALUES: dict[str, Any] = {
    "backbone_type": "transformer",
    "head_type": "standard",
    "d_model": 128,
    "dropout": 0.1,
    "transformer_protein_layers": 3,
    "transformer_ligand_layers": 2,
    "transformer_fusion_layers": 2,
    "transformer_nhead": 4,
    "quality_dim": 5,
    "quality_hidden": 64,
    "pose_pooling": "attention",
    "use_pose_quality": True,
    "two_d_hidden_1": 1024,
    "two_d_hidden_2": 512,
    "heads_hidden": 256,
    "fusion_gate_hidden": 128,
    "use_3d_branch": True,
    "fusion_mode": "cf_gated",
    "per_channel_heads": False,
    "use_channel_token": False,
    "assay_adapter_enabled": False,
    "state_aware_3d": True,
    "state_pooling": "attention",
    "state_gate_hidden": 128,
    "state_feature_dim": 0,
    "state_type_emb_dim": 0,
    "state_aux_logit_bias": 0.0,
    "open_inact_delta_scale": 0.0,
    "dual_gaussian_enabled": False,
    "use_ifp": False,
    "ifp_input_dim": 0,
    "ifp_aux_enabled": False,
    "ifp_aux_dim": 0,
    "use_maccs_residual_fusion": False,
    "use_plec_head_residual": False,
    "plec_sidecar_dim": 0,
}

REQUIRED_LOSS_VALUES: dict[str, float] = {
    "state_aux_attention_lambda": 0.0,
    "state_aux_nonherg_attention_lambda": 0.0,
    "state_entropy_lambda": 0.0,
}


class ModelContractError(ValueError):
    """Raised when a configuration or model drifts from the released contract."""


class ZeroWidthEmbedding(torch.nn.Embedding):
    """Embedding with a safe forward path when ``embedding_dim == 0``.

    The replacement preserves the exact state-dict key and tensor shape used by
    the checkpoints without applying a process-wide PyTorch monkeypatch.
    """

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.embedding_dim == 0:
            return self.weight.new_empty((*indices.shape, 0))
        return super().forward(indices)


def _require_values(
    section: Mapping[str, Any],
    required: Mapping[str, Any],
    section_name: str,
) -> None:
    for key, expected in required.items():
        if key not in section:
            raise ModelContractError(f"{section_name}.{key} must be explicit")
        observed = section[key]
        if observed != expected:
            raise ModelContractError(
                f"{section_name}.{key}={observed!r}; expected {expected!r}"
            )


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless *config* explicitly describes the released model."""

    if config.get("schema_version") != 1:
        raise ModelContractError("schema_version must be 1")
    if config.get("strict_config_keys") is not True:
        raise ModelContractError("strict_config_keys must be true")
    if config.get("state_metadata_ablation_mode") != "symmetric_structure_only":
        raise ModelContractError(
            "state_metadata_ablation_mode must be symmetric_structure_only"
        )
    allowed_top_level = {
        "schema_version",
        "strict_config_keys",
        "state_metadata_ablation_mode",
        "channel_key",
        "channel_label",
        "dataset",
        "model",
        "loss",
    }
    unknown_top_level = sorted(set(config) - allowed_top_level)
    if unknown_top_level:
        raise ModelContractError(
            f"unknown top-level keys: {', '.join(unknown_top_level)}"
        )

    dataset = config.get("dataset")
    model = config.get("model")
    loss = config.get("loss")
    if not isinstance(dataset, Mapping):
        raise ModelContractError("dataset must be a mapping")
    if not isinstance(model, Mapping) or not isinstance(loss, Mapping):
        raise ModelContractError("model and loss must be mappings")
    _require_values(model, REQUIRED_MODEL_VALUES, "model")
    _require_values(loss, REQUIRED_LOSS_VALUES, "loss")

    unknown_dataset_keys = sorted(set(dataset) - {"morgan_bits", "channels"})
    if unknown_dataset_keys:
        raise ModelContractError(
            f"unknown dataset keys: {', '.join(unknown_dataset_keys)}"
        )
    unknown_loss_keys = sorted(set(loss) - set(REQUIRED_LOSS_VALUES))
    if unknown_loss_keys:
        raise ModelContractError(
            f"unknown loss keys: {', '.join(unknown_loss_keys)}"
        )

    channels = dataset.get("channels")
    if not isinstance(channels, list) or len(channels) != 1 or not str(channels[0]).strip():
        raise ModelContractError("dataset.channels must contain exactly one channel")
    if int(dataset.get("morgan_bits", 0)) != 2048:
        raise ModelContractError("dataset.morgan_bits must be 2048")

    channel_contract = {
        "herg": ("hERG", (4.0, 5.0, 6.0, 7.0), (3.0, 8.0)),
        "nav1d5": ("NaV1.5", (4.0, 5.0, 6.0, 7.0), (3.0, 8.0)),
        "cav1d2": ("CaV1.2", (5.0, 6.0, 7.0, 8.0), (3.0, 10.0)),
    }
    channel_key = config.get("channel_key")
    if channel_key not in channel_contract:
        raise ModelContractError(f"unsupported channel_key: {channel_key!r}")
    channel_label, expected_thresholds, expected_clip = channel_contract[channel_key]
    if config.get("channel_label") != channel_label:
        raise ModelContractError("channel_label does not match channel_key")
    if channels != [channel_label]:
        raise ModelContractError("dataset.channels does not match channel_key")
    observed_thresholds = tuple(float(value) for value in model.get("ordinal_thresholds", ()))
    observed_clip = tuple(float(value) for value in model.get("ordinal_value_clip", ()))
    if observed_thresholds != expected_thresholds:
        raise ModelContractError("ordinal_thresholds do not match channel contract")
    if observed_clip != expected_clip:
        raise ModelContractError("ordinal_value_clip does not match channel contract")

    allowed_model_keys = {item.name for item in fields(TriChannelPoseMILConfig)}
    derived_keys = {"morgan_bits", "descriptor_dim", "channels"}
    unknown = sorted(set(model) - allowed_model_keys - derived_keys)
    if unknown:
        raise ModelContractError(f"unknown model keys: {', '.join(unknown)}")


def _replace_zero_width_embedding(model: TriChannelPoseMIL) -> None:
    aggregator = getattr(model, "state_aggregator", None)
    embedding = getattr(aggregator, "state_type_emb", None)
    if not isinstance(embedding, torch.nn.Embedding) or embedding.embedding_dim != 0:
        raise ModelContractError("model must contain a zero-width state embedding")
    replacement = ZeroWidthEmbedding(
        embedding.num_embeddings,
        embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        _weight=embedding.weight.detach(),
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    aggregator.state_type_emb = replacement


def build_model(config: Mapping[str, Any]) -> TriChannelPoseMIL:
    """Construct the exact architecture expected by the registered checkpoints."""

    validate_model_config(config)
    dataset = config["dataset"]
    model_values = dict(config["model"])
    model_values.setdefault("morgan_bits", int(dataset.get("morgan_bits", 2048)))
    model_values.setdefault("descriptor_dim", 0)
    model_values.setdefault("channels", tuple(dataset["channels"]))

    allowed = {item.name for item in fields(TriChannelPoseMILConfig)}
    unknown = sorted(set(model_values) - allowed)
    if unknown:
        raise ModelContractError(f"unknown resolved model keys: {', '.join(unknown)}")

    model = TriChannelPoseMIL(TriChannelPoseMILConfig(**model_values))
    _replace_zero_width_embedding(model)

    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    state_numel = sum(item.numel() for item in model.state_dict().values())
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise ModelContractError(
            f"trainable parameter count {trainable:,}; "
            f"expected {EXPECTED_TRAINABLE_PARAMETERS:,}"
        )
    if state_numel != EXPECTED_STATE_DICT_NUMEL:
        raise ModelContractError(
            f"state-dict element count {state_numel:,}; "
            f"expected {EXPECTED_STATE_DICT_NUMEL:,}"
        )
    return model


# Scientific-name aliases retained for checkpoint documentation and notebooks.
build_posemil_model = build_model
validate_posemil_config = validate_model_config
PoseMILContractError = ModelContractError


__all__ = [
    "EXPECTED_STATE_DICT_NUMEL",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "ModelContractError",
    "PoseMILContractError",
    "ZeroWidthEmbedding",
    "build_model",
    "build_posemil_model",
    "validate_model_config",
    "validate_posemil_config",
]
