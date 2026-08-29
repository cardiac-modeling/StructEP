"""StructEP public inference API."""

from .batch import (
    ModelBatch,
    load_npz_batch,
    make_smoke_batch,
    save_npz_batch,
    validate_batch,
)
from .errors import InferenceError, InputBatchError, RegistryError, StructEPError
from .inference import (
    MemberOutput,
    aggregate_member_outputs,
    load_registered_model,
    predict_batch,
    predict_npz,
    verify_registered_models,
)
from .model import (
    ModelContractError,
    PoseMILContractError,
    build_model,
    build_posemil_model,
    load_weights_strict,
    validate_model_config,
    validate_posemil_config,
)
from .registry import (
    ModelSpec,
    get_model_spec,
    load_model_config,
    normalize_channel,
    read_model_registry,
    select_model_specs,
)

__version__ = "0.1.0"

__all__ = [
    "InferenceError",
    "InputBatchError",
    "MemberOutput",
    "ModelBatch",
    "ModelContractError",
    "ModelSpec",
    "PoseMILContractError",
    "RegistryError",
    "StructEPError",
    "aggregate_member_outputs",
    "build_model",
    "build_posemil_model",
    "get_model_spec",
    "load_model_config",
    "load_npz_batch",
    "load_registered_model",
    "load_weights_strict",
    "make_smoke_batch",
    "normalize_channel",
    "predict_batch",
    "predict_npz",
    "read_model_registry",
    "save_npz_batch",
    "select_model_specs",
    "validate_batch",
    "validate_model_config",
    "validate_posemil_config",
    "verify_registered_models",
]
