"""Model registry and runtime configuration access."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .errors import RegistryError


CONFIG_DIRECTORY = Path(__file__).resolve().parent / "configs"
REGISTRY_PATH = CONFIG_DIRECTORY / "model_registry.csv"
EXPECTED_CHANNELS = ("cav1d2", "herg", "nav1d5")
EXPECTED_MODEL_COUNT = 30
EXPECTED_MEMBERS_PER_CHANNEL = 10
CHANNEL_LABELS = {
    "cav1d2": "CaV1.2",
    "herg": "hERG",
    "nav1d5": "NaV1.5",
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable metadata for one StructEP ensemble member."""

    model_id: str
    channel_key: str
    channel_label: str
    ensemble_member_index: int
    config_file: str
    weight_file: str
    weight_bytes: int


def normalize_channel(value: str) -> str:
    """Return the canonical channel key accepted by the public API."""

    compact = str(value).strip().lower().replace(".", "").replace("_", "").replace("-", "")
    aliases = {
        "herg": "herg",
        "kv11": "herg",
        "nav15": "nav1d5",
        "nav1d5": "nav1d5",
        "scn5a": "nav1d5",
        "cav12": "cav1d2",
        "cav1d2": "cav1d2",
        "cacna1c": "cav1d2",
    }
    try:
        return aliases[compact]
    except KeyError as exc:
        allowed = ", ".join(EXPECTED_CHANNELS)
        raise RegistryError(f"unknown channel {value!r}; choose one of: {allowed}") from exc


def _safe_relative_path(value: object, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RegistryError(f"unsafe {label}: {value!r}")
    return path


def _contained_file(root: Path, relative: Path, label: str) -> Path:
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / relative).resolve(strict=True)
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise RegistryError(f"{label} is outside its root or is not a file: {relative}")
    return path


def read_model_registry(path: str | Path = REGISTRY_PATH) -> tuple[ModelSpec, ...]:
    """Read and validate the complete three-channel, 30-member registry."""

    registry_path = Path(path).resolve(strict=True)
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != EXPECTED_MODEL_COUNT:
        raise RegistryError(
            f"expected {EXPECTED_MODEL_COUNT} model rows, found {len(rows)}"
        )

    specs: list[ModelSpec] = []
    for row in rows:
        try:
            spec = ModelSpec(
                model_id=str(row["model_id"]).strip(),
                channel_key=str(row["channel_key"]).strip(),
                channel_label=str(row["channel_label"]).strip(),
                ensemble_member_index=int(row["ensemble_member_index"]),
                config_file=str(_safe_relative_path(row["config_file"], "config file")),
                weight_file=str(_safe_relative_path(row["weight_file"], "weight file")),
                weight_bytes=int(row["weight_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError("invalid model registry row") from exc
        _validate_spec(spec)
        specs.append(spec)

    if len({spec.model_id for spec in specs}) != EXPECTED_MODEL_COUNT:
        raise RegistryError("model IDs must be unique")

    identities = {
        (spec.channel_key, spec.ensemble_member_index) for spec in specs
    }
    expected_identities = {
        (channel, member)
        for channel in EXPECTED_CHANNELS
        for member in range(EXPECTED_MEMBERS_PER_CHANNEL)
    }
    if identities != expected_identities:
        raise RegistryError("registry is not a complete three-channel ensemble")

    order = {channel: index for index, channel in enumerate(EXPECTED_CHANNELS)}
    return tuple(
        sorted(specs, key=lambda item: (order[item.channel_key], item.ensemble_member_index))
    )


def _validate_spec(spec: ModelSpec) -> None:
    if spec.channel_key not in EXPECTED_CHANNELS:
        raise RegistryError(f"invalid channel key for {spec.model_id}: {spec.channel_key}")
    if spec.channel_label != CHANNEL_LABELS[spec.channel_key]:
        raise RegistryError(f"invalid channel label for {spec.model_id}")
    if not 0 <= spec.ensemble_member_index < EXPECTED_MEMBERS_PER_CHANNEL:
        raise RegistryError(f"invalid member index for {spec.model_id}")
    expected_id = f"posemil_{spec.channel_key}_member{spec.ensemble_member_index:02d}"
    if spec.model_id != expected_id:
        raise RegistryError(
            f"model ID {spec.model_id!r} does not match expected {expected_id!r}"
        )
    if spec.config_file != f"{spec.channel_key}.yaml":
        raise RegistryError(f"unexpected config file for {spec.model_id}")
    if spec.weight_file != f"{spec.model_id}.safetensors":
        raise RegistryError(f"unexpected weight file for {spec.model_id}")
    if spec.weight_bytes <= 0:
        raise RegistryError(f"invalid checkpoint size for {spec.model_id}")


def get_model_spec(model_id: str) -> ModelSpec:
    """Return one registry entry by exact model ID."""

    matches = [spec for spec in read_model_registry() if spec.model_id == model_id]
    if len(matches) != 1:
        raise RegistryError(f"unknown model ID: {model_id!r}")
    return matches[0]


def select_model_specs(
    *,
    channel: str | None = None,
    model_ids: Sequence[str] | None = None,
) -> tuple[ModelSpec, ...]:
    """Select a channel ensemble or an explicit set of model IDs."""

    specs = read_model_registry()
    requested_ids = tuple(
        dict.fromkeys(model_ids if model_ids is not None else ())
    )
    if channel is not None and requested_ids:
        raise RegistryError("select either a channel or model IDs, not both")
    if channel is not None:
        key = normalize_channel(channel)
        return tuple(spec for spec in specs if spec.channel_key == key)
    if requested_ids:
        by_id = {spec.model_id: spec for spec in specs}
        unknown = [model_id for model_id in requested_ids if model_id not in by_id]
        if unknown:
            raise RegistryError(f"unknown model IDs: {', '.join(unknown)}")
        selected = tuple(by_id[model_id] for model_id in requested_ids)
        channels = {spec.channel_key for spec in selected}
        if len(channels) != 1:
            raise RegistryError("one inference request cannot mix channel-specific models")
        return selected
    raise RegistryError("a channel or at least one model ID is required")


def load_model_config(
    channel: str,
    *,
    config_directory: str | Path = CONFIG_DIRECTORY,
) -> Mapping[str, Any]:
    """Load one channel-specific architecture configuration."""

    key = normalize_channel(channel)
    root = Path(config_directory)
    path = _contained_file(root, Path(f"{key}.yaml"), "model config")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise RegistryError(f"model config must be a mapping: {path}")
    if config.get("channel_key") != key:
        raise RegistryError(f"config channel identity differs: {path}")
    if config.get("channel_label") != CHANNEL_LABELS[key]:
        raise RegistryError(f"config channel label differs: {path}")
    return config


def resolve_weight_path(
    weights_directory: str | Path,
    spec: ModelSpec,
    *,
    verify_size: bool = True,
) -> Path:
    """Resolve a checkpoint from a flat directory or a ``model_weights`` child."""

    root = Path(weights_directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RegistryError(f"weights directory is not a directory: {root}")

    candidates = (
        root / spec.weight_file,
        root / "model_weights" / spec.weight_file,
    )
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not matches:
        raise RegistryError(
            f"checkpoint for {spec.model_id} was not found below {root}"
        )
    if len(set(matches)) != 1:
        raise RegistryError(
            f"checkpoint for {spec.model_id} exists in two supported locations below {root}"
        )
    path = matches[0]
    if path.suffix != ".safetensors":
        raise RegistryError(f"checkpoint must use safetensors: {path}")
    if verify_size and path.stat().st_size != spec.weight_bytes:
        raise RegistryError(
            f"checkpoint size differs for {spec.model_id}: expected "
            f"{spec.weight_bytes}, found {path.stat().st_size}"
        )
    return path


__all__ = [
    "CHANNEL_LABELS",
    "CONFIG_DIRECTORY",
    "EXPECTED_CHANNELS",
    "ModelSpec",
    "get_model_spec",
    "load_model_config",
    "normalize_channel",
    "read_model_registry",
    "resolve_weight_path",
    "select_model_specs",
]
