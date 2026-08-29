"""Checkpoint loading, forward inference, and ensemble aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .batch import (
    ModelBatch,
    load_npz_batch,
    make_smoke_batch,
    move_batch_to_device,
    validate_batch,
)
from .errors import InferenceError
from .model import build_model, load_weights_strict
from .registry import (
    ModelSpec,
    get_model_spec,
    load_model_config,
    resolve_weight_path,
    select_model_specs,
)


REQUIRED_OUTPUTS = (
    "mu_pic50",
    "log_var_pic50",
    "blocker_logit",
    "ordinal_logits",
)


@dataclass(frozen=True, slots=True)
class MemberOutput:
    """CPU tensors retained from one ensemble member."""

    model_id: str
    mu_pic50: torch.Tensor
    log_var_pic50: torch.Tensor
    blocker_logit: torch.Tensor
    ordinal_logits: torch.Tensor


def load_registered_model(
    model: str | ModelSpec,
    weights_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_size: bool = True,
) -> tuple[torch.nn.Module, Mapping[str, Any], ModelSpec]:
    """Construct and strictly load one registered checkpoint."""

    spec = get_model_spec(model) if isinstance(model, str) else model
    config = load_model_config(spec.channel_key)
    weight_path = resolve_weight_path(
        weights_directory,
        spec,
        verify_size=verify_size,
    )
    target = _resolve_device(device)
    network = build_model(config)
    load_weights_strict(network, weight_path)
    network.to(target)
    network.eval()
    return network, config, spec


def run_member(
    network: torch.nn.Module,
    batch: Mapping[str, torch.Tensor | int],
    *,
    model_id: str,
) -> MemberOutput:
    """Run one model and retain only the public prediction tensors."""

    with torch.inference_mode():
        output = network(dict(batch))
    missing = [key for key in REQUIRED_OUTPUTS if key not in output]
    if missing:
        raise InferenceError(
            f"forward output for {model_id} is missing: {', '.join(missing)}"
        )

    selected: dict[str, torch.Tensor] = {}
    for key in REQUIRED_OUTPUTS:
        value = output[key]
        if not isinstance(value, torch.Tensor):
            raise InferenceError(f"forward output {key} for {model_id} is not a tensor")
        value = value.detach().float().cpu()
        if not bool(torch.isfinite(value).all()):
            raise InferenceError(f"forward output {key} for {model_id} is non-finite")
        selected[key] = value

    sample_count = selected["mu_pic50"].shape[0]
    if selected["mu_pic50"].shape != (sample_count,):
        raise InferenceError(f"mu_pic50 for {model_id} must have shape [samples]")
    if selected["log_var_pic50"].shape != (sample_count,):
        raise InferenceError(f"log_var_pic50 for {model_id} must have shape [samples]")
    if selected["blocker_logit"].shape != (sample_count,):
        raise InferenceError(f"blocker_logit for {model_id} must have shape [samples]")
    if selected["ordinal_logits"].ndim != 2 or selected["ordinal_logits"].shape[0] != sample_count:
        raise InferenceError(
            f"ordinal_logits for {model_id} must have shape [samples, thresholds]"
        )

    return MemberOutput(
        model_id=model_id,
        mu_pic50=selected["mu_pic50"],
        log_var_pic50=selected["log_var_pic50"],
        blocker_logit=selected["blocker_logit"],
        ordinal_logits=selected["ordinal_logits"],
    )


def predict_batch(
    batch: Mapping[str, torch.Tensor | int],
    *,
    weights_directory: str | Path,
    channel: str | None = None,
    model_ids: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    include_members: bool = False,
    verify_size: bool = True,
) -> dict[str, Any]:
    """Run a channel ensemble or selected members on one model-ready batch."""

    specs = select_model_specs(channel=channel, model_ids=model_ids)
    channel_key = specs[0].channel_key
    channel_label = specs[0].channel_label
    config = load_model_config(channel_key)
    names = tuple(sample_ids) if sample_ids is not None else ()
    validate_batch(batch, config, sample_ids=names if names else None)
    num_bags = int(batch["num_bags"])
    if not names:
        names = tuple(f"sample_{index:06d}" for index in range(num_bags))

    target = _resolve_device(device)
    device_batch = move_batch_to_device(batch, target)
    member_outputs: list[MemberOutput] = []

    for spec in specs:
        network, member_config, _ = load_registered_model(
            spec,
            weights_directory,
            device=target,
            verify_size=verify_size,
        )
        validate_batch(device_batch, member_config, sample_ids=names)
        member_outputs.append(run_member(network, device_batch, model_id=spec.model_id))
        del network
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()

    return aggregate_member_outputs(
        member_outputs,
        sample_ids=names,
        channel_key=channel_key,
        channel_label=channel_label,
        ordinal_thresholds=tuple(float(value) for value in config["model"]["ordinal_thresholds"]),
        include_members=include_members,
    )


def predict_npz(
    input_path: str | Path,
    *,
    weights_directory: str | Path,
    channel: str | None = None,
    model_ids: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    include_members: bool = False,
    verify_size: bool = True,
) -> dict[str, Any]:
    """Load a safe NPZ batch and run registered inference."""

    specs = select_model_specs(channel=channel, model_ids=model_ids)
    config = load_model_config(specs[0].channel_key)
    batch, sample_ids = load_npz_batch(input_path, config, device="cpu")
    return predict_batch(
        batch,
        weights_directory=weights_directory,
        model_ids=[spec.model_id for spec in specs],
        sample_ids=sample_ids,
        device=device,
        include_members=include_members,
        verify_size=verify_size,
    )


def aggregate_member_outputs(
    outputs: Sequence[MemberOutput],
    *,
    sample_ids: Sequence[str],
    channel_key: str,
    channel_label: str,
    ordinal_thresholds: Sequence[float],
    include_members: bool = False,
) -> dict[str, Any]:
    """Combine member predictions using the law of total variance."""

    if not outputs:
        raise InferenceError("at least one member output is required")
    sample_count = outputs[0].mu_pic50.shape[0]
    threshold_count = outputs[0].ordinal_logits.shape[1]
    if len(sample_ids) != sample_count:
        raise InferenceError("sample IDs do not match the prediction batch")
    if len(ordinal_thresholds) != threshold_count:
        raise InferenceError("ordinal thresholds do not match model outputs")

    for output in outputs:
        if output.mu_pic50.shape != (sample_count,):
            raise InferenceError(f"member shape differs: {output.model_id}")
        if output.log_var_pic50.shape != (sample_count,):
            raise InferenceError(f"member shape differs: {output.model_id}")
        if output.blocker_logit.shape != (sample_count,):
            raise InferenceError(f"member shape differs: {output.model_id}")
        if output.ordinal_logits.shape != (sample_count, threshold_count):
            raise InferenceError(f"member ordinal shape differs: {output.model_id}")

    means = torch.stack([output.mu_pic50 for output in outputs], dim=0)
    variances = torch.stack(
        [output.log_var_pic50.exp() for output in outputs], dim=0
    )
    blocker_probabilities = torch.stack(
        [output.blocker_logit.sigmoid() for output in outputs], dim=0
    )
    ordinal_probabilities = torch.stack(
        [output.ordinal_logits.sigmoid() for output in outputs], dim=0
    )

    ensemble_mean = means.mean(dim=0)
    aleatoric_variance = variances.mean(dim=0)
    epistemic_variance = means.var(dim=0, unbiased=False)
    total_variance = aleatoric_variance + epistemic_variance
    aleatoric_std = aleatoric_variance.clamp_min(0.0).sqrt()
    epistemic_std = epistemic_variance.clamp_min(0.0).sqrt()
    total_std = total_variance.clamp_min(0.0).sqrt()
    blocker_probability = blocker_probabilities.mean(dim=0)
    ordinal_probability = ordinal_probabilities.mean(dim=0)

    predictions: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        mean = float(ensemble_mean[sample_index])
        total = float(total_std[sample_index])
        record: dict[str, Any] = {
            "sample_id": str(sample_id),
            "mean_pic50": mean,
            "aleatoric_std_pic50": float(aleatoric_std[sample_index]),
            "epistemic_std_pic50": float(epistemic_std[sample_index]),
            "total_std_pic50": total,
            "pic50_interval_95": [mean - 1.96 * total, mean + 1.96 * total],
            "blocker_probability": float(blocker_probability[sample_index]),
            "ordinal_exceedance_probability": {
                _format_threshold(threshold): float(
                    ordinal_probability[sample_index, threshold_index]
                )
                for threshold_index, threshold in enumerate(ordinal_thresholds)
            },
        }
        if include_members:
            record["members"] = [
                {
                    "model_id": output.model_id,
                    "mean_pic50": float(output.mu_pic50[sample_index]),
                    "std_pic50": float(
                        torch.exp(0.5 * output.log_var_pic50[sample_index])
                    ),
                    "blocker_probability": float(
                        torch.sigmoid(output.blocker_logit[sample_index])
                    ),
                    "ordinal_exceedance_probability": {
                        _format_threshold(threshold): float(
                            torch.sigmoid(
                                output.ordinal_logits[sample_index, threshold_index]
                            )
                        )
                        for threshold_index, threshold in enumerate(ordinal_thresholds)
                    },
                }
                for output in outputs
            ]
        predictions.append(record)

    return {
        "schema_version": 1,
        "channel": {
            "key": channel_key,
            "label": channel_label,
        },
        "ensemble_size": len(outputs),
        "model_ids": [output.model_id for output in outputs],
        "sample_count": sample_count,
        "predictions": predictions,
    }


def verify_registered_models(
    weights_directory: str | Path,
    *,
    channel: str | None = None,
    model_ids: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    forward_smoke: bool = False,
    verify_size: bool = True,
) -> dict[str, Any]:
    """Strict-load selected checkpoints and optionally execute a smoke forward."""

    specs = select_model_specs(channel=channel, model_ids=model_ids)
    target = _resolve_device(device)
    records: list[dict[str, Any]] = []

    for spec in specs:
        network, config, _ = load_registered_model(
            spec,
            weights_directory,
            device=target,
            verify_size=verify_size,
        )
        record: dict[str, Any] = {
            "model_id": spec.model_id,
            "channel": spec.channel_key,
            "strict_load": "pass",
        }
        if forward_smoke:
            batch = make_smoke_batch(config, device=target)
            output = run_member(network, batch, model_id=spec.model_id)
            record.update(
                {
                    "forward_smoke": "pass",
                    "mu_pic50": float(output.mu_pic50[0]),
                    "std_pic50": float(torch.exp(0.5 * output.log_var_pic50[0])),
                }
            )
        records.append(record)
        del network
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "schema_version": 1,
        "status": "forward_smoke_verified" if forward_smoke else "strict_load_verified",
        "verified_model_count": len(records),
        "models": records,
    }


def _resolve_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA was requested but is not available")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise InferenceError("MPS was requested but is not available")
    if target.type not in {"cpu", "cuda", "mps"}:
        raise InferenceError(f"unsupported device type: {target.type}")
    return target


def _format_threshold(value: float) -> str:
    return f">={value:g}"


__all__ = [
    "MemberOutput",
    "aggregate_member_outputs",
    "load_registered_model",
    "predict_batch",
    "predict_npz",
    "run_member",
    "verify_registered_models",
]
