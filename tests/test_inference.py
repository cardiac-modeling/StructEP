from __future__ import annotations

import math

import pytest
import torch
from safetensors.torch import save_file

from structep import (
    MemberOutput,
    aggregate_member_outputs,
    build_model,
    get_model_spec,
    load_model_config,
    load_registered_model,
    make_smoke_batch,
    predict_batch,
    verify_registered_models,
)


def _write_synthetic_checkpoint(tmp_path):
    spec = get_model_spec("posemil_herg_member00")
    config = load_model_config(spec.channel_key)
    model = build_model(config)
    path = tmp_path / spec.weight_file
    save_file(model.state_dict(), str(path))
    assert path.stat().st_size == spec.weight_bytes
    return spec, config, path


def test_strict_loading_forward_verification_and_prediction(tmp_path) -> None:
    spec, config, _ = _write_synthetic_checkpoint(tmp_path)
    model, loaded_config, loaded_spec = load_registered_model(
        spec.model_id,
        tmp_path,
    )
    assert loaded_spec == spec
    assert loaded_config["channel_key"] == "herg"

    batch = make_smoke_batch(config)
    with torch.inference_mode():
        output = model(batch)
    assert output["mu_pic50"].shape == (1,)
    assert output["ordinal_logits"].shape == (1, 4)
    assert torch.isfinite(output["mu_pic50"]).all()

    verification = verify_registered_models(
        tmp_path,
        model_ids=[spec.model_id],
        forward_smoke=True,
    )
    assert verification["status"] == "forward_smoke_verified"
    assert verification["verified_model_count"] == 1
    assert verification["models"][0]["strict_load"] == "pass"
    assert verification["models"][0]["forward_smoke"] == "pass"

    report = predict_batch(
        batch,
        weights_directory=tmp_path,
        model_ids=[spec.model_id],
        sample_ids=["synthetic_sample"],
        include_members=True,
    )
    assert report["channel"] == {"key": "herg", "label": "hERG"}
    assert report["ensemble_size"] == 1
    assert report["sample_count"] == 1
    prediction = report["predictions"][0]
    assert prediction["sample_id"] == "synthetic_sample"
    assert math.isfinite(prediction["mean_pic50"])
    assert prediction["epistemic_std_pic50"] == 0.0
    assert prediction["total_std_pic50"] >= 0.0
    assert len(prediction["ordinal_exceedance_probability"]) == 4
    assert prediction["members"][0]["model_id"] == spec.model_id


def test_ensemble_aggregation_uses_total_variance() -> None:
    outputs = [
        MemberOutput(
            model_id="member_a",
            mu_pic50=torch.tensor([1.0, 4.0]),
            log_var_pic50=torch.log(torch.tensor([4.0, 1.0])),
            blocker_logit=torch.tensor([0.0, 2.0]),
            ordinal_logits=torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        ),
        MemberOutput(
            model_id="member_b",
            mu_pic50=torch.tensor([3.0, 6.0]),
            log_var_pic50=torch.log(torch.tensor([4.0, 3.0])),
            blocker_logit=torch.tensor([2.0, 0.0]),
            ordinal_logits=torch.tensor([[2.0, 1.0], [-1.0, -2.0]]),
        ),
    ]

    report = aggregate_member_outputs(
        outputs,
        sample_ids=["a", "b"],
        channel_key="herg",
        channel_label="hERG",
        ordinal_thresholds=[4.0, 5.0],
        include_members=True,
    )

    first = report["predictions"][0]
    assert first["mean_pic50"] == pytest.approx(2.0)
    assert first["aleatoric_std_pic50"] == pytest.approx(2.0)
    assert first["epistemic_std_pic50"] == pytest.approx(1.0)
    assert first["total_std_pic50"] == pytest.approx(math.sqrt(5.0))
    assert first["blocker_probability"] == pytest.approx(
        (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))).item()
        / 2.0
    )
    assert first["ordinal_exceedance_probability"][">=4"] == pytest.approx(
        (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))).item()
        / 2.0
    )
    assert len(first["members"]) == 2
