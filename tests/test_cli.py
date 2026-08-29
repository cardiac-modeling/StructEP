from __future__ import annotations

import pytest

from structep import load_model_config, load_npz_batch
from structep.cli import main


def test_list_models_command_filters_channel(capsys) -> None:
    exit_code = main(["list-models", "--channel", "herg"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "posemil_herg_member00" in captured.out
    assert "posemil_herg_member09" in captured.out
    assert "posemil_nav1d5_member00" not in captured.out


def test_make_smoke_input_command(tmp_path, capsys) -> None:
    destination = tmp_path / "smoke.npz"
    exit_code = main(
        [
            "make-smoke-input",
            "--channel",
            "cav1d2",
            "--sample-id",
            "deployment_check",
            "--output",
            str(destination),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert destination.is_file()
    assert str(destination) in captured.out
    batch, sample_ids = load_npz_batch(
        destination,
        load_model_config("cav1d2"),
    )
    assert batch["num_bags"] == 1
    assert sample_ids == ("deployment_check",)


def test_cli_reports_unknown_channel(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["list-models", "--channel", "not-a-channel"])
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "unknown channel" in captured.err
