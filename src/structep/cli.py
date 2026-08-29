"""Command-line interface for StructEP inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__
from .batch import make_smoke_batch, save_npz_batch
from .errors import StructEPError
from .inference import predict_npz, verify_registered_models
from .registry import (
    EXPECTED_CHANNELS,
    load_model_config,
    normalize_channel,
    read_model_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structep",
        description="Structure-aware cardiac ion-channel potency inference.",
    )
    parser.add_argument("--version", action="version", version=f"StructEP {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-models",
        help="List registered ensemble members.",
    )
    list_parser.add_argument(
        "--channel",
        help="Filter by channel: cav1d2, herg, or nav1d5.",
    )
    list_parser.set_defaults(handler=_handle_list_models)

    smoke_parser = subparsers.add_parser(
        "make-smoke-input",
        help="Create a deterministic NPZ input for runtime validation.",
    )
    smoke_parser.add_argument("--channel", required=True, choices=EXPECTED_CHANNELS)
    smoke_parser.add_argument("--output", required=True, type=Path)
    smoke_parser.add_argument("--sample-id", default="smoke_sample")
    smoke_parser.set_defaults(handler=_handle_make_smoke_input)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run one member or a channel ensemble on a model-ready NPZ batch.",
    )
    _add_model_selection(predict_parser)
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--weights-dir", required=True, type=Path)
    predict_parser.add_argument("--output", default="-", help="JSON path, or - for stdout.")
    predict_parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or mps")
    predict_parser.add_argument(
        "--include-members",
        action="store_true",
        help="Include individual member predictions in the JSON output.",
    )
    predict_parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip the registry byte-size check; tensor inventory remains strict.",
    )
    predict_parser.set_defaults(handler=_handle_predict)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Strict-load checkpoints and optionally execute a forward smoke test.",
    )
    _add_model_selection(verify_parser)
    verify_parser.add_argument("--weights-dir", required=True, type=Path)
    verify_parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or mps")
    verify_parser.add_argument("--forward-smoke", action="store_true")
    verify_parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip the registry byte-size check; tensor inventory remains strict.",
    )
    verify_parser.add_argument("--output", default="-", help="JSON path, or - for stdout.")
    verify_parser.set_defaults(handler=_handle_verify)
    return parser


def _add_model_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--channel",
        help="Run all ten registered members for one channel.",
    )
    selection.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Run one exact model ID; repeat to select several members.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (StructEPError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _handle_list_models(args: argparse.Namespace) -> int:
    specs = read_model_registry()
    if args.channel:
        channel = normalize_channel(args.channel)
        specs = tuple(spec for spec in specs if spec.channel_key == channel)
    print(f"{'MODEL ID':<29} {'CHANNEL':<8} {'MEMBER':>6}  CHECKPOINT")
    for spec in specs:
        print(
            f"{spec.model_id:<29} {spec.channel_key:<8} "
            f"{spec.ensemble_member_index:>6}  {spec.weight_file}"
        )
    return 0


def _handle_make_smoke_input(args: argparse.Namespace) -> int:
    config = load_model_config(args.channel)
    batch = make_smoke_batch(config)
    output = save_npz_batch(
        args.output,
        batch,
        config,
        sample_ids=[args.sample_id],
    )
    print(output)
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    payload = predict_npz(
        args.input,
        weights_directory=args.weights_dir,
        channel=args.channel,
        model_ids=args.model_ids,
        device=args.device,
        include_members=args.include_members,
        verify_size=not args.skip_size_check,
    )
    _write_json(payload, args.output)
    return 0


def _handle_verify(args: argparse.Namespace) -> int:
    payload = verify_registered_models(
        args.weights_dir,
        channel=args.channel,
        model_ids=args.model_ids,
        device=args.device,
        forward_smoke=args.forward_smoke,
        verify_size=not args.skip_size_check,
    )
    _write_json(payload, args.output)
    return 0


def _write_json(payload: dict[str, Any], destination: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if destination == "-":
        sys.stdout.write(text)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
