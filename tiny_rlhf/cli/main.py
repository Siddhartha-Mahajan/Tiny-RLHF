"""Command-line interface for TinyRLHF."""
from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

from rich.console import Console
from rich.table import Table

from tiny_rlhf.config import ExperimentConfig, load_experiment_config
from tiny_rlhf.pipelines import get_pipeline
from tiny_rlhf.validators.run_checks import run_preflight_checks

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-rlhf", description="TinyRLHF command-line interface")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute an experiment config")
    run_parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    run_parser.add_argument("--set", nargs="*", default=[], help="Override config values (key=value)")
    run_parser.set_defaults(func=_handle_run)

    inspect_parser = subparsers.add_parser("inspect-config", help="Print the resolved configuration")
    inspect_parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    inspect_parser.add_argument("--set", nargs="*", default=[], help="Override config values (key=value)")
    inspect_parser.set_defaults(func=_handle_inspect)

    return parser


def _load_config(path: str, overrides: Iterable[str]) -> ExperimentConfig:
    return load_experiment_config(path, overrides=overrides)


def _handle_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config, args.set)
    run_preflight_checks(config)
    pipeline = get_pipeline(config.run.pipeline)
    console.log(f"Running pipeline [bold]{config.run.pipeline}[/] – output -> {config.run.output_dir}")
    pipeline(config)
    console.log("Run completed.")


def _handle_inspect(args: argparse.Namespace) -> None:
    config = _load_config(args.config, args.set)
    table = Table(title="Resolved Configuration")
    table.add_column("Section")
    table.add_column("Values", overflow="fold")

    for section in ["dataset", "model", "lora", "trainer", "run", "reward"]:
        value = getattr(config, section)
        if value is None:
            continue
        table.add_row(section, value.json(indent=2))

    console.print(table)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
