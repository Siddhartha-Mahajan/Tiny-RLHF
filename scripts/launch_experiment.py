#!/usr/bin/env python3
"""Helper to invoke the TinyRLHF CLI with a config file and optional overrides."""
from __future__ import annotations

import argparse
from typing import List

from tiny_rlhf.cli.main import main as cli_main


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a TinyRLHF experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Configuration overrides in key=value format.",
    )
    return parser.parse_args(argv)


def run(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    cli_args = ["run", "--config", args.config]
    if args.set:
        cli_args.extend(["--set", *args.set])
    cli_main(cli_args)


if __name__ == "__main__":
    run()
