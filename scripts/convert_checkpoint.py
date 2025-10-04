#!/usr/bin/env python3
"""Convert TinyRLHF checkpoints into Hugging Face adapter format."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a TinyRLHF checkpoint to Hugging Face format.")
    parser.add_argument("--input", required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output directory for the converted adapter.")
    return parser.parse_args(argv)


def run(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = torch.load(input_dir / "adapter.pt", map_location="cpu")
    torch.save(adapter_state, output_dir / "adapter_model.bin")
    print(f"Saved adapter weights to {output_dir / 'adapter_model.bin'}")


if __name__ == "__main__":
    run()
