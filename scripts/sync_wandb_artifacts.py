#!/usr/bin/env python3
"""Download checkpoints from Weights & Biases artifact storage."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    import wandb  # type: ignore

try:
    import wandb  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install tiny-rlhf[wandb] to use this script.") from exc


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync a WandB artifact to the local filesystem.")
    parser.add_argument("artifact", help="Artifact path, e.g. entity/project/run:alias")
    parser.add_argument("--target", default="artifacts", help="Directory to place downloaded files.")
    return parser.parse_args(argv)


def run(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    artifact = api.artifact(args.artifact)
    artifact.download(root=str(target_dir))
    print(f"Downloaded {artifact.name} to {target_dir}")


if __name__ == "__main__":
    run()
