#!/usr/bin/env bash
set -euo pipefail

# Wrapper to call the modular pipeline defined within the trial package.
# Usage examples:
#   ./trial/run_pipeline.sh download
#   ./trial/run_pipeline.sh prep --limit 500
#   ./trial/run_pipeline.sh baseline --model_name <hf_id>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 -m trial.pipeline "$@"
