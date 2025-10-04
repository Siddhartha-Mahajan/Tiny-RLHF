#!/usr/bin/env bash
set -euo pipefail

flake8 tiny_rlhf tests
mypy tiny_rlhf
