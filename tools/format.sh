#!/usr/bin/env bash
set -euo pipefail

black tiny_rlhf tests examples scripts
isort tiny_rlhf tests examples scripts
