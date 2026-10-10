#!/usr/bin/env bash
set -euo pipefail
pro_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BRAINPROMPT_PYTHON:-/home/panyao/miniconda3/envs/brainprompt/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" "$pro_root/llm-main.py" "$@"
