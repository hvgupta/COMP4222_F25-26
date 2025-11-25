#!/usr/bin/env bash
set -euo pipefail

# Choose Python interpreter (prefer python3)
PY=${PY:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "No python interpreter found (tried python3 and python)" >&2
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    "$PY" -m venv "$VENV_DIR"
fi

# Activate the environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Ensure pip is up-to-date
pip install --upgrade pip setuptools wheel

# Install requirements if present
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found — skipping dependency installation"
fi

# Run the desired command
python -m scripts.hparam_search --max-combos 5 --output-dir exp1