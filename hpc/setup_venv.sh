#!/bin/bash
# On the CLUSTER, once and after hpc/requirements.txt changes: build .venv and check it imports.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/settings.sh
module load "$HPC_PYTHON_MODULE"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r hpc/requirements.txt
pip install --no-deps -e .
python -c "import numpy, torch, gymnasium, order_book; print('numpy', numpy.__version__, 'torch', torch.__version__, 'ok')"
