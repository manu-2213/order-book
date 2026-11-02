# Sourced by every job, from the repo root.
source hpc/settings.sh
module load "$HPC_PYTHON_MODULE"
source .venv/bin/activate
# Parallelism is over processes. One thread each, or NumPy and torch oversubscribe the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE="${WANDB_MODE:-offline}"
