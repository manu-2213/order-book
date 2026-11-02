# Everything specific to the cluster, in one place. Read by push.sh, pull.sh, setup_venv.sh and submit.sh.
export HPC_USER="${HPC_USER:-acw796}"
export HPC_HOST="${HPC_HOST:-login.hpc.qmul.ac.uk}"
export HPC_REPO="${HPC_REPO:-/data/home/$HPC_USER/order-book}"
export HPC_PYTHON_MODULE="${HPC_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"

# Cores, memory, partition and runtime are in each .slurm file, as the cluster's guide recommends. The simulator is CPU-only,
# so jobs use the default compute partition, open to every user, and need no account or GPU.
