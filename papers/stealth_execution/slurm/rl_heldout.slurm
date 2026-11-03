#!/bin/bash
#SBATCH -J rl_heldout
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # the email reports CPU and memory efficiency

# The sweep's policies against TWAP on selected and on unseen seeds.
# Submit with hpc/submit.sh.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.stealth_execution.scripts.rl_heldout --workers $SLURM_NTASKS
