#!/bin/bash
#SBATCH -J e7
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # the email reports CPU and memory efficiency

# E7 regimes. Not sharded or resumable: fits the hour at 16 cores from the Mac's speed; re-submitting starts over.
# Submit with hpc/submit.sh.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.stealth_execution.scripts.e7_regimes --workers $SLURM_NTASKS
