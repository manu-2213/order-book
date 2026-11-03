#!/bin/bash
#SBATCH -J rl_ceiling
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
# The margin over TWAP available to 27 tuned heuristics on the policies' unseen seeds (4,480 paired runs), and what detection and
# predation would cost TWAP under a more sensitive adversary (240). Needs no policies; runs with the rl results.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.rl_ceiling --workers $SLURM_NTASKS
