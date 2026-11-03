#!/bin/bash
#SBATCH -J runaway_causes
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
# Post-arrival interventions in the runaway cells: 7 variants x 8 cells x 20 seeds, 1,120 runs in the full market. Writes runaway_causes/.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.runaway_causes --workers $SLURM_NTASKS
