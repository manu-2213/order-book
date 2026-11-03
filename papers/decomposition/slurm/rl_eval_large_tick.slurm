#!/bin/bash
#SBATCH -J rl_eval_large_tick
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL
# Every policy of the large-tick sweep against TWAP on the validation seeds and on 80 unseen seeds, paired, in the paired worlds, and the per-seed analysis. After the rl chain.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.rl_eval --sweep-set large_tick --workers $SLURM_NTASKS
