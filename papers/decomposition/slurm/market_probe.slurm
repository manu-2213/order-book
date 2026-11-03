#!/bin/bash
#SBATCH -J market_probe
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
# The market with no executor on tuning seeds 0-19 (spread, depth, volatility, turnover), which the paper's "Scale and clock" quotes. Writes market_probe/.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.market_probe --workers $SLURM_NTASKS
