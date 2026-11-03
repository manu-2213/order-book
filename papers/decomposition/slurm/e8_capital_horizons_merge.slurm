#!/bin/bash
#SBATCH -J e8_capital_horizons_merge
#SBATCH -o logs/%x.o%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # tells you the whole grid has finished, and how efficiently this step ran
# Combines the shards of e8_capital_horizons and writes the summary. Fails, naming the tasks to resubmit, if any run is missing.
# hpc/submit.sh queues this after the array.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_capital_horizons --regime deep_cover_capital10 --strategies twap_market twap_passive twap_random pov almgren_chriss_1e-06 --sizes 0.05 0.2 0.4 --horizons 500 2000 8000 --first-seed 1020 --seeds 20 --merge --workers $SLURM_NTASKS
