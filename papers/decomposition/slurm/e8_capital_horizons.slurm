#!/bin/bash
#SBATCH -J e8_capital_horizons
#SBATCH -o logs/%x.o%A.%a
#SBATCH -e logs/%x.e%A.%a
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=1G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH -a 1-20
# Robustness: E8 with speculators and the noise pool holding 10 times as much. 5 executors x 3 sizes x 3 horizons x 20 worlds, deep cover, one seed per task, seeds 1020-1039, 18,000 runs.
# Submit with hpc/submit.sh, which queues the merge after it. A task killed at the hour keeps its finished runs; the merge prints what to resubmit.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_capital_horizons --regime deep_cover_capital10 --strategies twap_market twap_passive twap_random pov almgren_chriss_1e-06 --sizes 0.05 0.2 0.4 --horizons 500 2000 8000 --first-seed 1020 --seeds 20 --shard $SLURM_ARRAY_TASK_ID --workers $SLURM_NTASKS
