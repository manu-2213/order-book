#!/bin/bash
#SBATCH -J e8_low_cover_merge
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # tells you the whole grid has finished, and how efficiently this step ran
# Combines the shards of e8_low_cover and writes the summary. Fails, naming the tasks to resubmit, if any run is missing.
# hpc/submit.sh queues this after the array.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_low_cover --regime low_cover --strategies immediate twap_market twap_passive twap_random pov almgren_chriss_1e-07 almgren_chriss_1e-06 almgren_chriss_1e-05 obizhaeva_wang limit_market footprint mimic rl@outputs/stealth_execution/rl/sweep/lr1e-3_ent0.003_ep4_sz0.20/policy.pt rl@outputs/stealth_execution/rl/sweep/lr1e-3_ent0.01_ep10_sz0.40/best_policy.pt --seeds 20 --merge --workers $SLURM_NTASKS
