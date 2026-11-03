#!/bin/bash
#SBATCH -J e8_low_cover
#SBATCH -o logs/%x.o%A.%a
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 4
#SBATCH --mem-per-cpu=1G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH -a 1-20
# E8 in the low-cover market with the learned executors: 14 x 5 sizes x 20 worlds, one seed per task, 20 seeds, 28,000 runs.
# Submit with hpc/submit.sh, which queues the merge after it. A task killed at the hour keeps its finished runs; the merge prints what to resubmit.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_low_cover --regime low_cover --strategies immediate twap_market twap_passive twap_random pov almgren_chriss_1e-07 almgren_chriss_1e-06 almgren_chriss_1e-05 obizhaeva_wang limit_market footprint mimic rl@outputs/stealth_execution/rl/sweep/lr1e-3_ent0.003_ep4_sz0.20/policy.pt rl@outputs/stealth_execution/rl/sweep/lr1e-3_ent0.01_ep10_sz0.40/best_policy.pt --seeds 20 --shard $SLURM_ARRAY_TASK_ID --workers $SLURM_NTASKS
