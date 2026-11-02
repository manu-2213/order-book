#!/bin/bash
#SBATCH -J detection_curve_deep
#SBATCH -o logs/%x.o%A.%a
#SBATCH -e logs/%x.e%A.%a
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 4
#SBATCH --mem-per-cpu=1G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH -a 1-20
# The detection curve in the deep-cover market, one seed per task, 20 seeds, 2,100 episodes.
# Submit with hpc/submit.sh, which queues the merge after it. A task killed at the hour keeps its finished runs; the merge prints what to resubmit.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.detection_curve --out detection_curve_deep --regime deep_cover --seeds 20 --shard $SLURM_ARRAY_TASK_ID --workers $SLURM_NTASKS
