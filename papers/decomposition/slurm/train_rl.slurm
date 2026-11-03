#!/bin/bash
#SBATCH -J train_rl
#SBATCH -o logs/%x.o%A.%a
#SBATCH -e logs/%x.e%A.%a
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
#SBATCH -a 1-10
# Trains the executor: one task per row of SWEEP in papers/decomposition/scripts/train_rl.py. Each task stops with a checkpoint before the
# hour and resumes in the next link of the chain: about 9.6 s an iteration on 16 cores, so 1,000 iterations need 3 links and submit_all.sh rl
# chains 5 (CHAIN=5 bash hpc/submit.sh decomposition train_rl; a link that finds the run finished exits at once), then queues rl_eval and e8_rl behind the last. A link that finds its run complete exits at once.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.train_rl --sweep $SLURM_ARRAY_TASK_ID --workers $SLURM_NTASKS --resume --hours 0.9
