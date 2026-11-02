#!/bin/bash
#SBATCH -J replications_merge
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # tells you the whole grid has finished, and how efficiently this step ran
# Combines the shards of replications and writes the summary. Fails, naming the tasks to resubmit, if any run is missing.
# hpc/submit.sh queues this after the array.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.replications --seeds 20 --merge --workers $SLURM_NTASKS
