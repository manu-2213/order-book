#!/bin/bash
#SBATCH -J e8_rl_merge
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL
# Combines the shards of e8_rl and writes the summary. Fails, naming the tasks to resubmit, if any run is missing.
# hpc/submit.sh queues this after the array.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_rl --regime low_cover_capital10 --strategies twap_market twap_passive rl@outputs/decomposition/rl/ppo/best_policy.pt rl@outputs/decomposition/rl/grpo/best_policy.pt rl@outputs/decomposition/rl/ppo_net/best_policy.pt rl@outputs/decomposition/rl/ppo_detect0.03/best_policy.pt rl@outputs/decomposition/rl/grpo_detect0.03/best_policy.pt rl@outputs/decomposition/rl/ppo_mixed/best_policy.pt rl@outputs/decomposition/rl/grpo_mixed/best_policy.pt rl@outputs/decomposition/rl/ppo_seed1/best_policy.pt rl@outputs/decomposition/rl/grpo_seed1/best_policy.pt rl@outputs/decomposition/rl/grpo_g8/best_policy.pt --sizes 0.1 0.2 0.4 --first-seed 1020 --seeds 20 --merge --workers $SLURM_NTASKS
