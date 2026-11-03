#!/bin/bash
#SBATCH -J e8_rl
#SBATCH -o logs/%x.o%A.%a
#SBATCH -e logs/%x.e%A.%a
#SBATCH -p compute,computeshort # a 1-hour limit qualifies for the short queue
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=1G
#SBATCH -t 1:0:0
#SBATCH -a 1-20
# The decomposition of every trained policy next to TWAP: 12 executors x 3 sizes x 20 worlds, one seed per task, seeds 1020-1039, 14,400 runs.
# Reads the policies, so submit_all.sh rl queues it after the training chain, with its merge.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u

python -u -m papers.decomposition.scripts.decompose --out e8_rl --regime low_cover_capital10 --strategies twap_market twap_passive rl@outputs/decomposition/rl/ppo/best_policy.pt rl@outputs/decomposition/rl/grpo/best_policy.pt rl@outputs/decomposition/rl/ppo_net/best_policy.pt rl@outputs/decomposition/rl/ppo_detect0.03/best_policy.pt rl@outputs/decomposition/rl/grpo_detect0.03/best_policy.pt rl@outputs/decomposition/rl/ppo_mixed/best_policy.pt rl@outputs/decomposition/rl/grpo_mixed/best_policy.pt rl@outputs/decomposition/rl/ppo_seed1/best_policy.pt rl@outputs/decomposition/rl/grpo_seed1/best_policy.pt rl@outputs/decomposition/rl/grpo_g8/best_policy.pt --sizes 0.1 0.2 0.4 --first-seed 1020 --seeds 20 --shard $SLURM_ARRAY_TASK_ID --workers $SLURM_NTASKS
