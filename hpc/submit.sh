#!/bin/bash
# On the CLUSTER, from anywhere. Submits named jobs of one paper with the settings in settings.sh:
#   bash hpc/submit.sh decomposition smoke_test
#   bash hpc/submit.sh decomposition e8_deep_cover protocol
#   ARRAY=3,7 bash hpc/submit.sh decomposition e8_deep_cover     # only those tasks, then the merge again
#   CHAIN=4 bash hpc/submit.sh decomposition train_rl             # the same job 4 times in a row, each resuming the last: 4 hours of training
# Use this rather than sbatch: the logs/ folder must exist first or a job dies without a log, and it refuses
# real jobs until that paper's smoke test has passed (FORCE=1 overrides).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
paper="${1:?usage: submit.sh <paper> <job> [<job> ...]}"; shift
cd "$repo/papers/$paper/slurm"
mkdir -p logs
for job in "$@"; do
    [ -f "$job.slurm" ] || { echo "no such job: papers/$paper/slurm/$job.slurm"; exit 1; }
    if [ "$job" != smoke_test ] && [ "${FORCE:-0}" != 1 ] && ! grep -qs "^ALL PASSED" logs/smoke_test.o*; then
        echo "refusing $job: no passed smoke test in papers/$paper/slurm/logs (run: bash hpc/submit.sh $paper smoke_test)"; exit 1
    fi
    id=""
    for link in $(seq 1 "${CHAIN:-1}"); do
        id=$(sbatch --parsable ${ARRAY:+--array="$ARRAY"} ${id:+--dependency=afterany:"$id"} "$job.slurm")
        echo "$job${CHAIN:+ (link $link of $CHAIN)}: $id"
    done
    if [ -f "${job}_merge.slurm" ]; then
        echo "${job}_merge: $(sbatch --parsable --dependency=afterany:"$id" "${job}_merge.slurm")"
    fi
done
