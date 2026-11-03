#!/bin/bash
# On the CLUSTER, from anywhere. Finds what failed without reading the logs, and clears finished logs out of the way.
#   bash hpc/logs.sh                 # jobs of the last 3 days that did not complete, each with its log and last line
#   bash hpc/logs.sh --tidy          # moves every finished job's log to logs/old/, except the smoke test's, which submit.sh reads
# Nothing is deleted, and globs such as logs/train_rl.o* then see only the latest run.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = --tidy ]; then
    running=" $(squeue --me -h -o %F | sort -u | tr '\n' ' ')"
    moved=0
    for log in "$repo"/papers/*/slurm/logs/*.[oe][0-9]*; do
        [ -e "$log" ] || continue
        name="$(basename "$log")"
        case "$name" in smoke_test.*) continue ;; esac
        id="${name##*.[oe]}"
        id="${id%%.*}"
        case "$running" in *" $id "*) continue ;; esac  # still running or queued: its log stays
        mkdir -p "$(dirname "$log")/old"
        mv "$log" "$(dirname "$log")/old/"
        moved=$((moved + 1))
    done
    echo "moved $moved logs to logs/old/"
    exit 0
fi

sacct -X -n -P -S "$(date -d '3 days ago' +%F)" -s F,TO,CA,OOM,NF,BF -o JobID,JobName,State,Elapsed | while IFS='|' read -r id name state elapsed; do
    task="${id/_/.}"  # 123_4 in sacct is 123.4 in the log's name
    echo "$id $name $state after $elapsed"
    find "$repo"/papers/*/slurm/logs -name "$name.[oe]$task" -size +0 | while read -r log; do
        echo "    $log"
        echo "    $(tail -n 1 "$log")"
    done
done
