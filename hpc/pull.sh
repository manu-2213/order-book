#!/bin/bash
# On the MAC: fetch one paper's results, without the bulky per-episode logs unless asked.
#   bash hpc/pull.sh decomposition            # summaries and manifests
#   bash hpc/pull.sh decomposition --all      # also episodes.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/settings.sh
paper="${1:?usage: pull.sh <paper> [--all]}"
excludes=(--exclude='*_quick/' --exclude='*.pt')
[ "${2:-}" = "--all" ] || excludes+=(--exclude='*.jsonl')
mkdir -p "outputs/$paper"
rsync -avz --progress "${excludes[@]}" "$HPC_USER@$HPC_HOST:$HPC_REPO/outputs/$paper/" "outputs/$paper/"
