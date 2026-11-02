#!/bin/bash
# On the MAC: send the code and the experiment inputs to the cluster. Add -n for a dry run.
# Records the commit in .source_version first, because the cluster has no .git and every manifest cites it.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/settings.sh
dirty=false
[ -n "$(git status --porcelain -- order_book papers scripts)" ] && dirty=true
printf '{"commit": "%s", "dirty": %s}\n' "$(git rev-parse HEAD)" "$dirty" > .source_version
[ "$dirty" = true ] && echo "note: pushing uncommitted changes; manifests will say dirty=true"
rsync -avz --progress --filter="merge hpc/rsync.filter" "$@" ./ "$HPC_USER@$HPC_HOST:$HPC_REPO/"
