#!/bin/bash
# On the CLUSTER, after smoke_test printed ALL PASSED. Each job is an array of 1-hour tasks, one seed per task, followed by a merge:
#   bash submit_all.sh            -> the grids that give the paper's main tables
#   bash submit_all.sh extras     -> horizons, the deep-cover curve and replications
#   bash submit_all.sh all        -> the two above
#   bash submit_all.sh capital    -> the robustness check: the E8 grids with 10 times the capital, on seeds 1020-1039
#   bash submit_all.sh rl         -> train the executor: 8 variants, each a chain of CHAIN (default 4) 1-hour jobs that resume
#   bash submit_all.sh rl_results -> after rl: the policies against TWAP on unseen seeds, and their decomposition
set -euo pipefail
cd "$(dirname "$0")"
main=(e8_deep_cover e8_low_cover protocol detection_curve)
extras=(e8_horizons detection_curve_deep replications)
capital=(e8_capital_deep e8_capital_low e8_capital_horizons)
rl=(train_rl)
rl_results=(rl_eval e8_rl)
case "${1:-main}" in
  main) jobs=("${main[@]}") ;;
  extras) jobs=("${extras[@]}") ;;
  all) jobs=("${main[@]}" "${extras[@]}") ;;
  capital) jobs=("${capital[@]}") ;;
  rl) export CHAIN="${CHAIN:-4}"; jobs=("${rl[@]}") ;;
  rl_results) jobs=("${rl_results[@]}") ;;
  *) echo "unknown group $1"; exit 1 ;;
esac
bash ../../../hpc/submit.sh decomposition "${jobs[@]}"
