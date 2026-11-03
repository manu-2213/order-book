#!/bin/bash
# On the CLUSTER, after smoke_test printed ALL PASSED. Each grid is an array of 1-hour tasks, one seed per task, followed by a merge:
#   bash submit_all.sh final      -> everything the paper reads, in one go: the grids, the single jobs, the first-generation policies on
#                                    unseen seeds, then training and, queued behind it, the policies' scores and decomposition
#   bash submit_all.sh            -> the grids that give the paper's main tables
#   bash submit_all.sh extras     -> horizons, the deep-cover curve and replications
#   bash submit_all.sh all        -> the two above
#   bash submit_all.sh capital    -> the robustness check: the E8 grids with 10 times the capital, on seeds 1020-1039
#   bash submit_all.sh singles    -> the runaway interventions and the market probe
#   bash submit_all.sh rl         -> train the executor (10 variants, each a chain of CHAIN, default 4, 1-hour jobs that resume), then, queued
#                                    behind the last link: the policies against TWAP on unseen seeds (rl_eval) and their decomposition (e8_rl)
#   bash submit_all.sh rl_results -> rl_eval and e8_rl again, now, from the policies already trained
set -euo pipefail
cd "$(dirname "$0")"
submit="../../../hpc/submit.sh"
main=(e8_deep_cover e8_low_cover protocol detection_curve)
extras=(e8_horizons detection_curve_deep replications)
capital=(e8_capital_deep e8_capital_low e8_capital_horizons)
singles=(runaway_causes market_probe)
rl=(train_rl)
rl_results=(rl_eval e8_rl)

train_then_results() {
    # the results jobs read best_policy.pt, so they wait for every task of the last link; submit.sh prints "<job> (link n of N): <id>"
    local last
    last=$(CHAIN="${CHAIN:-4}" bash "$submit" decomposition "${rl[@]}" | tee /dev/stderr | tail -n 1 | awk '{print $NF}')
    AFTER="$last" CHAIN=1 bash "$submit" decomposition "${rl_results[@]}"
}

case "${1:-main}" in
  main) jobs=("${main[@]}") ;;
  extras) jobs=("${extras[@]}") ;;
  all) jobs=("${main[@]}" "${extras[@]}") ;;
  capital) jobs=("${capital[@]}") ;;
  singles) jobs=("${singles[@]}") ;;
  rl) train_then_results; exit ;;
  rl_results) jobs=("${rl_results[@]}") ;;
  final)
    bash "$submit" decomposition "${main[@]}" "${extras[@]}" "${capital[@]}" "${singles[@]}"
    FORCE=1 bash "$submit" stealth_execution rl_heldout   # the first-generation policies on unseen seeds; this paper's smoke test covers the simulator
    train_then_results
    exit ;;
  *) echo "unknown group $1"; exit 1 ;;
esac
bash "$submit" decomposition "${jobs[@]}"
