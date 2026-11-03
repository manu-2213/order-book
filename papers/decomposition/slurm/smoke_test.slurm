#!/bin/bash
#SBATCH -J smoke_test
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute,computeshort
#SBATCH -N 1                    # the workers are processes on one machine, so all tasks must share a node
#SBATCH -n 8
#SBATCH --mem-per-cpu=2G         # a worker peaks near 200 MB
#SBATCH -t 1:0:0
#SBATCH --mail-type=END,FAIL    # the email reports CPU and memory efficiency

# Run first. The suite, then each experiment script end to end at tiny sizes, plus real speed on this machine.
# Quick runs write to *_quick folders, so they can never overwrite results.
# Look for ALL PASSED in logs/smoke_test.o<id>; hpc/submit.sh refuses real jobs without it.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR/../../.."
source hpc/env.sh
set -u
python -c "import numpy, platform; print(platform.platform(), 'numpy', numpy.__version__, flush=True)"
cat .source_version 2>/dev/null || echo "no .source_version: synced without hpc/push.sh, so manifests will not name the commit"

# On its own, because a failure here alone means results differ across machines rather than broken code.
if python -m pytest -q tests/test_simulator_version.py; then
    echo "FINGERPRINT MATCHES: this machine reproduces the Mac's runs bit for bit"
else
    echo "FINGERPRINT DIFFERS: results here are valid but not byte-comparable with the Mac's; do not mix the two"
fi
python -m pytest -q --deselect tests/test_simulator_version.py
python -u -m papers.decomposition.scripts.decompose --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.detection_curve --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.crn_variance --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.replications --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.runaway_causes --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.market_probe --quick --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.rl_ceiling --quick --workers $SLURM_NTASKS > /dev/null
# The training path: a few iterations, a checkpoint, then the evaluation of what it saved.
python -u -m papers.decomposition.scripts.train_rl --quick --name smoke --workers $SLURM_NTASKS > /dev/null
python -u -m papers.decomposition.scripts.rl_eval --quick --policies outputs/decomposition/rl_quick/smoke/best_policy.pt --workers $SLURM_NTASKS > /dev/null
# The path the real jobs take: two shards, then the merge. Shards always resume, so start from nothing.
rm -rf outputs/decomposition/smoke_shards_quick
for shard in 1 2; do
    python -u -m papers.decomposition.scripts.decompose --quick --out smoke_shards --shard $shard --workers $SLURM_NTASKS > /dev/null
done
python -u -m papers.decomposition.scripts.decompose --quick --out smoke_shards --merge --workers $SLURM_NTASKS > /dev/null
python -m scripts.benchmark | sed -n 1,5p
echo "ALL PASSED"
