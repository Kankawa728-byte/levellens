#!/bin/bash
# C3: a batch of GPU runs from one grid file, one after the other, in a single job.
#
#   scripts/submit.sh grid configs/grid/tier1_transformers.txt 1-5
#   sbatch --gres=gpu:1 --time=06:00:00 scripts/grid.sbatch configs/grid/tier1_transformers.txt 1-5
#
# The first argument is the grid file, the second the lines to run: `1-5`, `3,7`, one number, or
# `all`. Anything after that is added to every run, for example `device=cuda:1`.
#
# One job per run (scripts/job.sbatch) gives the shortest queue wait and is the template to prefer.
# Use this one when the scheduler lets you run only a few jobs at a time: a slice of five runs
# waits in the queue once instead of five times. A run that fails does not stop the others;
# `levellens.grid` lists the failed lines at the end and the job exits 1.
#SBATCH --job-name=levellens-grid
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source scripts/job_setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the weights are in the cache; see scripts/download_data.py
source scripts/stage_cache

python scripts/env_report.py --require-gpu || exit 1
GRID=${1:-}
LINES=${2:-all}
[ -f "$GRID" ] || { echo "JOB FAILED: no grid file given. First argument, e.g. configs/grid/tier1_transformers.txt"; exit 2; }
shift $(( $# > 2 ? 2 : $# ))

# `1-5` means the lines 1 to 5; `3,7` and `4` are passed on as they are; `all` runs the whole file.
case $LINES in
  all) ONLY=() ;;
  *-*) ONLY=(--only "$(seq -s, "${LINES%-*}" "${LINES#*-}")") ;;
  *)   ONLY=(--only "$LINES") ;;
esac

echo "##### grid start $(date '+%F %T'): $GRID lines $LINES ${*:+extra: $*}"
python -m levellens.grid "$GRID" "${ONLY[@]}" --list "$@"
python -m levellens.grid "$GRID" "${ONLY[@]}" "$@"
rc=$?
echo "##### grid done rc=$rc $(date '+%F %T')"
exit $rc
