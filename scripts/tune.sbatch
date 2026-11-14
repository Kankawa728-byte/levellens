#!/bin/bash
# C4: the Optuna search, as its own GPU job.
#
#   scripts/submit.sh tune data=arc model=distilbert head=ce owner=E n_trials=12
#
# It is a job of its own, and only one at a time, because `levellens.tune` keeps the study in a
# single SQLite file in the cache. Two jobs on the same study would fight over it. The search ends
# with the best setting trained once more as a normal run, so the job writes a result row like any
# other. `--dependency=singleton` makes the scheduler queue a second job of the same name instead of
# starting it next to the first.
#SBATCH --job-name=levellens-tune
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --dependency=singleton
source scripts/job_setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source scripts/stage_cache

python scripts/env_report.py --require-gpu || exit 1
[ "$#" -gt 0 ] || { echo "JOB FAILED: no arguments, for example data=arc model=distilbert head=ce n_trials=12"; exit 2; }

echo "##### tune start $(date '+%F %T'): $*"
python -m levellens.tune "$@"
rc=$?
echo "##### tune done rc=$rc $(date '+%F %T')"
exit $rc
