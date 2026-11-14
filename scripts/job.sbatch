#!/bin/bash
# C3: one GPU run of the project. Everything after the script name is passed to `levellens.run`.
#
#   scripts/submit.sh run data=arc model=distilbert head=ce seed=0 owner=E save_model=true
#   sbatch --gres=gpu:1 --time=01:00:00 scripts/job.sbatch data=arc model=distilbert head=ce
#
# The wrapper is the usual way in: it reads the GPU request and the time limit from
# scripts/site.env and writes the submission to the run log. The plain sbatch line shows what it
# does. No #SBATCH line here asks for a GPU, because the way to ask differs between clusters.
#
# Every fine-tune stops itself after `model.params.max_minutes` (50 by default), so one run and
# its scoring fit in the hour this job asks for. Specification R6.
#SBATCH --job-name=levellens-run
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source scripts/job_setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the weights are in the cache; see scripts/download_data.py
source scripts/stage_cache

python scripts/env_report.py --require-gpu || exit 1
[ "$#" -gt 0 ] || { echo "JOB FAILED: no overrides given, for example data=arc model=distilbert head=ce"; exit 2; }

echo "##### run start $(date '+%F %T'): $*"
python -m levellens.run "$@"
rc=$?
echo "##### run done rc=$rc $(date '+%F %T')"
exit $rc
