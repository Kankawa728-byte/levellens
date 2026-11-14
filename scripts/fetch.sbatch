#!/bin/bash
# C2: download the splits and the model weights, as a job on a CPU node. Needs internet.
#
#   scripts/submit.sh fetch            # splits, the two encoders and GloVe
#   scripts/submit.sh fetch --all      # the same, plus the 8 GB open LLM
#
# Afterwards the GPU jobs read `$LEVELLENS_DATA` and `$HF_HOME` with the hub switched off. The last
# step of this job is that offline check, so a failure shows here and not in a GPU job.
#SBATCH --job-name=levellens-fetch
# This is a CPU job, so it asks for cores. A GPU job does not: schedulers that hand out
# whole GPUs usually fix the cores per GPU themselves and reject a job that asks again.
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source scripts/job_setup.sh

echo "##### fetch start $(date '+%F %T'): ${*:-default targets}"
python scripts/download_data.py "$@"
rc=$?
echo "##### offline check"
python scripts/download_data.py --check || rc=1
echo "##### fetch done rc=$rc $(date '+%F %T')"
exit $rc
