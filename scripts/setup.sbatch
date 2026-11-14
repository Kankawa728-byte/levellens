#!/bin/bash
# C1: build the Python environment, as a job on a CPU node.
#
#   scripts/submit.sh setup
#   sbatch --time=03:00:00 scripts/setup.sbatch
#
# A login node is for editing and submitting, not for installing: the install needs a lot of disk
# writes and several minutes of CPU. This job does it on a compute node instead. It needs internet.
#
# What it does: create the environment if `LEVELLENS_ACTIVATE` does not work yet, install
# `requirements.txt` (PyTorch first from `LEVELLENS_TORCH_INDEX`, when that is set, so that the
# build matches the GPU), run the tests and print the versions for `docs/compute.md`.
#SBATCH --job-name=levellens-setup
# This is a CPU job, so it asks for cores. A GPU job does not: schedulers that hand out
# whole GPUs usually fix the cores per GPU themselves and reject a job that asks again.
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
export LEVELLENS_SETUP_MODE=1
source scripts/job_setup.sh

if ! python -c "import levellens" 2>/dev/null; then
  if [ -n "${LEVELLENS_ENV_CREATE:-}" ]; then
    echo "##### creating the environment: $LEVELLENS_ENV_CREATE"
    eval "$LEVELLENS_ENV_CREATE" || { echo "SETUP FAILED: cannot create the environment"; exit 1; }
    eval "${LEVELLENS_ACTIVATE:-true}" || { echo "SETUP FAILED: cannot activate the new environment"; exit 1; }
  fi
fi
python -c "import sys; print('python', sys.version.split()[0], 'at', sys.executable)" \
  || { echo "SETUP FAILED: no usable python. Set LEVELLENS_ENV_CREATE in scripts/site.env"; exit 1; }

python -m pip install --upgrade pip setuptools wheel || { echo "SETUP FAILED: pip"; exit 1; }
if [ -n "${LEVELLENS_TORCH_INDEX:-}" ]; then
  echo "##### PyTorch from $LEVELLENS_TORCH_INDEX"
  python -m pip install torch --index-url "$LEVELLENS_TORCH_INDEX" || { echo "SETUP FAILED: torch"; exit 1; }
fi
python -m pip install -r requirements.txt || { echo "SETUP FAILED: requirements.txt"; exit 1; }

echo "##### versions"
python scripts/env_report.py --json "logs/env-setup-${SLURM_JOB_ID:-local}.json"
echo "##### tests"
python -m pytest -q
rc=$?
echo "##### setup done rc=$rc (two tests are skipped until the splits are downloaded)"
exit $rc
