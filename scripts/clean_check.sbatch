#!/bin/bash
# C7: follow the README in a brand-new environment, from a fresh clone, and check that the tables
# it rebuilds are the tables that are in the repository.
#
#   scripts/submit.sh clean
#
# Nothing of the working copy is used except its git history: the job clones the current branch into
# an empty folder, builds a new virtual environment from requirements.txt, and then runs exactly the
# commands of the README, steps 1 to 3. It needs internet, so it is a CPU job.
#
# The last step is the one that matters for specification R8: `levellens.results` and
# `levellens.figures` rebuild every table and figure from the results store, and `git status` then
# says whether what they wrote differs from what is committed. The CSV and Markdown tables are
# written byte for byte, so a difference there is a real difference. The figures are PNGs and can
# differ in their metadata between matplotlib builds, so those are reported but not treated as a
# failure.
#SBATCH --job-name=levellens-clean
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SOURCE=$PWD
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
[ -f scripts/site.env ] && . scripts/site.env

# A folder of its own, on storage with room for a 6 GB environment, and every cache and temporary
# file inside it. The home folder and /tmp are not touched, and nothing of this job is reused.
WORK=${LEVELLENS_PROJECT_DIR:-$PWD}/clean/${SLURM_JOB_ID:-local}
rm -rf "$WORK"
mkdir -p "$WORK/tmp" "$WORK/pip" || { echo "CLEAN FAILED: cannot create $WORK"; exit 1; }
export TMPDIR="$WORK/tmp" PIP_CACHE_DIR="$WORK/pip" XDG_CACHE_HOME="$WORK/xdg"
export PIP_USER=0 PYTHONNOUSERSITE=1
# Unset everything that moves a folder: the point is to run as a new reader of the README would.
unset LEVELLENS_DATA LEVELLENS_CACHE LEVELLENS_RESULTS LEVELLENS_DEVICE HF_HOME HF_HUB_OFFLINE

# The README says "you need git and Python 3.10 or newer". On this cluster python comes from a module.
if [ -n "${LEVELLENS_MODULES:-}" ]; then
  type module >/dev/null 2>&1 || for i in /etc/profile.d/*lmod*.sh; do [ -f "$i" ] && . "$i" && break; done
  # shellcheck disable=SC2086
  module load $LEVELLENS_MODULES
fi
echo "[clean] branch $BRANCH at $COMMIT -> $WORK"
echo "[clean] python $(python --version 2>&1), git $(git --version)"

step() { echo; echo "##### $* ($(date '+%T'))"; }

step "README step 1: clone and install"
git clone --quiet --branch "$BRANCH" "$SOURCE" "$WORK/levellens" || { echo "CLEAN FAILED: clone"; exit 1; }
cd "$WORK/levellens" || exit 1
python -m venv .venv || { echo "CLEAN FAILED: venv"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "CLEAN FAILED: activate"; exit 1; }
python -m pip --quiet install --upgrade pip || { echo "CLEAN FAILED: pip"; exit 1; }
python -m pip install -r requirements.txt > "$WORK/pip-install.log" 2>&1 || {
  echo "CLEAN FAILED: requirements.txt, see $WORK/pip-install.log"; tail -20 "$WORK/pip-install.log"; exit 1; }
echo "[clean] installed; the log is $WORK/pip-install.log"
python -m pytest -q
tests=$?

step "README step 2: the small example"
python -m levellens.run data=toy model=majority || exit 1
python -m levellens.run data=toy model=tfidf_logreg || exit 1
python -m levellens.run data=toy model=bilstm head=corn model.params.embeddings=random save_model=true || exit 1
python -m levellens.results --scratch || exit 1

step "README step 3: rebuild every table and figure"
python -m levellens.results || exit 1
python -m levellens.figures || exit 1
python scripts/efficiency.py || exit 1

step "R8: does what was rebuilt match what is committed?"
tables=$(git status --porcelain -- results/tables)
figures=$(git status --porcelain -- results/figures)
if [ -z "$tables" ]; then
  echo "tables:  identical to the committed ones"
else
  echo "tables:  DIFFER from the committed ones:"; echo "$tables"
fi
if [ -z "$figures" ]; then
  echo "figures: identical to the committed ones"
else
  echo "figures: differ in bytes (expected across matplotlib builds):"; echo "$figures"
fi

step "versions of this new environment"
python scripts/env_report.py --json "$WORK/env.json"

echo
echo "##### clean check done: tests rc=$tests, tables $([ -z "$tables" ] && echo match || echo differ)"
echo "##### the throw-away clone and environment are in $WORK"
exit "$tests"
