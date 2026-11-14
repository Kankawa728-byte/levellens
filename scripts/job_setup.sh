# What every job script does before it runs anything: find the repository, read the site settings,
# put the large folders where they belong, load the modules and activate the Python environment.
#
#   source scripts/job_setup.sh      # from a job script; it changes the working directory
#
# Nothing here is specific to one machine. The machine is described in scripts/site.env, which
# git ignores; see scripts/site.env.example.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEVELLENS_REPO=$PWD
[ -f scripts/site.env ] && . scripts/site.env

# The three folders of levellens/paths.py, plus the Hugging Face cache. A folder that is already
# set wins, so a single job can be redirected with `sbatch --export`.
if [ -n "${LEVELLENS_PROJECT_DIR:-}" ]; then
  export LEVELLENS_DATA="${LEVELLENS_DATA:-$LEVELLENS_PROJECT_DIR/data}"
  export LEVELLENS_CACHE="${LEVELLENS_CACHE:-$LEVELLENS_PROJECT_DIR/cache}"
  export LEVELLENS_RESULTS="${LEVELLENS_RESULTS:-$LEVELLENS_PROJECT_DIR/results}"
  export HF_HOME="${HF_HOME:-$LEVELLENS_PROJECT_DIR/hf}"
  # The temporary and cache folders are set, not defaulted: the job inherits a TMPDIR of /tmp and an
  # XDG_CACHE_HOME in the home folder, and both can be small volumes with a quota. Moving them is the
  # whole point of naming a project directory. LEVELLENS_TMPDIR overrides it on purpose.
  export TMPDIR="${LEVELLENS_TMPDIR:-$LEVELLENS_PROJECT_DIR/tmp}/${SLURM_JOB_ID:-local}"
  export PIP_CACHE_DIR="$LEVELLENS_PROJECT_DIR/cache/pip"
  export XDG_CACHE_HOME="$LEVELLENS_PROJECT_DIR/cache/xdg"
  export TRITON_CACHE_DIR="$LEVELLENS_PROJECT_DIR/cache/triton"
  export CONDA_PKGS_DIRS="$LEVELLENS_PROJECT_DIR/cache/conda-pkgs"
  mkdir -p "$LEVELLENS_DATA" "$LEVELLENS_CACHE" "$LEVELLENS_RESULTS" "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
fi
mkdir -p logs

# One thread per core the scheduler gave this job: more threads than cores only slows it down.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1   # so that a log shows the progress of a run while it is still going

if [ -n "${LEVELLENS_MODULES:-}" ]; then
  if ! type module >/dev/null 2>&1; then
    for _init in /etc/profile.d/*lmod*.sh /etc/profile.d/*module*.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$_init" ] && . "$_init" && break
    done
    unset _init
  fi
  # shellcheck disable=SC2086
  module load $LEVELLENS_MODULES || { echo "JOB SETUP FAILED: cannot load modules: $LEVELLENS_MODULES"; exit 1; }
fi

# pip falls back to a user install, into the home folder, when it cannot write to the site-packages
# of the Python it is run with. That silently fills a home quota, so switch the fallback off.
export PIP_USER=0 PYTHONNOUSERSITE=1

# scripts/setup.sbatch sets LEVELLENS_SETUP_MODE, because it is the job that builds the
# environment: there, a missing environment is the starting point and not a failure.
export LEVELLENS_ACTIVATED=0
if [ -n "${LEVELLENS_ACTIVATE:-}" ]; then
  if eval "$LEVELLENS_ACTIVATE"; then
    export LEVELLENS_ACTIVATED=1
  else
    echo "cannot activate the environment: $LEVELLENS_ACTIVATE"
    [ "${LEVELLENS_SETUP_MODE:-0}" = 1 ] || { echo "JOB SETUP FAILED"; exit 1; }
  fi
else
  export LEVELLENS_ACTIVATED=1   # nothing to activate: whatever python is on PATH is meant to be used
fi

# From the repository root, `import levellens` succeeds with any Python, because levellens/ is a
# folder right there. Ask from somewhere else, so that this really tests `pip install -e .`.
( cd / && python -c "import levellens" ) 2>/dev/null || {
  echo "this Python cannot import an installed levellens; python: $(command -v python || echo none)"
  [ "${LEVELLENS_SETUP_MODE:-0}" = 1 ] || {
    echo "JOB SETUP FAILED. Build the environment first: scripts/submit.sh setup"
    exit 1
  }
}

echo "[job] repository  $LEVELLENS_REPO"
echo "[job] host        $(hostname)${SLURM_JOB_ID:+, Slurm job $SLURM_JOB_ID}"
echo "[job] python      $(command -v python)"
echo "[job] data        ${LEVELLENS_DATA:-<repo>/data}"
echo "[job] cache       ${LEVELLENS_CACHE:-<repo>/cache}"
echo "[job] results     ${LEVELLENS_RESULTS:-<repo>/results}"
echo "[job] hugging face${HF_HOME:+ $HF_HOME}"
echo "[job] tmpdir      ${TMPDIR:-<not set>}"
