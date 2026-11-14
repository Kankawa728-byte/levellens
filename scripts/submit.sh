#!/bin/bash
# C3 and C4: submit one of the four job templates and write the submission to the run log.
#
#   scripts/submit.sh setup                                         build the environment (CPU)
#   scripts/submit.sh fetch --all                                   splits and weights (CPU)
#   scripts/submit.sh run data=arc model=distilbert head=ce         one run (GPU)
#   scripts/submit.sh grid configs/grid/tier1_transformers.txt 1-5  five runs in one job (GPU)
#   scripts/submit.sh tune data=arc model=distilbert head=ce n_trials=12   Optuna search (GPU)
#
#   -t TIME      wall-clock limit; the default comes from scripts/site.env
#   -g GRES      the GPU request, for example gpu:1; the default comes from scripts/site.env
#   -r FOLDER    write the result rows to this folder instead of results/ (for try-outs)
#   -n NAME      job name, and the name in the log
#   -d JOBID     start only after that job has finished without an error
#   -x           print the sbatch command and stop
#   -h           this help
#
# Everything the scheduler needs beyond this is in scripts/site.env, which git ignores. The log is
# logs/submissions.tsv; `python scripts/runlog.py` turns it into the table of docs/compute.md.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f scripts/site.env ] && . scripts/site.env
LOG=logs/submissions.tsv

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

TIME="" GRES="" RESULTS="" NAME="" AFTER="" DRY=0
while getopts ":t:g:r:n:d:xh" option; do
  case $option in
    t) TIME=$OPTARG ;;
    g) GRES=$OPTARG ;;
    r) RESULTS=$OPTARG ;;
    n) NAME=$OPTARG ;;
    d) AFTER=$OPTARG ;;
    x) DRY=1 ;;
    h) usage; exit 0 ;;
    *) echo "unknown option -$OPTARG"; usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

KIND=${1:-}
shift $(( $# > 0 ? 1 : 0 ))
case $KIND in
  setup) SCRIPT=scripts/setup.sbatch; NEEDS_GPU=0; TIME=${TIME:-03:00:00} ;;
  fetch) SCRIPT=scripts/fetch.sbatch; NEEDS_GPU=0; TIME=${TIME:-06:00:00} ;;
  run)   SCRIPT=scripts/job.sbatch;   NEEDS_GPU=1; TIME=${TIME:-${LEVELLENS_JOB_TIME:-01:00:00}} ;;
  grid)  SCRIPT=scripts/grid.sbatch;  NEEDS_GPU=1; TIME=${TIME:-${LEVELLENS_GRID_TIME:-06:00:00}} ;;
  tune)  SCRIPT=scripts/tune.sbatch;  NEEDS_GPU=1; TIME=${TIME:-${LEVELLENS_TUNE_TIME:-06:00:00}} ;;
  *)     echo "First argument: setup, fetch, run, grid or tune."; echo; usage; exit 2 ;;
esac

OPTIONS=(--time="$TIME" --job-name="${NAME:-levellens-$KIND}")
if [ "$NEEDS_GPU" = 1 ]; then
  GRES=${GRES:-${LEVELLENS_GRES:-}}
  [ -n "$GRES" ] || {
    echo "No GPU request. Put LEVELLENS_GRES=gpu:1 in scripts/site.env, or pass -g."
    echo "Some clusters reject a request that does not name the model: -g gpu:<model>:1"
    exit 2
  }
  OPTIONS+=(--gres="$GRES")
fi
[ -n "$AFTER" ] && OPTIONS+=(--dependency=afterok:"$AFTER")
[ -n "$RESULTS" ] && { mkdir -p "$RESULTS"; OPTIONS+=(--export=ALL,LEVELLENS_RESULTS="$(cd "$RESULTS" && pwd)"); }
# shellcheck disable=SC2206,SC2086
[ -n "${LEVELLENS_SBATCH_EXTRA:-}" ] && OPTIONS+=(${LEVELLENS_SBATCH_EXTRA})

mkdir -p logs
if [ "$DRY" = 1 ]; then
  echo "sbatch ${OPTIONS[*]} $SCRIPT $*"
  exit 0
fi

OUTPUT=$(sbatch --parsable "${OPTIONS[@]}" "$SCRIPT" "$@" 2>&1)
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "sbatch refused the job: $OUTPUT"
  echo "The command was: sbatch ${OPTIONS[*]} $SCRIPT $*"
  exit "$STATUS"
fi
JOB_ID=${OUTPUT%%;*}

[ -s "$LOG" ] || printf 'submitted\tkind\tjob_id\tname\tgres\ttime\targuments\n' > "$LOG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date '+%F %T')" "$KIND" "$JOB_ID" "${NAME:-levellens-$KIND}" "${GRES:--}" "$TIME" "$*" >> "$LOG"
echo "Submitted job $JOB_ID: $KIND $*"
echo "Log: $LOG    Output: logs/${NAME:-levellens-$KIND}-$JOB_ID.out"
