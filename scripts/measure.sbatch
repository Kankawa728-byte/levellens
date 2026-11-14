#!/bin/bash
# C6: measure how long one question takes, then rebuild the efficiency table.
#
#   scripts/submit.sh -r results measure cpu  <run ids>    seconds per question on a CPU (R7)
#   scripts/submit.sh -r results measure gpu  <run ids>    the same models on the GPU
#
# The first argument is `cpu` or `gpu`; the rest are run ids whose model was saved. With no run ids,
# both modes take the best saved model of every dataset, by dev QWK.
#
# cpu runs `python -m levellens.export`, which does two things: it checks that the Predictor, loaded
# from the saved folder alone, gives the probabilities the training run stored, and it times one
# question at a time on the CPU. That timing is what specification R7 asks for.
#
# gpu runs `python scripts/latency.py --device cuda`, the other half of C6.
#
# Pass `-r results` so that the reports land in the results store of the repository, next to the
# rows they describe, and not in the scratch store of the machine.
#SBATCH --job-name=levellens-measure
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
source scripts/job_setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source scripts/stage_cache

MODE=${1:-}
shift $(( $# > 0 ? 1 : 0 ))
case $MODE in
  cpu)
    python scripts/env_report.py
    echo "##### export start $(date '+%F %T'): ${*:-the best saved model of every dataset}"
    LEVELLENS_DEVICE=cpu python -m levellens.export "$@"
    rc=$? ;;
  gpu)
    python scripts/env_report.py --require-gpu || exit 1
    echo "##### latency start $(date '+%F %T'): ${*:-the best saved model of every dataset}"
    python scripts/latency.py --device cuda "$@"
    rc=$? ;;
  *)
    echo "JOB FAILED: first argument must be cpu or gpu"; exit 2 ;;
esac

echo "##### efficiency table"
python scripts/efficiency.py
echo "##### measure done rc=$rc $(date '+%F %T')"
exit $rc
