#!/bin/bash
# Run one plan on a compute node (timings are only valid from here). Usage: sbatch scripts/slurm/run_plan.sbatch PLAN DS CACHE [extra args]
#SBATCH --job-name=mem27-run
#SBATCH --account=cmsc848n-class
#SBATCH --partition=standard
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PLAN=${1:?plan name: bm25|dense|hybrid|summary_first|one_hop|bounded}
DS=${2:?dataset name}
CACHE=${3:-warm}
shift 3 || true
source "${SLURM_SUBMIT_DIR}/scripts/slurm/zaratan_env.sh"  # stays inside SCRATCH_ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# caches are filled on the login node first (CR step 1); compute nodes may have no internet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXTRA=()
if [[ "${CACHE}" == "cold" ]]; then EXTRA+=(--no-warmup); fi
uv run --extra encode python scripts/run_plan.py --plan "${PLAN}" --dataset "data/${DS}" --encoder model \
  --seed 0 --cache "${CACHE}" "${EXTRA[@]}" "$@"
