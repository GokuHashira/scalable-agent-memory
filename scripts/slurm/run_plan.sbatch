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
PLAN=${1:?plan p0..p5}
DS=${2:?dataset name}
CACHE=${3:-warm}
shift 3 || true
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HOME=${HF_HOME:-$PWD/.cache}
# caches are filled on the login node first (CR step 1); compute nodes may have no internet
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$PWD/.cache/tiktoken} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXTRA=()
if [[ "${CACHE}" == "cold" ]]; then EXTRA+=(--no-warmup); fi
uv run --extra encode python scripts/run_plan.py --plan "${PLAN}" --dataset "data/${DS}" --encoder bge \
  --seed 0 --cache "${CACHE}" "${EXTRA[@]}" "$@"
