#!/bin/bash
#SBATCH --job-name=mem27-run
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=standard
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=cmsc848n-class
#SBATCH --mail-user=gsakthiv@umd.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=2:00:00
#SBATCH --array=0-2
# Run plans as a job array, one task per plan; timings are only valid from compute nodes.
# Usage: sbatch scripts/slurm/run_plan.sbatch PLANS DS CACHE   (e.g. bm25,dense,hybrid lme_s warm)
# The array size must match the number of plans (default 0-2 = three plans); override with --array=0-N.

set -euo pipefail
IFS=',' read -r -a PLANS <<< "${1:?comma-separated plan names, e.g. bm25,dense,hybrid}"
DS=${2:?dataset name, e.g. lme_s}
CACHE=${3:-warm}
PLAN=${PLANS[$SLURM_ARRAY_TASK_ID]:?no plan for array task ${SLURM_ARRAY_TASK_ID}}

source "${SLURM_SUBMIT_DIR}/scripts/slurm/zaratan_env.sh"   # caches under scratch; refuses to run outside it
eval "$(micromamba shell hook --shell bash)"
export MAMBA_ROOT_PREFIX=${SCRATCH_ROOT}/micromamba
micromamba activate "${SCRATCH_ROOT}/envs/mem27"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # single-threaded query path for stable latency

echo "task=${SLURM_ARRAY_TASK_ID} plan=${PLAN} dataset=${DS} cache=${CACHE}"
echo "host=$(hostname)"

EXTRA=()
if [[ "${CACHE}" == "cold" ]]; then EXTRA+=(--no-warmup); fi
uv run --extra encode python scripts/run_plan.py --plan "${PLAN}" --dataset "data/${DS}" --encoder model \
  --seed 0 --cache "${CACHE}" "${EXTRA[@]}"
