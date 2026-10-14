#!/bin/bash
# Build per-history indexes on a GPU fraction (embeddings). Usage: sbatch scripts/slurm/build.sbatch DS PLANS
#SBATCH --job-name=mem27-build
#SBATCH --account=cmsc848n-class
#SBATCH --partition=gpu-a100_1g.5gb
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
DS=${1:?dataset name, e.g. lme_s}
PLANS=${2:-bm25,dense}
source "${SLURM_SUBMIT_DIR}/scripts/slurm/zaratan_env.sh"  # stays inside SCRATCH_ROOT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# caches are filled on the login node first (CR step 1); compute nodes may have no internet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --extra encode python scripts/build_indexes.py --dataset "data/${DS}" --plans "${PLANS}" --encoder model
