#!/bin/bash
#SBATCH --job-name=mem27-build
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=cmsc848n-class
#SBATCH --mail-user=gsakthiv@umd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=4:00:00
# Build per-history indexes. Usage: sbatch scripts/slurm/build.sbatch DS COMPONENTS   (e.g. lme_s bm25,dense)

set -euo pipefail
DS=${1:?dataset name, e.g. lme_s}
COMPONENTS=${2:-bm25,dense}

source "${SLURM_SUBMIT_DIR}/scripts/slurm/zaratan_env.sh"   # caches under scratch; refuses to run outside it
eval "$(micromamba shell hook --shell bash)"
export MAMBA_ROOT_PREFIX=${SCRATCH_ROOT}/micromamba
micromamba activate "${SCRATCH_ROOT}/envs/mem27"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_NO_SYNC=1   # never re-sync the shared .venv from a job (concurrent array tasks raced on it); sync once on the login node
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "dataset=${DS} components=${COMPONENTS}"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python scripts/build_indexes.py --dataset "data/${DS}" --plans "${COMPONENTS}" --encoder model
