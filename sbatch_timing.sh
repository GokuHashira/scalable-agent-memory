#!/usr/bin/env bash
#SBATCH --job-name=k_timing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/timing_sbatch_%j.out
# Timing under a dedicated Slurm allocation (planner.md §3: run timings under srun/sbatch).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"  # submit from the repo root
source env.sh
python -m mem27.planner.timing --data data/lme_s_k --n 160 --a2-tau 0.05 \
    --out runs/analysis/timing_srun.json
echo "SBATCH_TIMING_DONE"
