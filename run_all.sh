#!/usr/bin/env bash
# One-shot reproduction of Slice K on this machine (4x A6000).
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

echo "[1/7] venv + deps"
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install -q -r requirements.txt

echo "[2/7] download LongMemEval_S"
mkdir -p data/raw
[ -s data/raw/longmemeval_s.json ] || \
  curl -sL -o data/raw/longmemeval_s.json \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"

echo "[3/7] convert -> data/lme_s_k"
python -m mem27.planner.ref_data --raw data/raw/longmemeval_s.json --out data/lme_s_k

echo "[4/7] embed passages+queries on 4 GPUs"
if [ ! -f data/lme_s_k/emb/queries.npy ]; then
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g python -m mem27.planner.embed \
      --data data/lme_s_k --gpu $g --num-gpus 4 > logs/embed_gpu$g.log 2>&1 &
  done
  wait
fi

echo "[5/7] tune cascade on dev, then run all plans"
TAU=$(python -m mem27.planner.cascade_tune --data data/lme_s_k --split dev --k 5 \
        --success-on note --out runs/analysis/cascade_tune_note_k5.json | awk -F= '/CHOSEN_TAU/{print $2}')
python -m mem27.planner.cascade_tune --data data/lme_s_k --split dev --k 20 --success-on note \
        --out runs/analysis/cascade_tune_note_k20.json >/dev/null
python -m mem27.planner.run_plan --data data/lme_s_k --plans all \
        --tag lme_s_k_seed0_warm --workers 12 --a2-tau "${TAU:-0.05}" --a0-plan ref:p2

echo "[6/7] oracle (session/turn x k=5/20) + timing + failures"
for cfg in "session 5" "session 20" "note 5" "note 20"; do
  set -- $cfg
  python -m mem27.oracle --dataset data/lme_s_k --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
    --success-on $1 --split test --k $2 --out runs/analysis/oracle_${1}_test_k$2.json >/dev/null
  python -m mem27.oracle --dataset data/lme_s_k --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
    --success-on $1 --split dev --k $2 --out runs/analysis/oracle_${1}_dev_k$2.json >/dev/null
done
python -m mem27.planner.timing --data data/lme_s_k --n 160 --a2-tau "${TAU:-0.05}" \
        --out runs/analysis/timing.json >/dev/null
python -m mem27.planner.annotate_failures --data data/lme_s_k \
        --out runs/analysis/cascade_failures.json >/dev/null

echo "[7/7] build report"
python -m mem27.build_report
echo "DONE -> docs/midterm/K.md, runs/analysis/metrics.json"
