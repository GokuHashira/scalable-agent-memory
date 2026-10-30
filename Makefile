.PHONY: all data embed run oracle tune timing failures report test clean
SHELL := /bin/bash
PY := source env.sh && python

all: ; bash run_all.sh

data: ; $(PY) -m mem27.planner.ref_data --raw data/raw/longmemeval_s.json --out data/lme_s_k

embed: ; @for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$$g $(PY) -m mem27.planner.embed \
	--data data/lme_s_k --gpu $$g --num-gpus 4 > logs/embed_gpu$$g.log 2>&1 & done; wait

run: ; $(PY) -m mem27.planner.run_plan --data data/lme_s_k --plans all \
	--tag lme_s_k_seed0_warm --workers 12 --a2-tau 0.05 --a0-plan ref:p2

oracle: ; $(PY) -m mem27.oracle --dataset data/lme_s_k --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
	--success-on note --split test --k 5 --out runs/analysis/oracle_note_test_k5.json

tune: ; $(PY) -m mem27.planner.cascade_tune --data data/lme_s_k --split dev --k 5 --success-on note

timing: ; $(PY) -m mem27.planner.timing --data data/lme_s_k --n 160 --out runs/analysis/timing.json

failures: ; $(PY) -m mem27.planner.annotate_failures --data data/lme_s_k --out runs/analysis/cascade_failures.json

report: ; $(PY) -m mem27.build_report

test: ; $(PY) -m pytest tests/ -q

clean: ; rm -rf runs/*_smoke
