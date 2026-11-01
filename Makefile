.PHONY: fixture test check coverage baseline reproduce \
	planner-all planner-data planner-embed planner-run planner-oracle planner-tune \
	planner-timing planner-failures planner-report planner-clean

SHELL := /bin/bash
DS ?= lme_s_raw
PY := uv run python
CONFIG := configs/retrieval.yaml

fixture:  ## build data/fixture/ if absent (deterministic, seed 0)
	$(PY) scripts/make_fixture.py --n 5000 --seed 0 --out data/fixture

test:
	uv run pytest -q

check:  ## the local quality gate (constitution: merge gate)
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check src
	uv run pytest -q

coverage:
	uv run coverage run -m pytest -q
	uv run coverage report

baseline:  ## cluster only: build, run P0-P5 warm, P0/P2 cold, evaluate
	$(PY) scripts/build_indexes.py --dataset data/$(DS) --plans bm25,dense,summary_first,links --config $(CONFIG)
	for p in bm25 dense hybrid summary_first one_hop bounded; do $(PY) scripts/run_plan.py --plan $$p --dataset data/$(DS) --config $(CONFIG) --seed 0 --cache warm; done
	for p in bm25 hybrid; do $(PY) scripts/run_plan.py --plan $$p --dataset data/$(DS) --config $(CONFIG) --seed 0 --cache cold --no-warmup; done
	$(PY) scripts/evaluate.py --runs 'runs/*-$(DS)-*' --questions data/$(DS)/questions.jsonl

reproduce:  ## regenerate the midterm tables from existing run directories
	$(PY) scripts/evaluate.py --runs 'runs/*-$(DS)-*' --questions data/$(DS)/questions.jsonl
	$(PY) scripts/profile_indexes.py --dataset data/$(DS)

# ---- Planner, oracle and headroom experiments (GPU machine; source env.sh first) ----
PLANNER_PY := source env.sh && python
PLANNER_DS := data/lme_s_k

planner-all:  ## one-shot reproduction of the planner experiments
	bash run_all.sh

planner-data:
	$(PLANNER_PY) -m mem27.planner.ref_data --raw data/raw/longmemeval_s.json --out $(PLANNER_DS)

planner-embed:
	@for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$$g $(PLANNER_PY) -m mem27.planner.embed \
		--data $(PLANNER_DS) --gpu $$g --num-gpus 4 > logs/embed_gpu$$g.log 2>&1 & done; wait

planner-run:
	$(PLANNER_PY) -m mem27.planner.run_plan --data $(PLANNER_DS) --plans all \
		--tag lme_s_k_seed0_warm --workers 12 --a2-tau 0.05 --a0-plan ref:p2

planner-oracle:
	$(PLANNER_PY) -m mem27.oracle --dataset $(PLANNER_DS) --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
		--success-on note --split test --k 5 --out runs/analysis/oracle_note_test_k5.json

planner-tune:
	$(PLANNER_PY) -m mem27.planner.cascade_tune --data $(PLANNER_DS) --split dev --k 5 --success-on note

planner-timing:
	$(PLANNER_PY) -m mem27.planner.timing --data $(PLANNER_DS) --n 160 --out runs/analysis/timing.json

planner-failures:
	$(PLANNER_PY) -m mem27.planner.annotate_failures --data $(PLANNER_DS) --out runs/analysis/cascade_failures.json

planner-report:
	$(PLANNER_PY) -m mem27.build_report

planner-clean:
	rm -rf runs/*_smoke
