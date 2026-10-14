.PHONY: fixture test check coverage baseline reproduce

DS ?= lme_s_g
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
	$(PY) scripts/build_indexes.py --dataset data/$(DS) --plans p0,p1,p3,links --config $(CONFIG)
	for p in p0 p1 p2 p3 p4 p5; do $(PY) scripts/run_plan.py --plan $$p --dataset data/$(DS) --config $(CONFIG) --seed 0 --cache warm; done
	for p in p0 p2; do $(PY) scripts/run_plan.py --plan $$p --dataset data/$(DS) --config $(CONFIG) --seed 0 --cache cold --no-warmup; done
	$(PY) scripts/eval_g.py --runs 'runs/p*-$(DS)-*' --questions data/$(DS)/questions.jsonl

reproduce:  ## regenerate the midterm tables from existing run directories
	$(PY) scripts/eval_g.py --runs 'runs/p*-$(DS)-*' --questions data/$(DS)/questions.jsonl
	$(PY) scripts/profile_indexes.py --dataset data/$(DS)
