PY ?= .venv/bin/python
CHORE ?= desk_clearing
export PYTHONPATH := $(CURDIR)
export MPLBACKEND := Agg

.PHONY: setup test safety-test lint typecheck ci sim-smoke sim-eval process-videos train queue-session reproduce-quick reproduce-full paper check-claims scan

setup:
	uv venv .venv -p 3.12 && uv pip install -p $(PY) -r requirements.lock && uv pip install -p $(PY) -e . --no-deps

test:
	$(PY) -m pytest -q

safety-test:
	$(PY) -m pytest -q -m safety

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy

sim-smoke:
	$(PY) scripts/supervised_session.py safety-smoke --sim --session ci_smoke

ci: lint typecheck test sim-smoke

sim-eval:
	$(PY) -m bench.run_all --stage sim --chore $(CHORE)

process-videos:
	$(PY) -m perception.pipeline --chore $(CHORE)

train:
	$(PY) -m policies.train --chore $(CHORE)

queue-session:
	$(PY) -m eval.queue --chore $(CHORE)

reproduce-quick:
	$(PY) -m bench.reproduce --quick

reproduce-full:
	$(PY) -m bench.reproduce --full

paper:
	$(PY) -m bench.make_figures && $(PY) -m bench.check_claims

check-claims:
	$(PY) -m bench.check_claims

scan:
	$(PY) -m bench.repo_scan
