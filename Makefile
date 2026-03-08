PYTHON ?= python

.PHONY: reproduce train test api eda

reproduce: train

train:
	$(PYTHON) scripts/run_pipeline.py --config configs/default.json

test:
	$(PYTHON) -m unittest discover -s tests -v

api:
	uvicorn risk_case.api.main:app --app-dir src --reload

eda:
	$(PYTHON) scripts/run_eda.py --config configs/eda.json
