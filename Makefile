PYTHON ?= .venv/Scripts/python.exe

.PHONY: setup unzip cache weaklabels train tune eval predict validate test

setup:
	$(PYTHON) scripts/download_models.py
	$(PYTHON) scripts/make_folds.py

unzip:
	@echo "Dataset files are located in dataset/"

cache:
	$(PYTHON) -m src.eval.ablate --action cache --split development

weaklabels:
	$(PYTHON) -m src.eval.ablate --action weaklabels
	$(PYTHON) scripts/debug_render.py --num-pages 24

train:
	$(PYTHON) -m src.eval.ablate --action train

tune:
	$(PYTHON) -m src.eval.ablate --action tune

eval:
	$(PYTHON) -m src.eval.ablate --action eval

predict:
	$(PYTHON) -m src.eval.ablate --action predict

validate:
	$(PYTHON) -m src.pipeline.write_submission --submission submission.jsonl --sample dataset/sample_submission.jsonl

test:
	$(PYTHON) -m pytest -v
