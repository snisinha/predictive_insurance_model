
PYTHON      ?= python3
VENV        ?= .venv
PIP         := $(VENV)/bin/pip
PY          := $(VENV)/bin/python

.PHONY: default install build test run run-fast run-eda-only clean help

default: help

help:
	@echo "Predictive Insurance Model"
	@echo ""
	@echo "  make install     Create $(VENV) and install dependencies"
	@echo "  make build       Same as install (reproducible environment)"
	@echo "  make test        Run pytest"
	@echo "  make run         Full pipeline (EDA + train/evaluate models)"
	@echo "  make run-fast    Models only (--skip-eda)"
	@echo "  make run-eda-only  EDA only, then exit"
	@echo "  make clean       Remove virtualenv"
	@echo ""
	@echo "Requires: $(PYTHON), datasets/dataset_main.csv for run targets"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

build: install

test: install
	$(PY) -m pytest tests/ -q

run: install
	$(PY) main.py

run-fast: install
	$(PY) main.py --skip-eda

run-eda-only: install
	$(PY) main.py --eda-only

clean:
	rm -rf $(VENV)
