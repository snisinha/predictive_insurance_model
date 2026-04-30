.PHONY: install install-deps run eda models clean help

VENV    = venv
PYTHON  = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

# ── Setup ─────────────────────────────────────────────────────────────────────

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: $(VENV)/bin/activate

install-deps: install
	$(PIP) install -r requirements.txt

# ── Run ───────────────────────────────────────────────────────────────────────

run: install
	$(PYTHON) main.py

eda: install
	$(PYTHON) main.py --eda-only

models: install
	$(PYTHON) main.py --skip-eda

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-outputs:
	rm -rf outputs/

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  make install        Create venv"
	@echo "  make install-deps   Create venv and install requirements.txt"
	@echo "  make run            Run the full pipeline (EDA + all models)"
	@echo "  make eda            Run EDA only"
	@echo "  make models         Run models only (skip EDA)"
	@echo "  make clean          Remove venv and cache files"
	@echo "  make clean-outputs  Remove generated outputs/"