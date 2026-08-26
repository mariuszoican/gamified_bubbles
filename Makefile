# Gamified Bubbles — analysis pipeline
# Run from the repo root.

PYTHON ?= .venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
  PYTHON := python3
endif

export PYTHONPATH := src/build

.PHONY: panels session payments analyze explore clean-interim help

help:
	@echo "Targets:"
	@echo "  make payments ID=20260512 Write data/payments/payments_YYYYMMDD.xlsx"
	@echo "  make panels              Rebuild interim + full panels for include:true sessions"
	@echo "  make session ID=20260512 Process one session from config/sessions.yaml"
	@echo "  make analyze             Run hypothesis_tests.R (writes output/tables/)"
	@echo "  make explore             Open exploratory sandbox plots"
	@echo "  make clean-interim       Delete rebuildable data/interim panels"

panels:
	$(PYTHON) src/build/build_panels.py

session:
	@test -n "$(ID)" || (echo "Usage: make session ID=20260512"; exit 1)
	$(PYTHON) src/build/process_session.py --session $(ID)

payments:
	@test -n "$(ID)" || (echo "Usage: make payments ID=20260512"; exit 1)
	$(PYTHON) src/build/process_payments.py --session $(ID)

analyze:
	Rscript src/analyze/hypothesis_tests.R

explore:
	$(PYTHON) src/explore/sandbox_data.py

clean-interim:
	rm -rf data/interim/*
	@echo "Removed data/interim/* (raw and processed untouched)"
