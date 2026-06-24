# mem0 baseline uses a separate venv (eval-venv) — not the uv workspace.
# Override: make run-mem0-baseline MEM0_PYTHON=/path/to/python
MEM0_PYTHON ?= /Users/sarthak/eval-venv/bin/python

.PHONY: install sync lint format test run dev frontend-install frontend-dev frontend-build run-mem0-baseline

install:
	uv sync --all-packages

sync:
	uv sync --all-packages

lint:
	uv run ruff check .

format:
	uv run ruff format .

lint-fix:
	uv run ruff check --fix .

test:
	uv run pytest

run:
	uv run --package api uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev:
	cp .env.development.example .env.development && echo "Edit .env.development with your keys"

check:
	uv run ruff check . && uv run pytest

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

run-mem0-baseline:
	mkdir -p eval/logs eval/results
	$(MEM0_PYTHON) eval/eval_qa_mem0_baseline.py \
		--conv-ids conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
		--skip-ingestion \
		--collection mem0_locomo_batched \
		--output eval/results/qa_mem0_baseline_results.json \
		> eval/logs/mem0_baseline.log 2>&1