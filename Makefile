.PHONY: install sync lint format test run dev frontend-install frontend-dev frontend-build

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