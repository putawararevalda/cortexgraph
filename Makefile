.PHONY: setup test run lint format check ingest-sample eval clean docker-up docker-down

PYTHON ?= python
UV ?= uv

setup:
	$(UV) pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check .
	mypy src

format:
	ruff format .
	ruff check --fix .

check: lint test

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

ingest-sample:
	$(PYTHON) -m src.rag.ingestion --vault-dir data/sample_vault

eval:
	$(PYTHON) -m evals.run_evals

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
