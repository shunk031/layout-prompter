.PHONY: setup
setup:
	pip install -U pip uv
	uv sync

.PHONY: format
format:
	uv run ruff format --check --diff .

.PHONY: lint
lint:
	uv run ruff check --output-format=github .

.PHONY: typecheck
typecheck:
	uv run mypy .

.PHONY: check
check: format lint typecheck

.PHONY: test-package
test-package:
	uv run pytest tests

.PHONY: test-notebooks
test-notebooks:
	uv run pytest --nbmake notebooks/*.ipynb	

.PHONY: test
test: test-package test-notebooks
