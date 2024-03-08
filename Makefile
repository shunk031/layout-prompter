.PHONY: setup
setup:
	pip install -U pip setuptools wheel poetry
	poetry install

.PHONY: format
format:
	poetry run black --check .

.PHONY: lint
lint:
	poetry run ruff .

.PHONY: typecheck
typecheck:
	poetry run mypy .

.PHONY: check
check: format lint typecheck
