.PHONY: setup
setup:
	pip install -U pip setuptools wheel poetry
	poetry install

.PHONY: format
format:
	poetry run ruff format --check --diff .

.PHONY: lint
lint:
	poetry run ruff check --output-format=github .

.PHONY: typecheck
typecheck:
	poetry run mypy layout_prompter

.PHONY: check
check: format lint typecheck

.PHONY: test-package
test-package:
	poetry run pytest tests

.PHONY: test-notebooks
test-notebooks:
	poetry run pytest --nbmake notebooks/*.ipynb	

.PHONY: test
test: test-package test-notebooks
