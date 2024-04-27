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
	poetry run mypy .

.PHONY: check
check: format lint typecheck

.PHONY: test
test:
	poetry run pytest --nbmake notebooks/*.ipynb
