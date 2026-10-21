# Backend test shortcuts. Paths are rooted at the repository, so targets work
# whether this file is invoked from the repository root or backend/tests.

PYTHON ?= python3
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MAKEFILE_DIR)/../..)
PYTEST := $(PYTHON) -m pytest

.PHONY: help install-dev collect test test-unit test-integration test-system test-ci lint format clean
.DEFAULT_GOAL := help

help: ## List available backend checks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

install-dev: ## Install backend and test dependencies
	cd $(REPO_ROOT) && $(PYTHON) -m pip install -r backend/requirements.txt

collect: ## Verify all backend tests import and collect
	cd $(REPO_ROOT) && $(PYTEST) --collect-only --no-cov backend/tests

test: ## Run hardware-free backend tests
	cd $(REPO_ROOT) && $(PYTEST) backend/tests -m "not slow and not hardware"

test-unit: ## Run tests in the unit-test suite
	cd $(REPO_ROOT) && $(PYTEST) backend/tests/unit

test-integration: ## Run backend integration contract tests
	cd $(REPO_ROOT) && $(PYTEST) backend/tests/integration

test-system: ## Run hardware-free backend system contract tests
	cd $(REPO_ROOT) && $(PYTEST) backend/tests/system -m "system and not hardware"

test-ci: collect ## Run the same hardware-free coverage gate used by CI
	cd $(REPO_ROOT) && $(PYTEST) backend/tests -m "not slow and not hardware" --cov-report=xml:backend/coverage.xml --cov-report=html:backend/coverage-html --junitxml=backend/junit.xml

lint: ## Check formatting and import order
	cd $(REPO_ROOT) && $(PYTHON) -m black --check --diff backend
	cd $(REPO_ROOT) && $(PYTHON) -m isort --check-only --diff backend

format: ## Format backend code and imports
	cd $(REPO_ROOT) && $(PYTHON) -m black backend
	cd $(REPO_ROOT) && $(PYTHON) -m isort backend

clean: ## Remove generated pytest and coverage artifacts
	rm -rf $(REPO_ROOT)/.coverage $(REPO_ROOT)/.pytest_cache $(REPO_ROOT)/backend/coverage.xml $(REPO_ROOT)/backend/coverage-html $(REPO_ROOT)/backend/junit.xml $(REPO_ROOT)/htmlcov
	find $(REPO_ROOT)/backend -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
