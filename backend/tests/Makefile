# Makefile for billiards trainer backend tests
# Provides convenient shortcuts for common testing tasks

.PHONY: help test test-unit test-integration test-performance test-system test-all test-quick test-ci
.PHONY: lint format coverage report clean watch install-dev
.DEFAULT_GOAL := help

# Python executable
PYTHON := python3

# Test directories
TESTS_DIR := tests
COVERAGE_DIR := $(TESTS_DIR)/coverage
LOGS_DIR := $(TESTS_DIR)/logs

help: ## Show this help message
	@echo "Billiards Trainer Backend Test Commands"
	@echo "======================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -m pip install pytest pytest-asyncio pytest-cov pytest-html pytest-xdist
	@$(PYTHON) -m pip install black isort mypy watchdog

setup: ## Set up test environment
	@echo "🔧 Setting up test environment..."
	@mkdir -p $(COVERAGE_DIR) $(LOGS_DIR) $(TESTS_DIR)/data
	@touch $(LOGS_DIR)/.gitkeep $(TESTS_DIR)/data/.gitkeep

test: test-quick ## Run quick tests (default)

test-unit: setup ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/unit/ -m unit -v

test-integration: setup ## Run integration tests only
	@echo "🔗 Running integration tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/integration/ -m integration -v

test-performance: setup ## Run performance tests only
	@echo "⚡ Running performance tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/performance/ -m performance -v --tb=short

test-system: setup ## Run system tests only
	@echo "🔄 Running system tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/system/ -m system -v --tb=short

test-all: setup ## Run all tests
	@echo "🚀 Running all tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ -v

test-quick: setup ## Run quick tests (unit + fast integration)
	@echo "⚡ Running quick tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/unit/ $(TESTS_DIR)/integration/ -m "not slow and not hardware" --tb=short

test-ci: setup ## Run CI-suitable tests
	@echo "🤖 Running CI tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ \
		--cov=backend \
		--cov-report=xml:coverage.xml \
		--cov-report=term \
		--cov-fail-under=80 \
		--junit-xml=junit.xml \
		-m "not hardware and not slow" \
		--tb=short \
		-q

test-hardware: setup ## Run hardware-dependent tests
	@echo "🔧 Running hardware tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ -m hardware -v

test-parallel: setup ## Run tests in parallel
	@echo "🚀 Running tests in parallel..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ -n auto --tb=short

lint: ## Run code linting
	@echo "📝 Running lint checks..."
	@echo "  🖤 Black formatting..."
	@black --check backend/
	@echo "  📚 Import sorting..."
	@isort --check-only backend/
	@echo "  🐍 Type checking..."
	@mypy backend/

format: ## Format code
	@echo "🎨 Formatting code..."
	@black backend/
	@isort backend/
	@echo "✅ Code formatted"

coverage: setup ## Run tests with coverage
	@echo "📊 Running tests with coverage..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ \
		--cov=backend \
		--cov-report=html:$(COVERAGE_DIR)/html \
		--cov-report=xml:coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		-m "not hardware and not slow"
	@echo "📁 Coverage report: $(COVERAGE_DIR)/html/index.html"

report: setup ## Generate comprehensive test report
	@echo "📊 Generating comprehensive test report..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ \
		--html=$(COVERAGE_DIR)/report.html \
		--self-contained-html \
		--cov=backend \
		--cov-report=html:$(COVERAGE_DIR)/html \
		--cov-report=xml:coverage.xml \
		--junit-xml=$(COVERAGE_DIR)/junit.xml \
		-m "not slow and not hardware"
	@echo "📁 Reports generated:"
	@echo "   - HTML: $(COVERAGE_DIR)/report.html"
	@echo "   - Coverage: $(COVERAGE_DIR)/html/index.html"
	@echo "   - JUnit: $(COVERAGE_DIR)/junit.xml"

watch: setup ## Watch for changes and run tests
	@echo "👀 Watching for changes..."
	@$(PYTHON) ../run_tests.py --watch

clean: ## Clean test artifacts
	@echo "🧹 Cleaning test artifacts..."
	@rm -rf $(COVERAGE_DIR)
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf .coverage coverage.xml junit.xml
	@rm -rf __pycache__ .pytest_cache
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned test artifacts"

benchmark: setup ## Run performance benchmarks
	@echo "📈 Running performance benchmarks..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/performance/ -m performance --benchmark-only

profile: setup ## Run tests with profiling
	@echo "🔍 Running tests with profiling..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ --profile --profile-svg

debug: setup ## Run tests in debug mode
	@echo "🐛 Running tests in debug mode..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/ -vvv -s --tb=long --pdb-trace

smoke: setup ## Run smoke tests (fastest essential tests)
	@echo "💨 Running smoke tests..."
	@$(PYTHON) -m pytest $(TESTS_DIR)/unit/test_config.py $(TESTS_DIR)/unit/test_api.py -v --tb=short

validate: lint test-ci ## Validate code (lint + CI tests)
	@echo "✅ Code validation complete"

# Docker targets
docker-test: ## Run tests in Docker container
	@echo "🐳 Running tests in Docker..."
	@docker build -t billiards-test -f Dockerfile.test .
	@docker run --rm billiards-test

docker-clean: ## Clean Docker test artifacts
	@echo "🐳 Cleaning Docker test artifacts..."
	@docker rmi billiards-test 2>/dev/null || true

# Documentation targets
docs-test: ## Test documentation examples
	@echo "📚 Testing documentation examples..."
	@$(PYTHON) -m pytest docs/ --doctest-modules

# Continuous testing
continuous: ## Run tests continuously (alias for watch)
	@$(MAKE) watch

# Environment info
info: ## Show environment information
	@echo "Environment Information"
	@echo "======================"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pytest: $(shell $(PYTHON) -m pytest --version 2>/dev/null || echo 'Not installed')"
	@echo "Coverage: $(shell $(PYTHON) -m coverage --version 2>/dev/null || echo 'Not installed')"
	@echo "Working Directory: $(PWD)"
	@echo "Tests Directory: $(TESTS_DIR)"
