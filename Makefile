# Billiards Trainer Development Makefile
#
# Common development tasks and utilities for the billiards trainer project.
#
# Usage:
#   make help          Show this help message
#   make install       Install the project and dependencies
#   make dev           Setup development environment
#   make test          Run tests
#   make lint          Run linting tools
#   make format        Format code
#   make clean         Clean build artifacts
#   make run           Run the application
#   make deploy        Build production distribution
#

.PHONY: help install dev test lint format clean run docker build deploy deploy-clean logs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
VENV_DIR := venv
BACKEND_DIR := backend
CONFIG_DIR := config
LOGS_DIR := logs
DATA_DIR := data
TEMP_DIR := temp

# Use venv if it exists
ifneq ($(wildcard $(VENV_DIR)/bin/python),)
	PYTHON := $(VENV_DIR)/bin/python
	PIP := $(VENV_DIR)/bin/pip
endif

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)Billiards Trainer Development Makefile$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ===== Environment Setup =====

install: venv ## Install the project and dependencies
	@echo "$(BLUE)Installing billiards-trainer...$(RESET)"
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/pip install -e .
	@echo "$(GREEN)Installation complete!$(RESET)"
	@echo "$(YELLOW)Activate the virtual environment: source $(VENV_DIR)/bin/activate$(RESET)"

install-dev: venv ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/pip install -e ".[dev,test,docs,performance]"
	@echo "$(GREEN)Development dependencies installed!$(RESET)"

venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(RESET)"
	@if [ ! -d $(VENV_DIR) ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "$(GREEN)Virtual environment created in $(VENV_DIR)$(RESET)"; \
		echo "$(YELLOW)Activate with: source $(VENV_DIR)/bin/activate$(RESET)"; \
	else \
		echo "$(GREEN)Virtual environment already exists$(RESET)"; \
	fi

dev: venv setup-dirs ## Setup complete development environment
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/pip install -e ".[dev,test,docs,performance]"
	@echo "$(GREEN)Development environment setup complete!$(RESET)"
	@echo "$(YELLOW)Activate the virtual environment:$(RESET)"
	@echo "  source $(VENV_DIR)/bin/activate"

setup-dirs: ## Create necessary directories
	@echo "$(BLUE)Creating project directories...$(RESET)"
	@mkdir -p $(LOGS_DIR) $(DATA_DIR) $(TEMP_DIR) debug_images
	@touch $(LOGS_DIR)/.gitkeep $(DATA_DIR)/.gitkeep $(TEMP_DIR)/.gitkeep
	@echo "$(GREEN)Directories created!$(RESET)"

setup-git-hooks: ## Setup git pre-commit hooks
	@echo "$(BLUE)Setting up git hooks...$(RESET)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)Git hooks installed!$(RESET)"; \
	else \
		echo "$(YELLOW)pre-commit not found. Install with: pip install pre-commit$(RESET)"; \
	fi

# ===== Code Quality =====

lint: ## Run all linting tools
	@echo "$(BLUE)Running linting tools...$(RESET)"
	@echo "$(CYAN)Running ruff...$(RESET)"
	ruff check $(BACKEND_DIR)
	@echo "$(CYAN)Running mypy...$(RESET)"
	mypy $(BACKEND_DIR)
	@echo "$(GREEN)Linting complete!$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	@echo "$(CYAN)Running black...$(RESET)"
	black $(BACKEND_DIR)
	@echo "$(CYAN)Running isort...$(RESET)"
	isort $(BACKEND_DIR)
	@echo "$(GREEN)Code formatting complete!$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check $(BACKEND_DIR)
	isort --check-only $(BACKEND_DIR)
	@echo "$(GREEN)Format check complete!$(RESET)"

fix: format lint ## Fix code formatting and run linting

# ===== Testing =====

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest -m "unit"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest -m "integration"

test-vision: ## Run computer vision tests
	@echo "$(BLUE)Running vision tests...$(RESET)"
	pytest -m "vision"

test-api: ## Run API tests
	@echo "$(BLUE)Running API tests...$(RESET)"
	pytest -m "api"

test-hardware: ## Run hardware tests (requires camera/projector)
	@echo "$(YELLOW)Running hardware tests (requires camera/projector)...$(RESET)"
	pytest -m "hardware"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest --cov=backend --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (exclude slow tests)
	@echo "$(BLUE)Running fast tests...$(RESET)"
	pytest -m "not slow"

# ===== Build =====

build-frontend: ## Build the frontend web app
	@echo "$(BLUE)Building frontend...$(RESET)"
	cd frontend/web && npm install && npm run build
	@echo "$(GREEN)Frontend built to backend/static/$(RESET)"

build: build-frontend ## Build the entire application

# ===== Development Server =====

run: ## Run the main application
	@echo "$(BLUE)Starting billiards trainer...$(RESET)"
	$(PYTHON) -m backend.main

run-api: ## Run API server only
	@echo "$(BLUE)Starting API server...$(RESET)"
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run frontend dev server only
	@echo "$(BLUE)Starting frontend dev server...$(RESET)"
	cd frontend/web && npm run dev

run-vision: ## Run vision system only
	@echo "$(BLUE)Starting vision system...$(RESET)"
	$(PYTHON) -m backend.vision.main

run-projector: ## Run projector system only
	@echo "$(BLUE)Starting projector system...$(RESET)"
	$(PYTHON) -m backend.projector.main

run-dev: ## Run in development mode with hot reload
	@echo "$(BLUE)Starting in development mode...$(RESET)"
	ENVIRONMENT=development $(PYTHON) -m backend.main

# ===== Monitoring and Debugging =====

logs: ## Show recent logs
	@echo "$(BLUE)Recent application logs:$(RESET)"
	@if [ -f $(LOGS_DIR)/info.log ]; then tail -50 $(LOGS_DIR)/info.log; else echo "No logs found"; fi

logs-error: ## Show error logs
	@echo "$(RED)Recent error logs:$(RESET)"
	@if [ -f $(LOGS_DIR)/error.log ]; then tail -50 $(LOGS_DIR)/error.log; else echo "No error logs found"; fi

logs-debug: ## Show debug logs
	@echo "$(BLUE)Recent debug logs:$(RESET)"
	@if [ -f $(LOGS_DIR)/debug.log ]; then tail -50 $(LOGS_DIR)/debug.log; else echo "No debug logs found"; fi

monitor: ## Monitor log files in real-time
	@echo "$(BLUE)Monitoring logs (Ctrl+C to stop)...$(RESET)"
	@if [ -f $(LOGS_DIR)/info.log ]; then tail -f $(LOGS_DIR)/info.log; else echo "No logs to monitor"; fi

profile: ## Run with performance profiling
	@echo "$(BLUE)Running with profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats -m backend.main
	@echo "$(GREEN)Profile saved to profile.stats$(RESET)"

# ===== Configuration =====

config-check: ## Validate configuration files
	@echo "$(BLUE)Checking configuration files...$(RESET)"
	@if [ -f .env ]; then echo "$(GREEN).env file found$(RESET)"; else echo "$(RED).env file missing$(RESET)"; fi
	@if [ -f $(CONFIG_DIR)/logging.yaml ]; then echo "$(GREEN)Logging config found$(RESET)"; else echo "$(RED)Logging config missing$(RESET)"; fi
	@$(PYTHON) -c "import yaml; yaml.safe_load(open('$(CONFIG_DIR)/logging.yaml'))" && echo "$(GREEN)Logging config is valid$(RESET)" || echo "$(RED)Logging config is invalid$(RESET)"

config-example: ## Create example configuration files
	@echo "$(BLUE)Creating example configuration files...$(RESET)"
	@cp .env .env.example
	@echo "$(GREEN).env.example created$(RESET)"

# ===== Cleanup =====

clean: ## Clean build artifacts and temporary files
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -f profile.stats
	@echo "$(GREEN)Cleanup complete!$(RESET)"

clean-logs: ## Clean log files
	@echo "$(BLUE)Cleaning log files...$(RESET)"
	rm -f $(LOGS_DIR)/*.log*
	@echo "$(GREEN)Log files cleaned!$(RESET)"

clean-data: ## Clean data and temporary files
	@echo "$(BLUE)Cleaning data files...$(RESET)"
	rm -rf $(DATA_DIR)/*
	rm -rf $(TEMP_DIR)/*
	rm -rf debug_images/*
	@echo "$(GREEN)Data files cleaned!$(RESET)"

clean-frontend: ## Clean frontend build artifacts
	@echo "$(BLUE)Cleaning frontend build artifacts...$(RESET)"
	rm -rf backend/static/
	rm -rf frontend/web/dist/
	@echo "$(GREEN)Frontend artifacts cleaned!$(RESET)"

clean-all: clean clean-logs clean-frontend ## Clean everything

# ===== Docker =====

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t billiards-trainer .
	@echo "$(GREEN)Docker image built!$(RESET)"

docker-build-dev: ## Build Docker development image
	@echo "$(BLUE)Building Docker development image...$(RESET)"
	docker-compose build
	@echo "$(GREEN)Docker development image built!$(RESET)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -p 8000:8000 -v $(PWD)/config:/app/config billiards-trainer

docker-dev: ## Start Docker containers in development mode with hot-reload
	@echo "$(BLUE)Starting Docker containers in development mode...$(RESET)"
	docker-compose up

docker-dev-build: ## Rebuild and start Docker containers in development mode
	@echo "$(BLUE)Rebuilding and starting Docker containers...$(RESET)"
	docker-compose up --build

docker-dev-detached: ## Start Docker containers in background with hot-reload
	@echo "$(BLUE)Starting Docker containers in background...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Containers running in background. Use 'make docker-logs' to view logs$(RESET)"

docker-stop: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(RESET)"
	docker-compose down
	@echo "$(GREEN)Containers stopped$(RESET)"

docker-restart: ## Restart Docker containers
	@echo "$(BLUE)Restarting Docker containers...$(RESET)"
	docker-compose restart
	@echo "$(GREEN)Containers restarted$(RESET)"

docker-logs: ## Show Docker container logs
	@echo "$(BLUE)Showing container logs (Ctrl+C to stop)...$(RESET)"
	docker-compose logs -f

docker-logs-backend: ## Show backend container logs only
	@echo "$(BLUE)Showing backend logs (Ctrl+C to stop)...$(RESET)"
	docker-compose logs -f backend

docker-shell: ## Open shell in backend container
	@echo "$(BLUE)Opening shell in backend container...$(RESET)"
	docker-compose exec backend /bin/bash

docker-clean: ## Remove all containers, volumes, and images
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	docker-compose down -v
	docker-compose rm -f
	@echo "$(GREEN)Docker resources cleaned!$(RESET)"

# ===== Utilities =====

deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r backend/requirements.txt
	@echo "$(GREEN)Dependencies updated!$(RESET)"

deps-check: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	$(PIP) list --outdated

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "$(YELLOW)safety not installed. Install with: pip install safety$(RESET)"; \
	fi

check-env: ## Check environment setup
	@echo "$(BLUE)Environment Check:$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "Active ($$VIRTUAL_ENV)"; else echo "Not active"; fi)"
	@echo "Environment: $$(if [ -f .env ]; then grep ENVIRONMENT .env || echo "Not set"; else echo ".env not found"; fi)"

# ===== Quick Start =====

quickstart: ## Quick setup for new developers
	@echo "$(CYAN)Welcome to Billiards Trainer!$(RESET)"
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	make dev
	@echo ""
	@echo "$(GREEN)Setup complete! Next steps:$(RESET)"
	@echo "  1. $(YELLOW)source venv/bin/activate$(RESET)  # Activate virtual environment"
	@echo "  2. $(YELLOW)make run$(RESET)                  # Run the application"
	@echo "  3. $(YELLOW)make test$(RESET)                 # Run tests"
	@echo ""
	@echo "$(CYAN)Available commands: make help$(RESET)"

# ===== Documentation =====

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs docs/_build/html; \
		echo "$(GREEN)Documentation generated in docs/_build/html$(RESET)"; \
	else \
		echo "$(YELLOW)Sphinx not installed. Install with: pip install sphinx$(RESET)"; \
	fi

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(RESET)"
	@if [ -d docs/_build/html ]; then \
		$(PYTHON) -m http.server 8080 --directory docs/_build/html; \
	else \
		echo "$(RED)Documentation not built. Run 'make docs' first.$(RESET)"; \
	fi

# ===== Deployment =====

deploy: build-frontend ## Build production distribution package
	@bash scripts/deploy/build-dist.sh

deploy-push: deploy ## Build and push Docker image to registry
	@echo "$(BLUE)Deploying to target...$(RESET)"
	@bash rsync -av dist/ $DEPLOY_TARGET
	@echo "$(GREEN)Deployed!$(RESET)"

deploy-clean: ## Clean deployment artifacts
	@echo "$(BLUE)Cleaning deployment artifacts...$(RESET)"
	@rm -rf dist/
	@echo "$(GREEN)Deployment artifacts cleaned!$(RESET)"
