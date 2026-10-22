# Billiards Trainer Development Makefile
#
# Common development tasks and utilities for the billiards trainer project.
#
# Usage:
#   make help          Show this help message
#   make install       Install the project and dependencies
#   make dev           Setup the Rust and browser development environment
#   make test          Run tests
#   make lint          Run linting tools
#   make format        Format code
#   make clean         Clean build artifacts
#   make run           Run the application
#   make deploy        Build production distribution
#

.PHONY: help install dev install-dev install-python-tools python-venv test test-all test-rust test-web test-python lint lint-rust lint-web lint-python format format-rust format-python format-check config-check config-example clean run docker build deploy deploy-clean logs docs docs-python docs-serve

# Default target
.DEFAULT_GOAL := help

# The supported application runtime is Rust. Python remains available only for
# offline model/evaluation workflows and the compatibility test suite.
PYTHON_VENV_DIR ?= .venv
PYTHON ?= $(if $(wildcard $(PYTHON_VENV_DIR)/bin/python),$(PYTHON_VENV_DIR)/bin/python,python3)
BACKEND_DIR := backend
RUST_MANIFEST := perception/Cargo.toml
RUST_FEATURES ?= camera-gstreamer-source
RUST_HTTP_BIND ?= 127.0.0.1
RUST_HTTP_PORT ?= 8001
RUST_SCENE_BIND ?= 127.0.0.1
RUST_SCENE_PORT ?= 49321
RUST_CONFIG ?= config.json
RUST_SERVER_ARGS ?=
LOGS_DIR := logs
DATA_DIR := data
TEMP_DIR := temp

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

install: build ## Build the Rust server and browser client; no Python app runtime is installed
	@echo "$(GREEN)Rust application and browser client built successfully.$(RESET)"

install-dev: install-python-tools ## Compatibility alias for installing retained Python tools

install-python-tools: python-venv ## Install retained Python evaluation and model-tool dependencies
	@$(PYTHON_VENV_DIR)/bin/python -m pip install --upgrade pip
	@$(PYTHON_VENV_DIR)/bin/python -m pip install -e ".[python-tooling,dev,test,docs,performance]"
	@echo "$(GREEN)Python evaluation and model-tool dependencies installed.$(RESET)"

python-venv: ## Create the optional Python tooling environment
	@echo "$(BLUE)Creating optional Python tooling environment...$(RESET)"
	@if [ ! -x "$(PYTHON_VENV_DIR)/bin/python" ]; then \
		python3 -m venv "$(PYTHON_VENV_DIR)"; \
		echo "$(GREEN)Python tooling environment created in $(PYTHON_VENV_DIR)$(RESET)"; \
	else \
		echo "$(GREEN)Python tooling environment already exists$(RESET)"; \
	fi

dev: setup-dirs ## Setup Rust and browser development dependencies without Python
	@command -v cargo >/dev/null || { echo "cargo is required; install the Rust toolchain first."; exit 1; }
	@command -v npm >/dev/null || { echo "npm is required; install Node.js first."; exit 1; }
	cargo fetch --locked --manifest-path $(RUST_MANIFEST)
	cd frontend/web && npm ci
	@bash scripts/verify-development-setup.sh
	@echo "$(GREEN)Rust and browser development environment is ready.$(RESET)"

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

lint: lint-rust lint-web ## Check Rust and browser code quality

lint-rust: ## Run strict Rust lint checks
	cargo clippy --locked --manifest-path $(RUST_MANIFEST) --all-targets -- -D warnings

lint-web: ## Type-check the browser application
	cd frontend/web && npm run typecheck

lint-python: ## Lint retained Python compatibility and offline tooling
	$(PYTHON) -m ruff check $(BACKEND_DIR)
	$(PYTHON) -m mypy $(BACKEND_DIR)

format: format-rust ## Format Rust source

format-rust: ## Format Rust source
	cargo fmt --manifest-path $(RUST_MANIFEST) --all

format-python: ## Format retained Python compatibility and offline tooling
	$(PYTHON) -m black $(BACKEND_DIR)
	$(PYTHON) -m isort $(BACKEND_DIR)

format-check: ## Check Rust source formatting
	cargo fmt --manifest-path $(RUST_MANIFEST) --all -- --check

fix: format lint ## Fix code formatting and run linting

# ===== Testing =====

test: test-rust test-web ## Run Rust runtime and browser tests

test-all: test test-python ## Run Rust, browser, and retained Python compatibility tests

test-rust: ## Run Rust unit, contract, and fixture tests
	cargo test --locked --manifest-path $(RUST_MANIFEST) --all-targets

test-web: ## Run browser client tests
	cd frontend/web && npm test

test-python: ## Run retained Python compatibility and model-tool tests
	$(PYTHON) -m pytest

test-unit: ## Run retained Python unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTHON) -m pytest -m "unit"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest -m "integration"

test-vision: ## Run computer vision tests
	@echo "$(BLUE)Running vision tests...$(RESET)"
	$(PYTHON) -m pytest -m "vision"

test-api: ## Run API tests
	@echo "$(BLUE)Running API tests...$(RESET)"
	$(PYTHON) -m pytest -m "api"

test-hardware: ## Run hardware tests (requires camera/projector)
	@echo "$(YELLOW)Running hardware tests (requires camera/projector)...$(RESET)"
	$(PYTHON) -m pytest -m "hardware"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest --cov=backend --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (exclude slow tests)
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(PYTHON) -m pytest -m "not slow"

# ===== Build =====

build-frontend: ## Build the frontend web app
	@echo "$(BLUE)Building frontend...$(RESET)"
	cd frontend/web && npm ci && npm run build
	@echo "$(GREEN)Frontend built to backend/static/$(RESET)"

build-rust: ## Build Rust server and live perception binaries
	cargo build --manifest-path $(RUST_MANIFEST) --features $(RUST_FEATURES) \
		--bin billiards-server \
		--bin billiards-perception-live \
		--bin billiards-perception-gstreamer-probe

build: build-frontend build-rust ## Build the Rust application and web client

# ===== Development Server =====

kill-processes: ## Stop existing Billiards Trainer Rust processes
	@echo "$(YELLOW)Stopping Billiards Trainer Rust processes...$(RESET)"
	@pkill -TERM -x billiards-server 2>/dev/null || true
	@pkill -TERM -x billiards-perception-live 2>/dev/null || true
	@pkill -TERM -x billiards-perception-gstreamer-probe 2>/dev/null || true
	@sleep 1
	@echo "$(GREEN)Rust app processes stopped$(RESET)"

run: build-frontend kill-processes setup-dirs ## Run Rust server, live detector, and retained web client
	@echo "$(BLUE)Starting Billiards Trainer Rust server...$(RESET)"
	cargo run --manifest-path $(RUST_MANIFEST) --features $(RUST_FEATURES) --bin billiards-server -- \
		--http-bind $(RUST_HTTP_BIND) --http-port $(RUST_HTTP_PORT) \
		--scene-bind $(RUST_SCENE_BIND) --scene-port $(RUST_SCENE_PORT) \
		--config "$(abspath $(RUST_CONFIG))" --project-root "$(PWD)" \
		--web-root "$(PWD)/backend/static" --start-live $(RUST_SERVER_ARGS)

run-video: ## Deprecated: camera capture is supervised by the Rust server
	@echo "Camera capture is started and supervised by 'make run' (Rust)."

run-api: build-frontend build-rust ## Run Rust gateway without live capture
	cargo run --manifest-path $(RUST_MANIFEST) --features $(RUST_FEATURES) --bin billiards-server -- \
		--http-bind $(RUST_HTTP_BIND) --http-port $(RUST_HTTP_PORT) \
		--scene-bind $(RUST_SCENE_BIND) --scene-port $(RUST_SCENE_PORT) \
		--config "$(abspath $(RUST_CONFIG))" --project-root "$(PWD)" \
		--web-root "$(PWD)/backend/static" $(RUST_SERVER_ARGS)

run-frontend: ## Run frontend dev server only
	cd frontend/web && npm run dev

run-visualizer: ## Run LOVE2D visualizer only
	love frontend/visualizer

test-visualizer: ## Test LOVE2D visualizer module initialization
	cd frontend/visualizer && ./run_test.sh

run-vision: ## Deprecated: perception is supervised by the Rust server
	@echo "Use 'make run' to start the Rust perception pipeline."

run-projector: ## Deprecated: projector runtime is not part of the Rust server package
	@echo "The legacy Python projector is not launched by the Rust live application."

run-dev: run ## Run the Rust app; the frontend development server is separate

# ===== Monitoring and Debugging =====

logs: ## Show recent logs
	@echo "$(BLUE)Recent application logs:$(RESET)"
	@if [ -f $(LOGS_DIR)/info.log ]; then tail -50 $(LOGS_DIR)/info.log; else echo "No logs found"; fi

logs-video: ## Show video module logs
	@echo "$(BLUE)Video module logs:$(RESET)"
	@if [ -f $(LOGS_DIR)/video.log ]; then tail -50 $(LOGS_DIR)/video.log; else echo "No video logs found"; fi

logs-all: ## Show all recent logs (info, error, video)
	@echo "$(CYAN)=== Info Logs ===$(RESET)"
	@if [ -f $(LOGS_DIR)/info.log ]; then tail -30 $(LOGS_DIR)/info.log; else echo "No info logs"; fi
	@echo ""
	@echo "$(RED)=== Error Logs ===$(RESET)"
	@if [ -f $(LOGS_DIR)/error.log ]; then tail -30 $(LOGS_DIR)/error.log; else echo "No error logs"; fi
	@echo ""
	@echo "$(BLUE)=== Video Logs ===$(RESET)"
	@if [ -f $(LOGS_DIR)/video.log ]; then tail -30 $(LOGS_DIR)/video.log; else echo "No video logs"; fi

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
	@echo "Rust profiling is available through docs/PERFORMANCE_AUDIT.md."

# ===== Configuration =====

config-check: ## Validate configuration files
	@echo "$(BLUE)Checking configuration files...$(RESET)"
	@node -e 'JSON.parse(require("node:fs").readFileSync("$(RUST_CONFIG)", "utf8"));' \
		&& echo "$(GREEN)Rust configuration JSON is valid$(RESET)"
	@temporary_store=$$(mktemp -d) && trap 'rm -rf "$$temporary_store"' EXIT && \
		node --input-type=module -e 'import fs from "node:fs"; const data=JSON.parse(fs.readFileSync("$(RUST_CONFIG)", "utf8")); process.stdout.write(JSON.stringify({protocol_version:1,operation:"validate",merge:false,data})+"\n");' | \
		cargo run --quiet --manifest-path $(RUST_MANIFEST) --bin billiards-perception-config -- --config "$$temporary_store/validation.json" | \
		node --input-type=module -e 'let raw=""; for await (const chunk of process.stdin) raw += chunk; const result=JSON.parse(raw); if (!result.ok || !result.valid) { console.error(JSON.stringify(result.validation_errors ?? result.error)); process.exit(1); } console.log("Rust configuration schema and runtime rules are valid");'

config-example: ## Create example configuration files
	@echo "$(BLUE)Creating example configuration files...$(RESET)"
	@if [ -e "$(RUST_CONFIG)" ]; then \
		echo "$(RED)$(RUST_CONFIG) already exists; refusing to overwrite it$(RESET)"; exit 1; \
	fi
	@cp perception/contracts/configuration/v1/defaults.json "$(RUST_CONFIG)"
	@echo "$(GREEN)Created $(RUST_CONFIG) from the Rust-owned configuration defaults$(RESET)"

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

deps-update: ## Update Rust and browser dependency lockfiles
	cargo update --manifest-path $(RUST_MANIFEST)
	cd frontend/web && npm update

deps-check: ## Check locked Rust dependencies and browser package updates
	cargo metadata --locked --manifest-path $(RUST_MANIFEST) --no-deps >/dev/null
	cd frontend/web && npm outdated

security-check: ## Audit Rust and browser dependencies
	@if command -v cargo-audit >/dev/null 2>&1; then \
		cd perception && cargo audit; \
	else \
		echo "$(YELLOW)cargo-audit is unavailable; install it to audit Rust dependencies.$(RESET)"; \
	fi
	cd frontend/web && npm audit --audit-level=high

check-env: ## Check Rust, Node.js, npm, and camera build prerequisites
	@echo "$(BLUE)Environment Check:$(RESET)"
	@cargo --version
	@rustc --version
	@node --version
	@npm --version
	@if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0; then \
		echo "$(GREEN)GStreamer development libraries are available$(RESET)"; \
	else \
		echo "$(YELLOW)GStreamer development libraries are required for the default camera-enabled build$(RESET)"; \
	fi

# ===== Quick Start =====

quickstart: ## Quick setup for new developers
	@echo "$(CYAN)Welcome to Billiards Trainer!$(RESET)"
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(MAKE) dev
	@echo ""
	@echo "$(GREEN)Setup complete! Next steps:$(RESET)"
	@echo "  1. $(YELLOW)make run$(RESET)                  # Run the Rust application"
	@echo "  2. $(YELLOW)make test$(RESET)                 # Run Rust and browser tests"
	@echo "  3. $(YELLOW)make test-python$(RESET)          # Optional retained Python checks"
	@echo ""
	@echo "$(CYAN)Available commands: make help$(RESET)"

# ===== Documentation =====

docs: ## Generate Rust API documentation
	cargo doc --locked --manifest-path $(RUST_MANIFEST) --no-deps

docs-python: ## Generate the retained Sphinx documentation site
	sphinx-build -b html docs docs/_build/html

docs-serve: docs ## Serve generated Rust API documentation locally
	@echo "$(BLUE)Serving Rust API documentation at http://localhost:8080$(RESET)"
	cd perception/target/doc && npx --yes http-server -p 8080

# ===== Deployment =====

deploy: ## Build production distribution package using the Rust shell bundler
	@bash scripts/deploy/build-dist.sh

deploy-clean: ## Clean deployment artifacts
	@echo "$(BLUE)Cleaning deployment artifacts...$(RESET)"
	@rm -rf dist/
	@echo "$(GREEN)Deployment artifacts cleaned!$(RESET)"
