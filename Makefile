# Billiards Trainer Development Makefile
#
# Common development tasks and utilities for the billiards trainer project.
#
# Usage:
#   make help          Show this help message
#   make dev           Setup the Rust and browser development environment
#   make test          Run tests
#   make lint          Run linting tools
#   make format        Format code
#   make clean         Clean build artifacts
#   make run           Run the application
#   make deploy        Build production distribution
#

.PHONY: help dev install-python-tools install-model-tools python-venv setup-git-hooks setup-dirs test test-all test-rust test-rust-camera test-web test-python test-deploy test-tooling test-unit test-integration test-system test-coverage lint lint-rust lint-web lint-python format format-rust format-python format-check fix config-check config-example clean clean-logs clean-data clean-frontend clean-all run run-api run-frontend docker-build docker-run docker-run-detached docker-stop docker-restart docker-logs docker-shell build build-frontend build-rust deploy deploy-clean deps-update deps-check security-check security-check-rust security-check-web check-env docs

# Default target
.DEFAULT_GOAL := help

# The supported application runtime is Rust. Python remains for offline
# compatibility tests and optional model/data authoring.
PYTHON_VENV_DIR ?= .venv
PYTHON ?= $(if $(wildcard $(PYTHON_VENV_DIR)/bin/python),$(PYTHON_VENV_DIR)/bin/python,python3)
UV ?= uv
BACKEND_DIR := backend
RUST_MANIFEST := perception/Cargo.toml
RUST_FEATURES ?= camera-gstreamer-source
# Keep `cargo test` artifacts separate from the binaries that `make run` and
# the Rust live supervisor launch from `perception/target/debug`. Cargo shares
# executable paths across feature sets inside one target directory, so a
# default-feature test build must not replace the camera-enabled runtime while
# it is running.
RUST_TEST_TARGET_DIR ?= $(CURDIR)/perception/target/test
RUST_CAMERA_TEST_TARGET_DIR ?= $(CURDIR)/perception/target/test-camera
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

install-python-tools: python-venv ## Install retained Python compatibility and reference-test dependencies
	UV_PROJECT_ENVIRONMENT="$(abspath $(PYTHON_VENV_DIR))" $(UV) sync --locked --extra python-tooling --extra dev
	@echo "$(GREEN)Optional Python authoring and compatibility-test dependencies installed.$(RESET)"

install-model-tools: python-venv ## Install only optional model-export dependencies
	UV_PROJECT_ENVIRONMENT="$(abspath $(PYTHON_VENV_DIR))" $(UV) sync --locked --inexact --extra model-export
	@echo "$(GREEN)Optional ONNX model-export dependencies installed.$(RESET)"

python-venv: ## Create the optional Python tooling environment
	@echo "$(BLUE)Creating optional Python tooling environment...$(RESET)"
	@command -v "$(UV)" >/dev/null 2>&1 || { echo "uv is required for the locked optional Python tooling environment; install uv first." >&2; exit 1; }
	@$(UV) venv "$(PYTHON_VENV_DIR)"

dev: setup-dirs ## Setup Rust and browser development dependencies without Python
	@command -v cargo >/dev/null || { echo "cargo is required; install the Rust toolchain first."; exit 1; }
	@command -v npm >/dev/null || { echo "npm is required; install Node.js first."; exit 1; }
	cargo fetch --locked --manifest-path $(RUST_MANIFEST)
	cd frontend/web && npm ci
	@bash scripts/verify-development-setup.sh
	@echo "$(GREEN)Rust and browser development environment is ready.$(RESET)"

setup-dirs: ## Create necessary directories
	@echo "$(BLUE)Creating project directories...$(RESET)"
	@mkdir -p $(LOGS_DIR) $(DATA_DIR) $(TEMP_DIR)
	@touch $(LOGS_DIR)/.gitkeep $(DATA_DIR)/.gitkeep $(TEMP_DIR)/.gitkeep
	@echo "$(GREEN)Directories created!$(RESET)"

setup-git-hooks: ## Configure the repository's Python-free shell commit hook
	@git config --local core.hooksPath "$$(git rev-parse --show-toplevel)/.githooks"
	@echo "$(GREEN)Python-free Git commit checks are enabled.$(RESET)"

# ===== Code Quality =====

lint: lint-rust lint-web ## Check Rust and browser code quality

lint-rust: ## Run strict Rust lint checks
	cargo clippy --locked --manifest-path $(RUST_MANIFEST) --all-targets -- -D warnings

lint-web: ## Type-check the browser application
	cd frontend/web && npm run typecheck

lint-python: ## Lint Python compatibility/offline tools; type-check the backend
	$(PYTHON) -m ruff check $(BACKEND_DIR) tools
	$(PYTHON) -m mypy $(BACKEND_DIR)

format: format-rust ## Format Rust source

format-rust: ## Format Rust source
	cargo fmt --manifest-path $(RUST_MANIFEST) --all

format-python: ## Format retained Python compatibility and offline tooling
	$(PYTHON) -m black $(BACKEND_DIR) tools
	$(PYTHON) -m isort $(BACKEND_DIR) tools

format-check: ## Check Rust source formatting
	cargo fmt --manifest-path $(RUST_MANIFEST) --all -- --check

fix: format lint ## Fix code formatting and run linting

# ===== Testing =====

test: test-rust test-web ## Run Rust runtime and browser tests

test-all: test test-rust-camera test-python test-tooling test-deploy ## Run Rust, browser, camera-feature, retained Python, tooling, and deployment tests

test-rust: ## Run Rust unit, contract, and fixture tests
	CARGO_TARGET_DIR="$(RUST_TEST_TARGET_DIR)" cargo test --locked --manifest-path $(RUST_MANIFEST) --all-targets

test-rust-camera: ## Test the production GStreamer camera feature (requires GStreamer development libraries)
	CARGO_TARGET_DIR="$(RUST_CAMERA_TEST_TARGET_DIR)" cargo test --locked --manifest-path $(RUST_MANIFEST) --all-targets --features camera-gstreamer-source

test-web: ## Run browser client tests
	cd frontend/web && npm test

test-python: ## Run retained Python compatibility and model-export tests
	$(PYTHON) -m pytest

test-deploy: ## Exercise Rust distribution packaging, cleanup, and restart scripts
	bash scripts/deploy/tests/build-dist-smoke.sh
	bash scripts/deploy/tests/update-smoke.sh

test-unit: ## Run all retained Python unit/core/common tests without the full-suite coverage gate
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTHON) -m pytest backend/tests/unit backend/tests/core backend/tests/common --no-cov

test-integration: ## Run all retained Python integration tests without the full-suite coverage gate
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest backend/tests/integration --no-cov

test-system: ## Run retained Python system tests without the full-suite coverage gate
	@echo "$(BLUE)Running system tests...$(RESET)"
	$(PYTHON) -m pytest backend/tests/system --no-cov

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest --cov=backend --cov-report=html --cov-report=term-missing

# ===== Build =====

build-frontend: ## Build the frontend web app
	@echo "$(BLUE)Building frontend...$(RESET)"
	@test -x frontend/web/node_modules/.bin/vite || { echo "Browser dependencies are missing; run 'make dev' first." >&2; exit 1; }
	cd frontend/web && npm run build
	@echo "$(GREEN)Frontend built to backend/static/$(RESET)"

build-rust: ## Build Rust server and live perception binaries
	cargo build --manifest-path $(RUST_MANIFEST) --features $(RUST_FEATURES) \
		--bin billiards-server \
		--bin billiards-perception-live \
		--bin billiards-perception-gstreamer-probe

build: build-frontend build-rust ## Build the Rust application and web client

# ===== Development Server =====

run: build-frontend build-rust setup-dirs ## Build and run Rust server, live detector, and retained web client
	@echo "$(BLUE)Starting Billiards Trainer Rust server...$(RESET)"
	cargo run --manifest-path $(RUST_MANIFEST) --features $(RUST_FEATURES) --bin billiards-server -- \
		--http-bind $(RUST_HTTP_BIND) --http-port $(RUST_HTTP_PORT) \
		--scene-bind $(RUST_SCENE_BIND) --scene-port $(RUST_SCENE_PORT) \
		--config "$(abspath $(RUST_CONFIG))" --project-root "$(PWD)" \
		--web-root "$(PWD)/backend/static" --start-live $(RUST_SERVER_ARGS)

run-api: build-frontend setup-dirs ## Run Rust gateway without live capture
	cargo run --manifest-path $(RUST_MANIFEST) --bin billiards-server -- \
		--http-bind $(RUST_HTTP_BIND) --http-port $(RUST_HTTP_PORT) \
		--scene-bind $(RUST_SCENE_BIND) --scene-port $(RUST_SCENE_PORT) \
		--config "$(abspath $(RUST_CONFIG))" --project-root "$(PWD)" \
		--web-root "$(PWD)/backend/static" $(RUST_SERVER_ARGS)

run-frontend: ## Run frontend dev server only
	cd frontend/web && npm run dev

# ===== Configuration =====

config-check: ## Validate configuration files
	@echo "$(BLUE)Checking configuration files...$(RESET)"
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
	@bash scripts/deploy/clean-dist.sh
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
	@echo "$(GREEN)Data files cleaned!$(RESET)"

clean-frontend: ## Clean frontend build artifacts
	@echo "$(BLUE)Cleaning frontend build artifacts...$(RESET)"
	rm -rf backend/static/
	rm -rf frontend/web/dist/
	@echo "$(GREEN)Frontend artifacts cleaned!$(RESET)"

clean-all: clean clean-logs clean-frontend ## Clean generated build, log, and browser assets

# ===== Docker Compose =====

docker-build: ## Build the Rust application image with Docker Compose
	docker compose build backend

docker-run: ## Run the Rust application in the foreground with Docker Compose
	docker compose up backend

docker-run-detached: ## Run the Rust application in the background with Docker Compose
	docker compose up -d backend

docker-stop: ## Stop the Docker Compose application without deleting its data volume
	docker compose down

docker-restart: ## Restart the Docker Compose application
	docker compose restart backend

docker-logs: ## Follow the Docker Compose application logs
	docker compose logs --follow backend

docker-shell: ## Open a shell in the running Docker Compose application
	docker compose exec backend /bin/sh

# ===== Utilities =====

deps-update: ## Update Rust and browser dependency lockfiles
	cargo update --manifest-path $(RUST_MANIFEST)
	cd frontend/web && npm update

deps-check: ## Check locked Rust dependencies and browser package updates
	cargo metadata --locked --manifest-path $(RUST_MANIFEST) --no-deps >/dev/null
	cd frontend/web && npm outdated

security-check: ## Audit Rust and browser dependencies
	@$(MAKE) --no-print-directory security-check-rust
	@$(MAKE) --no-print-directory security-check-web

security-check-rust: ## Audit Rust dependencies (requires cargo-audit)
	@command -v cargo-audit >/dev/null 2>&1 || { echo "$(RED)cargo-audit is required; install cargo-audit before running this check.$(RESET)" >&2; exit 1; }
	cd perception && cargo audit

security-check-web: ## Audit browser dependencies
	cd frontend/web && npm audit --audit-level=high

test-tooling: ## Exercise developer tooling failure and command-routing behavior
	bash scripts/tooling/tests/security-check-smoke.sh
	bash scripts/tests/serve-test-video-mjpeg-smoke.sh

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

# ===== Documentation =====

docs: ## Generate Rust API documentation
	cargo doc --locked --manifest-path $(RUST_MANIFEST) --no-deps

# ===== Deployment =====

deploy: ## Build production distribution package using the Rust shell bundler
	@bash scripts/deploy/build-dist.sh

deploy-clean: ## Clean deployment artifacts
	@bash scripts/deploy/clean-dist.sh
