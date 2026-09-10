.PHONY: install dev run test coverage lint typecheck format check clean help bootstrap bundle-frontend

# Variables
PYTHON := uv run python
PYTEST := uv run pytest
BLACK := uv run black
PYLINT := uv run pylint
MYPY := uv run mypy
SRC_DIR := src/finance_ai
TEST_DIR := tests

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Personal Finance AI - Development Commands"
	@echo "==========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	@echo "Installing dependencies..."
	uv sync --group dev
	@echo "Setting up pre-commit hooks..."
	uv run pre-commit install
	@echo "Done! Run 'make test' to verify setup."

dev:  ## Run development server with auto-reload
	@echo "Starting development server..."
	uv run uvicorn finance_ai.main:app --reload --host 0.0.0.0 --port 8080

run:  ## Run production server
	@echo "Starting production server..."
	uv run uvicorn finance_ai.main:app --host 0.0.0.0 --port 8080 --workers 4

test:  ## Run tests with coverage
	@echo "Running tests..."
	$(PYTEST)

coverage:  ## Generate HTML coverage report
	@echo "Generating coverage report..."
	$(PYTEST) --cov-report=html
	@echo "Coverage report generated at: htmlcov/index.html"

lint:  ## Run linting checks
	@echo "Running pylint..."
	$(PYLINT) $(SRC_DIR) $(TEST_DIR) --output-format=colorized
	@echo "Linting passed! (Score >= 9.0)"

typecheck:  ## Run type checking
	@echo "Running mypy type checker..."
	$(MYPY) $(SRC_DIR) $(TEST_DIR)
	@echo "Type checking passed!"

format:  ## Format code with black
	@echo "Formatting code with black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@echo "Code formatted!"

format-check:  ## Check if code is formatted (CI mode)
	@echo "Checking code formatting..."
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)

check: format lint typecheck test  ## Run all quality checks
	@echo "========================================"
	@echo "All checks passed! Ready to commit."
	@echo "========================================"

clean:  ## Clean generated files
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage build/ dist/ 2>/dev/null || true
	@echo "Clean complete!"

bundle-frontend:  ## Bundle frontend for iHost: make bundle-frontend API_BASE=https://<app>.onrender.com
	@echo "Bundling frontend for iHost upload..."
	$(PYTHON) scripts/bundle_frontend.py --api-base "$(API_BASE)" --output dist/ihost
	@echo "Bundle ready at dist/ihost — upload its contents to the iHost web root."

bootstrap:  ## Prepare runtime: apply migrations + index RAG knowledge base
	@echo "Running runtime bootstrap..."
	$(PYTHON) scripts/bootstrap_runtime.py
	@echo "Bootstrap complete!"

init-db:  ## Initialize database with migrations
	@echo "Initializing database..."
	$(PYTHON) -m alembic upgrade head
	@echo "Database initialized!"

migrate:  ## Create new migration
	@echo "Creating migration..."
	@read -p "Migration message: " msg; \
	$(PYTHON) -m alembic revision --autogenerate -m "$$msg"

shell:  ## Open IPython shell with app context
	@echo "Opening IPython shell..."
	$(PYTHON) -m IPython

evaluate:  ## Run all evaluations
	@echo "Running evaluation framework..."
	$(PYTHON) -m finance_ai.evaluation.cli --eval all

evaluate-routing:  ## Run routing evaluation only
	$(PYTHON) -m finance_ai.evaluation.cli --eval routing

evaluate-rag:  ## Run RAG evaluation only
	$(PYTHON) -m finance_ai.evaluation.cli --eval rag

evaluate-compare:  ## Compare multiple models
	$(PYTHON) -m finance_ai.evaluation.cli --provider google --eval all
	$(PYTHON) -m finance_ai.evaluation.cli --provider openrouter --eval all

.PHONY: all evaluate evaluate-routing evaluate-rag evaluate-compare
all: check  ## Alias for 'check' target
