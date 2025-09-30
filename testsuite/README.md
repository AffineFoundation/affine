# Affine Test Suite

This directory contains the comprehensive test suite for the Affine project using modern pytest-based testing frameworks.

## File Naming Convention

The testsuite follows a clear naming convention to organize different types of tests:

### Test Files
- `test_unit_*.py` - Unit tests (fast, isolated, no external dependencies)
- `test_integration_*.py` - Integration tests (slower, may use Docker/external services)
- `test_e2e_*.py` - End-to-end tests (full system tests)
- `test_performance_*.py` - Performance and load tests

### Support Files
- `conftest.py` - Shared pytest fixtures and configuration
- `helpers/` - Test utility functions and helper classes
- `fixtures/` - Test data and fixture files

### Current Test Files
- `test_integration_docker_compose.py` - Docker Compose stack integration tests
- `test_unit_example.py` - Example unit tests (template)

## Usage

From the `affine` directory, use the ergonomic test runner:

### Ergonomic Test Commands (Recommended)
```bash
# Run all tests
uv run test

# Run only unit tests (fast)
uv run test --unit

# Run only integration tests
uv run test --integration

# Run tests in parallel
uv run test --parallel
```

### Advanced pytest Commands
```bash
# Run all tests with pytest directly
uv run --extra dev pytest testsuite/

# Run by test type
uv run --extra dev pytest testsuite/ -m unit
uv run --extra dev pytest testsuite/ -m integration

# Run specific test files
uv run --extra dev pytest testsuite/test_integration_docker_compose.py -v
uv run --extra dev pytest testsuite/test_unit_example.py -v
```

## What the tests do

1. **Cleanup**: Removes any zombie Docker processes and containers before starting
2. **Service Startup**: Uses testcontainers to start the Docker Compose cluster
3. **Health Checks**: Waits for all critical services to become ready
4. **Endpoint Testing**: Verifies all HTTP endpoints are accessible:
   - Affine Validator: `http://localhost:8001` (Prometheus metrics)
   - Affine Runner: `http://localhost:8002` (Prometheus metrics)
   - Prometheus: `http://localhost:9090` (redirects to /query)
   - Grafana: `http://localhost:8000` (redirects to /login)
   - Affine Signer: `http://localhost:8080/healthz` (health check)
5. **Cleanup**: Removes containers and processes after testing

## Testing Framework

This testsuite uses state-of-the-art Python testing tools:

- **pytest**: Modern testing framework with excellent reporting
- **testcontainers**: Docker Compose lifecycle management
- **rich**: Beautiful terminal output and formatting
- **pytest-xdist**: Parallel test execution support
- **requests**: HTTP endpoint testing

## Requirements

- Python 3.9+
- Docker and Docker Compose
- `uv` package manager

## Features

- **Parametrized tests**: Each endpoint tested individually
- **Session fixtures**: Docker Compose managed across all tests
- **Rich output**: Beautiful terminal formatting and tables
- **Integration markers**: Run specific test categories
- **Parallel execution**: Run tests in parallel with `-n auto`
- **Comprehensive reporting**: Detailed test summaries and tables
