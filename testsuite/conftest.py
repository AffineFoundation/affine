"""
Shared pytest configuration and fixtures for Affine testsuite.

This module contains fixtures and configuration that are shared across
multiple test modules in the testsuite.
"""

import time
import subprocess
from pathlib import Path

import pytest
import requests
from testcontainers.compose import DockerCompose


class DockerComposeManager:
    """Manages Docker Compose lifecycle for integration testing."""

    def __init__(self):
        self.compose = None
        self.affine_dir = Path(__file__).parent.parent
        self.compose_file = self.affine_dir / "docker-compose.yml"

    def cleanup_existing_containers(self):
        """Clean up any existing affine containers."""
        print("Cleaning up existing containers...")

        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=affine"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                container_ids = result.stdout.strip().split("\n")
                print(f"Stopping {len(container_ids)} running containers...")
                subprocess.run(
                    ["docker", "stop"] + container_ids, capture_output=True, check=False
                )
                subprocess.run(
                    ["docker", "rm"] + container_ids, capture_output=True, check=False
                )

            subprocess.run(
                ["docker", "container", "prune", "-f"], capture_output=True, check=False
            )
            print("Cleanup completed")

        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def start_services(self):
        """Start Docker Compose services."""
        print("Starting Docker Compose services...")

        self.compose = DockerCompose(
            str(self.affine_dir), compose_file_name="docker-compose.yml", pull=False
        )
        self.compose.start()

        print("Services starting up...")
        time.sleep(10)  # Allow services to initialize

    def stop_services(self):
        """Stop Docker Compose services."""
        if self.compose:
            print("Stopping services...")
            self.compose.stop()

    def wait_for_service(
        self, url: str, max_retries: int = 30, delay: float = 2.0
    ) -> bool:
        """Wait for a service to become ready."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 302]:
                    return True
            except requests.RequestException:
                pass

            if attempt < max_retries - 1:
                print(f"  Waiting for {url} (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

        return False


@pytest.fixture(scope="session")
def docker_compose_manager():
    """Pytest fixture for Docker Compose management shared across all integration tests."""
    manager = DockerComposeManager()

    # Setup
    manager.cleanup_existing_containers()
    manager.start_services()

    yield manager

    # Teardown
    manager.stop_services()
    manager.cleanup_existing_containers()


@pytest.fixture(scope="session")
def service_endpoints():
    """Define the standard service endpoints for testing."""
    return {
        "Affine Validator": {
            "url": "http://localhost:8001",
            "expected_status": [200],
            "description": "Prometheus metrics endpoint",
        },
        "Affine Runner": {
            "url": "http://localhost:8002",
            "expected_status": [200],
            "description": "Prometheus metrics endpoint",
        },
        "Prometheus": {
            "url": "http://localhost:9090",
            "expected_status": [200, 302],
            "description": "Prometheus web interface",
        },
        "Grafana": {
            "url": "http://localhost:8000",
            "expected_status": [200, 302],
            "description": "Grafana dashboard",
        },
        "Affine Signer Health": {
            "url": "http://localhost:8080/healthz",
            "expected_status": [200],
            "description": "Signer health check",
        },
    }
