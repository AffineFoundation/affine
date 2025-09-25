"""
Integration tests for Docker Compose services.

Tests the complete Docker Compose stack including:
- Affine Validator service
- Affine Runner service
- Prometheus monitoring
- Grafana dashboards
- Affine Signer service

Naming convention: test_integration_*.py for integration tests
"""

import pytest
import requests


@pytest.mark.integration
def test_services_are_ready(docker_compose_manager, service_endpoints):
    """Test that all critical services are ready and responding."""
    print("\nVerifying service readiness...")

    critical_services = [
        "http://localhost:8080/healthz",  # Signer (dependency)
        "http://localhost:8001",  # Validator
        "http://localhost:8002",  # Runner
    ]

    results = []
    for service_url in critical_services:
        is_ready = docker_compose_manager.wait_for_service(service_url)
        results.append((service_url, is_ready))

        if is_ready:
            print(f"READY: {service_url}")
        else:
            print(f"FAIL: {service_url}")

    # Assert all critical services are ready
    failed_services = [url for url, ready in results if not ready]
    assert not failed_services, f"Critical services failed to start: {failed_services}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "service_name,config",
    [
        (
            "Affine Validator",
            {"url": "http://localhost:8001", "expected_status": [200]},
        ),
        ("Affine Runner", {"url": "http://localhost:8002", "expected_status": [200]}),
        ("Prometheus", {"url": "http://localhost:9090", "expected_status": [200, 302]}),
        ("Grafana", {"url": "http://localhost:8000", "expected_status": [200, 302]}),
        (
            "Affine Signer Health",
            {"url": "http://localhost:8080/healthz", "expected_status": [200]},
        ),
    ],
)
def test_endpoint_accessibility(service_name, config, docker_compose_manager):
    """Test individual endpoint accessibility."""
    url = config["url"]
    expected_status = config["expected_status"]

    try:
        response = requests.get(url, timeout=10)

        assert response.status_code in expected_status, (
            f"{service_name}: Expected status {expected_status}, got {response.status_code}"
        )

        # Log successful response
        content_preview = (
            response.text[:100].replace("\n", " ") if response.text else "No content"
        )
        print(f"PASS {service_name}: {response.status_code} - {content_preview}")

    except requests.RequestException as e:
        pytest.fail(f"{service_name}: Connection failed - {str(e)}")


@pytest.mark.integration
def test_all_endpoints_summary(docker_compose_manager, service_endpoints):
    """Generate a comprehensive test summary table."""
    print("\nEndpoint Test Summary")
    print("=" * 80)

    for name, config in service_endpoints.items():
        try:
            response = requests.get(config["url"], timeout=10)
            status = (
                "PASS" if response.status_code in config["expected_status"] else "FAIL"
            )
            status_code = str(response.status_code)
            content_preview = (
                response.text[:50].replace("\n", " ") if response.text else "No content"
            )

        except requests.RequestException as e:
            status = "FAIL"
            status_code = "N/A"
            content_preview = f"Connection failed: {str(e)[:50]}"

        print(
            f"{name:20} | {config['url']:30} | {status} ({status_code:3}) | {content_preview[:30]}"
        )

    print("=" * 80)


if __name__ == "__main__":
    # Allow running directly for development
    pytest.main([__file__, "-v", "--tb=short"])
