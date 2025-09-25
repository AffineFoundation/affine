"""
Unit tests example for Affine components.

This module demonstrates the naming convention for unit tests.
Unit tests should be fast, isolated, and not depend on external services.

Naming convention: test_unit_*.py for unit tests
"""

import pytest


@pytest.mark.unit
def test_example_unit_test():
    """Example unit test that runs quickly without external dependencies."""
    # This is just an example - replace with actual unit tests
    assert 1 + 1 == 2


@pytest.mark.unit
def test_another_unit_example():
    """Another example unit test."""
    test_data = {"key": "value"}
    assert test_data.get("key") == "value"
    assert test_data.get("missing") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_val,expected",
    [
        (0, False),
        (1, True),
        (-1, True),
        (100, True),
    ],
)
def test_parametrized_unit_test(input_val, expected):
    """Example of parametrized unit test."""
    result = bool(input_val)
    assert result == expected
