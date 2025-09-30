"""
Ergonomic test runner for the Affine project.

This script provides a convenient interface for running different types of tests
with sensible defaults and common options.

Usage:
    uv run test                    # Run all tests
    uv run test --unit            # Run only unit tests
    uv run test --integration     # Run only integration tests
    uv run test --parallel        # Run tests in parallel
    uv run test --verbose         # Run with verbose output
"""

import sys
import argparse
import subprocess
from pathlib import Path


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Ergonomic test runner for Affine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run test                    # Run all tests
  uv run test --unit            # Run only unit tests (fast)
  uv run test --integration     # Run only integration tests
  uv run test --parallel        # Run tests in parallel
  uv run test --verbose         # Run with verbose output
  uv run test --help-pytest     # Show all pytest options
        """,
    )

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit",
        "-u",
        action="store_true",
        help="Run only unit tests (fast, no external dependencies)",
    )
    test_group.add_argument(
        "--integration",
        "-i",
        action="store_true",
        help="Run only integration tests (slower, uses Docker)",
    )
    test_group.add_argument(
        "--all", "-a", action="store_true", help="Run all tests (default behavior)"
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel using all CPU cores",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests with verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Run tests with minimal output"
    )
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Run tests with coverage reporting",
    )
    parser.add_argument(
        "--help-pytest", action="store_true", help="Show pytest help and options"
    )

    # File/pattern matching
    parser.add_argument(
        "pattern",
        nargs="?",
        help="Test file pattern or specific test (e.g., test_unit_example.py::test_example)",
    )

    args, extra_args = parser.parse_known_args()

    # Handle pytest help
    if args.help_pytest:
        subprocess.run(["python", "-m", "pytest", "--help"])
        return

    # Build pytest command
    pytest_args = ["python", "-m", "pytest"]

    # Add testsuite directory
    testsuite_dir = Path(__file__).parent
    if args.pattern:
        if "/" in args.pattern or args.pattern.endswith(".py"):
            pytest_args.append(str(testsuite_dir / args.pattern))
        else:
            pytest_args.extend([str(testsuite_dir), "-k", args.pattern])
    else:
        pytest_args.append(str(testsuite_dir))

    # Add test type markers
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])

    # Add execution options
    if args.parallel:
        pytest_args.extend(["-n", "auto"])

    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")

    if args.coverage:
        pytest_args.extend(["--cov=affine", "--cov-report=term-missing"])

    # Add any extra arguments passed through
    pytest_args.extend(extra_args)

    # Print command being run (for transparency)
    print(f"Running: {' '.join(pytest_args)}")
    print("=" * 50)

    # Execute pytest
    result = subprocess.run(pytest_args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
