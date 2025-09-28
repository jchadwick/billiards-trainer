#!/usr/bin/env python3
"""
Test runner script for the billiards trainer backend.

This script provides a convenient way to run different test suites with various options.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def setup_test_environment():
    """Set up the test environment."""
    # Add backend to Python path
    backend_dir = Path(__file__).parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # Create test directories if they don't exist
    test_dirs = [
        "backend/tests/logs",
        "backend/tests/data",
        "backend/tests/coverage"
    ]

    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)

    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["BILLIARDS_CONFIG_FILE"] = "tests/test_config.yaml"


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    print("🧪 Running unit tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/unit/", "-m", "unit"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend([
            "--cov=backend",
            "--cov-report=html:backend/tests/coverage/unit",
            "--cov-report=term-missing"
        ])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_integration_tests(verbose=False, coverage=False):
    """Run integration tests."""
    print("🔗 Running integration tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/integration/", "-m", "integration"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend([
            "--cov=backend",
            "--cov-report=html:backend/tests/coverage/integration",
            "--cov-report=term-missing"
        ])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_performance_tests(verbose=False):
    """Run performance tests."""
    print("⚡ Running performance tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/performance/", "-m", "performance"]

    if verbose:
        cmd.extend(["-v", "-s"])

    # Performance tests should not run with coverage (affects timing)
    cmd.extend(["--tb=short"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_system_tests(verbose=False):
    """Run system tests."""
    print("🔄 Running system tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/system/", "-m", "system"]

    if verbose:
        cmd.extend(["-v", "-s"])

    cmd.extend(["--tb=short"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_all_tests(verbose=False, coverage=False, include_slow=False):
    """Run all test suites."""
    print("🚀 Running all tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend([
            "--cov=backend",
            "--cov-report=html:backend/tests/coverage/all",
            "--cov-report=xml:backend/coverage.xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])

    if not include_slow:
        cmd.extend(["-m", "not slow"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_quick_tests(verbose=False):
    """Run quick tests (unit + fast integration)."""
    print("⚡ Running quick tests...")

    cmd = [
        "python", "-m", "pytest",
        "backend/tests/unit/",
        "backend/tests/integration/",
        "-m", "not slow and not hardware"
    ]

    if verbose:
        cmd.extend(["-v"])

    cmd.extend(["--tb=short"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_hardware_tests(verbose=False):
    """Run hardware-dependent tests."""
    print("🔧 Running hardware tests...")

    cmd = ["python", "-m", "pytest", "backend/tests/", "-m", "hardware"]

    if verbose:
        cmd.extend(["-v", "-s"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_lint_checks():
    """Run code linting and formatting checks."""
    print("📝 Running lint checks...")

    # Check if tools are available
    tools = ["black", "isort", "mypy"]
    missing_tools = []

    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)

    if missing_tools:
        print(f"⚠️  Missing tools: {', '.join(missing_tools)}")
        print("Install with: pip install black isort mypy")
        return False

    success = True

    # Run black
    print("  🖤 Running black...")
    result = subprocess.run(["black", "--check", "backend/"], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("    ❌ Black formatting issues found")
        success = False
    else:
        print("    ✅ Black formatting OK")

    # Run isort
    print("  📚 Running isort...")
    result = subprocess.run(["isort", "--check-only", "backend/"], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("    ❌ Import sorting issues found")
        success = False
    else:
        print("    ✅ Import sorting OK")

    # Run mypy
    print("  🐍 Running mypy...")
    result = subprocess.run(["mypy", "backend/"], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("    ❌ Type checking issues found")
        success = False
    else:
        print("    ✅ Type checking OK")

    return success


def generate_test_report():
    """Generate a comprehensive test report."""
    print("📊 Generating test report...")

    cmd = [
        "python", "-m", "pytest",
        "backend/tests/",
        "--html=backend/tests/coverage/report.html",
        "--self-contained-html",
        "--cov=backend",
        "--cov-report=html:backend/tests/coverage/html",
        "--cov-report=xml:backend/coverage.xml",
        "--junit-xml=backend/tests/coverage/junit.xml",
        "-m", "not slow and not hardware"
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode == 0:
        print("✅ Test report generated successfully!")
        print("📁 Reports available at:")
        print("   - HTML: backend/tests/coverage/report.html")
        print("   - Coverage: backend/tests/coverage/html/index.html")
        print("   - JUnit XML: backend/tests/coverage/junit.xml")

    return result.returncode == 0


def run_ci_tests():
    """Run tests suitable for CI environment."""
    print("🤖 Running CI tests...")

    cmd = [
        "python", "-m", "pytest",
        "backend/tests/",
        "--cov=backend",
        "--cov-report=xml:backend/coverage.xml",
        "--cov-report=term",
        "--cov-fail-under=80",
        "--junit-xml=backend/junit.xml",
        "-m", "not hardware and not slow",
        "--tb=short",
        "-q"
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def watch_tests():
    """Watch for file changes and run tests automatically."""
    try:
        import watchdog
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("❌ Watchdog not installed. Install with: pip install watchdog")
        return False

    print("👀 Watching for changes... (Press Ctrl+C to stop)")

    class TestHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_run = 0

        def on_modified(self, event):
            if event.is_directory:
                return

            if not event.src_path.endswith('.py'):
                return

            # Debounce - don't run tests too frequently
            now = time.time()
            if now - self.last_run < 2:
                return

            self.last_run = now
            print(f"\n📝 File changed: {event.src_path}")
            run_quick_tests()

    event_handler = TestHandler()
    observer = Observer()
    observer.schedule(event_handler, "backend/", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Stopped watching")

    observer.join()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for billiards trainer backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests
  python run_tests.py --all --coverage          # Run all tests with coverage
  python run_tests.py --quick --verbose         # Run quick tests with verbose output
  python run_tests.py --ci                      # Run CI-suitable tests
  python run_tests.py --lint                    # Run lint checks
  python run_tests.py --report                  # Generate test report
  python run_tests.py --watch                   # Watch and run tests on changes
        """
    )

    # Test suite selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests")
    test_group.add_argument("--system", action="store_true", help="Run system tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--quick", action="store_true", help="Run quick tests (unit + fast integration)")
    test_group.add_argument("--hardware", action="store_true", help="Run hardware-dependent tests")
    test_group.add_argument("--ci", action="store_true", help="Run CI-suitable tests")
    test_group.add_argument("--watch", action="store_true", help="Watch for changes and run tests")

    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--lint", action="store_true", help="Run lint checks")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")

    args = parser.parse_args()

    # If no test suite specified, run quick tests
    if not any([args.unit, args.integration, args.performance, args.system,
                args.all, args.quick, args.hardware, args.ci, args.watch]):
        args.quick = True

    # Set up test environment
    setup_test_environment()

    success = True

    # Run lint checks if requested
    if args.lint:
        success = run_lint_checks() and success

    # Run tests
    if args.unit:
        success = run_unit_tests(args.verbose, args.coverage) and success
    elif args.integration:
        success = run_integration_tests(args.verbose, args.coverage) and success
    elif args.performance:
        success = run_performance_tests(args.verbose) and success
    elif args.system:
        success = run_system_tests(args.verbose) and success
    elif args.all:
        success = run_all_tests(args.verbose, args.coverage, args.slow) and success
    elif args.quick:
        success = run_quick_tests(args.verbose) and success
    elif args.hardware:
        success = run_hardware_tests(args.verbose) and success
    elif args.ci:
        success = run_ci_tests() and success
    elif args.watch:
        success = watch_tests() and success

    # Generate report if requested
    if args.report:
        success = generate_test_report() and success

    # Exit with appropriate code
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
