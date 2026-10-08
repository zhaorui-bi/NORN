"""Run numerical and end-to-end tests; install the dev extra first."""

from pathlib import Path
import sys

try:
    import pytest
except ImportError:
    raise SystemExit("Install norn-earth[dev] to run the complete test suite")
raise SystemExit(pytest.main([str(Path(__file__).resolve().parents[1] / "tests"), *sys.argv[1:]]))
