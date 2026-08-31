"""Smoke test: proves the package imports and the test runner works."""

import docmind


def test_package_has_version() -> None:
    assert docmind.__version__ == "0.1.0"
