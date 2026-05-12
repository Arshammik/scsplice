"""Sanity-check that the package imports cleanly and exposes the expected namespaces."""

from __future__ import annotations


def test_package_imports():
    import scsplice

    assert hasattr(scsplice, "__version__")
    assert isinstance(scsplice.__version__, str)


def test_namespaces_present():
    import scsplice

    for ns in ("io", "tl", "pp", "pl"):
        assert hasattr(scsplice, ns), f"scsplice.{ns} missing from public API"
