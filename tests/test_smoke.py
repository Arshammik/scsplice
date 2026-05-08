"""Sanity-check that the package imports cleanly and exposes the expected namespaces."""

from __future__ import annotations


def test_package_imports():
    import splikit

    assert hasattr(splikit, "__version__")
    assert isinstance(splikit.__version__, str)


def test_namespaces_present():
    import splikit

    for ns in ("io", "tl", "pp", "pl"):
        assert hasattr(splikit, ns), f"splikit.{ns} missing from public API"
