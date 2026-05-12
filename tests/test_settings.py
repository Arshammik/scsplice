"""scsplice.settings global knobs."""

from __future__ import annotations


def test_settings_defaults():
    import scsplice  # noqa: PLC0415

    assert scsplice.settings.verbosity == 1
    assert scsplice.settings.n_jobs == 1


def test_settings_mutable():
    import scsplice  # noqa: PLC0415

    original = scsplice.settings.verbosity
    try:
        scsplice.settings.verbosity = 3
        assert scsplice.settings.verbosity == 3
        scsplice.settings.n_jobs = 8
        assert scsplice.settings.n_jobs == 8
    finally:
        scsplice.settings.verbosity = original
        scsplice.settings.n_jobs = 1


def test_settings_singleton():
    import scsplice  # noqa: PLC0415
    from scsplice._settings import settings as alt

    assert scsplice.settings is alt
