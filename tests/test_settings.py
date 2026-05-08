"""splikit.settings global knobs."""

from __future__ import annotations


def test_settings_defaults():
    import splikit  # noqa: PLC0415

    assert splikit.settings.verbosity == 1
    assert splikit.settings.n_jobs == 1


def test_settings_mutable():
    import splikit  # noqa: PLC0415

    original = splikit.settings.verbosity
    try:
        splikit.settings.verbosity = 3
        assert splikit.settings.verbosity == 3
        splikit.settings.n_jobs = 8
        assert splikit.settings.n_jobs == 8
    finally:
        splikit.settings.verbosity = original
        splikit.settings.n_jobs = 1


def test_settings_singleton():
    import splikit  # noqa: PLC0415
    from splikit._settings import settings as alt

    assert splikit.settings is alt
