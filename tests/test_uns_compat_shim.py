"""Tests for the ``uns['splikit']`` -> ``uns['scsplice']`` compat shim.

Package rename: ``splikit-py`` 1.0.0 wrote ``adata.uns['splikit'] = {...}``.
``scsplice`` 1.x writes the canonical ``uns['scsplice']`` key. The shim lives
in :mod:`scsplice._core._validators` and migrates legacy AnnData objects in
place on first read, emitting a one-shot ``FutureWarning``.

The full ``tl.make_m2`` path is the most realistic exercise of the shim:
its precondition validator (``validate_var_schema``) is fast and the param-
record write at the end touches the namespace dict — together they cover
both read and write sides.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scsplice._core._validators import (
    get_scsplice_ns,
    setdefault_scsplice_ns,
)


def _make_adata_for_make_m2() -> ad.AnnData:
    """Tiny well-formed splicing AnnData; pre-rename schema except for uns key."""
    M1 = sp.csc_matrix(
        np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    )  # 3 events x 2 cells
    # AnnData is cells x events; transpose.
    var = pd.DataFrame(
        {
            "chr": ["chr1"] * 3,
            "start": [0, 100, 200],
            "end": [50, 150, 250],
            "strand": ["+"] * 3,
            "row_names_mtx": ["chr1:0-50", "chr1:100-150", "chr1:200-250"],
            "group_id": np.array([0, 0, 1], dtype=np.int32),
            "group_kind": pd.Categorical(["S"] * 3, categories=["S", "E"]),
            "group_count": np.array([2, 2, 1], dtype=np.int32),
        },
        index=["chr1:0-50_S", "chr1:100-150_S", "chr1:200-250_S"],
    )
    obs = pd.DataFrame(
        {"barcode": ["bc0", "bc1"], "sample_id": ["s1", "s1"]},
        index=["bc0", "bc1"],
    )
    return ad.AnnData(layers={"M1": M1.T.tocsc()}, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Direct helper-level tests (independent of the C++ extension).
# ---------------------------------------------------------------------------


def test_get_scsplice_ns_returns_new_key_directly():
    a = _make_adata_for_make_m2()
    a.uns["scsplice"] = {"m2_valid": True}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = get_scsplice_ns(a)
    assert ns == {"m2_valid": True}
    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert legacy == []


def test_get_scsplice_ns_migrates_legacy_key_with_warning():
    a = _make_adata_for_make_m2()
    a.uns["splikit"] = {"m2_valid": True, "params": {"foo": 1}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = get_scsplice_ns(a)
    assert ns == {"m2_valid": True, "params": {"foo": 1}}
    assert "scsplice" in a.uns
    assert "splikit" not in a.uns, "legacy key must be popped after migration"
    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert len(legacy) == 1, f"expected exactly one FutureWarning, got {len(legacy)}"
    assert "uns['splikit']" in str(legacy[0].message)


def test_get_scsplice_ns_returns_empty_when_neither_key_present():
    a = _make_adata_for_make_m2()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = get_scsplice_ns(a)
    assert ns == {}
    assert "scsplice" not in a.uns
    assert "splikit" not in a.uns
    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert legacy == []


def test_setdefault_scsplice_ns_creates_when_absent():
    a = _make_adata_for_make_m2()
    ns = setdefault_scsplice_ns(a)
    assert ns == {}
    assert "scsplice" in a.uns
    # Mutation through the returned dict is reflected in the namespace.
    ns["m2_valid"] = True
    assert a.uns["scsplice"]["m2_valid"] is True


def test_setdefault_scsplice_ns_migrates_legacy():
    a = _make_adata_for_make_m2()
    a.uns["splikit"] = {"m2_valid": False, "version": 1}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = setdefault_scsplice_ns(a)
    assert ns == {"m2_valid": False, "version": 1}
    assert "scsplice" in a.uns and "splikit" not in a.uns
    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert len(legacy) == 1


# ---------------------------------------------------------------------------
# End-to-end through tl.make_m2 (the canonical public-API entry point).
# ---------------------------------------------------------------------------


def _can_import_cpp() -> bool:
    try:
        import scsplice._scsplice_cpp  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _can_import_cpp(),
    reason="C++ extension _scsplice_cpp not built; tl.make_m2 path unavailable",
)
def test_legacy_uns_key_migrates_with_warning():
    """tl.make_m2 reads + writes the namespace; legacy key migrates with one warning."""
    import scsplice  # noqa: PLC0415

    a = _make_adata_for_make_m2()
    a.uns["splikit"] = {"m2_valid": False}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scsplice.tl.make_m2(a)

    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert len(legacy) >= 1, "FutureWarning must fire on first legacy-key read"
    # After migration the legacy key is gone and the canonical key carries
    # both the m2_valid flag (now True) and the params record.
    assert "scsplice" in a.uns
    assert "splikit" not in a.uns, "old key leaked after migration"
    assert a.uns["scsplice"]["m2_valid"] is True
    assert "params" in a.uns["scsplice"]


@pytest.mark.skipif(
    not _can_import_cpp(),
    reason="C++ extension _scsplice_cpp not built; tl.make_m2 path unavailable",
)
def test_new_uns_key_no_warning():
    """Same flow with the canonical key set must NOT emit a FutureWarning."""
    import scsplice  # noqa: PLC0415

    a = _make_adata_for_make_m2()
    a.uns["scsplice"] = {"m2_valid": False}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scsplice.tl.make_m2(a)

    legacy = [x for x in w if issubclass(x.category, FutureWarning)]
    assert legacy == [], (
        f"no FutureWarning expected with canonical uns key, got: "
        f"{[str(x.message) for x in legacy]}"
    )


@pytest.mark.skipif(
    not _can_import_cpp(),
    reason="C++ extension _scsplice_cpp not built; tl.make_m2 path unavailable",
)
def test_warning_emitted_once_per_adata():
    """The shim migrates in place, so subsequent calls see only the new key."""
    import scsplice  # noqa: PLC0415

    a = _make_adata_for_make_m2()
    a.uns["splikit"] = {"m2_valid": False}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scsplice.tl.make_m2(a)
        first_legacy = [x for x in w if issubclass(x.category, FutureWarning)]
        n_after_first = len(first_legacy)
        # Invalidate so we can call make_m2 again without short-circuiting.
        a.uns["scsplice"]["m2_valid"] = False
        scsplice.tl.make_m2(a)
        all_legacy = [x for x in w if issubclass(x.category, FutureWarning)]

    assert n_after_first >= 1
    # No new FutureWarning on the second call — legacy key already migrated.
    assert len(all_legacy) == n_after_first
