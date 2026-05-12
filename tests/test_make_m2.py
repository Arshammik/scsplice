"""Tests for scsplice.tl.make_m2.

Tolerance choice: bit-exact (np.array_equal) on CSC indptr/indices/data arrays.
The kernel is purely additive (no transcendentals, no IRLS), parallelises over
columns with disjoint output writes (no reductions, no critical sections), and
uses an indexed dense workspace (not a hash map) so floating-point operations
execute in identical order regardless of thread count or platform.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _openmp_enabled() -> bool:
    try:
        from scsplice import _scsplice_cpp  # noqa: PLC0415
        return bool(_scsplice_cpp.__openmp__)
    except ImportError:
        return False


def test_make_m2_smoke(synthetic_splicing_adata):
    import scsplice  # noqa: PLC0415

    adata = synthetic_splicing_adata(n_events=20, n_cells=50, n_groups=4)
    scsplice.tl.make_m2(adata)

    assert "M2" in adata.layers
    M2 = adata.layers["M2"]
    assert sp.issparse(M2)
    assert M2.format == "csc"
    assert M2.dtype == np.float64
    assert M2.shape == adata.shape
    assert adata.uns["scsplice"]["m2_valid"] is True
    assert adata.uns["scsplice"]["params"]["make_m2"]["n_threads"] == 1


def test_make_m2_singleton_group_produces_zero_row(synthetic_splicing_adata):
    import scsplice  # noqa: PLC0415

    adata = synthetic_splicing_adata(n_events=5, n_cells=20, n_groups=2)
    # Make event 0 a singleton in its own group; renumber remaining events to dense 0..G-1.
    adata.var["group_id"] = np.array([2, 0, 0, 1, 1], dtype=np.int32)
    adata.var["group_count"] = np.array([1, 2, 2, 2, 2], dtype=np.int32)

    scsplice.tl.make_m2(adata)

    # Event 0 (var index 0) is alone in group 2; its row in M2 (cells x events => col 0)
    # must have zero explicit nonzeros.
    M2 = adata.layers["M2"]
    assert M2.getcol(0).nnz == 0


@pytest.mark.openmp
def test_make_m2_thread_determinism(synthetic_splicing_adata):
    if not _openmp_enabled():
        pytest.skip("OpenMP not enabled in this build")
    import scsplice  # noqa: PLC0415

    a1 = synthetic_splicing_adata(n_events=200, n_cells=500, n_groups=20, density=0.2,
                                  seed=42)
    a4 = a1.copy()

    scsplice.tl.make_m2(a1, n_threads=1)
    scsplice.tl.make_m2(a4, n_threads=4)

    M2_1 = a1.layers["M2"]
    M2_4 = a4.layers["M2"]
    assert np.array_equal(M2_1.indptr, M2_4.indptr)
    assert np.array_equal(M2_1.indices, M2_4.indices)
    assert np.array_equal(M2_1.data, M2_4.data)


def test_make_m2_copy_semantic(synthetic_splicing_adata):
    import scsplice  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=8, n_cells=20, n_groups=2)
    result = scsplice.tl.make_m2(a, copy=True)
    assert result is not a
    assert "M2" in result.layers
    # Original adata must remain untouched (m2 not added, m2_valid still False).
    assert "M2" not in a.layers
    assert a.uns["scsplice"]["m2_valid"] is False

    # In-place: returns None, mutates the original.
    a2 = synthetic_splicing_adata(n_events=8, n_cells=20, n_groups=2)
    out = scsplice.tl.make_m2(a2)
    assert out is None
    assert "M2" in a2.layers
    assert a2.uns["scsplice"]["m2_valid"] is True


def test_make_m2_validates_inputs(synthetic_splicing_adata):
    import scsplice  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    del a.layers["M1"]
    with pytest.raises(KeyError, match="M1"):
        scsplice.tl.make_m2(a)

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    a.layers["M1"] = a.layers["M1"].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        scsplice.tl.make_m2(a)

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    a.var = a.var.drop(columns=["group_id"])
    with pytest.raises(KeyError, match="group_id"):
        scsplice.tl.make_m2(a)


def test_make_m2_auto_remaps_sparse_group_ids(synthetic_splicing_adata):
    """Sparse var['group_id'] (e.g., 0,0,2,2,2,2 — group 1 missing) is
    auto-remapped to dense 0..G-1 before crossing the C++ boundary, matching
    R splikit's wrapper behaviour. var['group_id'] itself is left untouched."""
    import scsplice  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=6, n_cells=10, n_groups=2)
    sparse_ids = np.array([0, 0, 2, 2, 2, 2], dtype=np.int32)
    a.var["group_id"] = sparse_ids
    scsplice.tl.make_m2(a)
    assert "M2" in a.layers
    assert a.uns["scsplice"]["m2_valid"] is True
    # var schema preserved verbatim — no auto-rewrite.
    np.testing.assert_array_equal(a.var["group_id"].to_numpy(), sparse_ids)


def test_make_m2_rejects_negative_group_ids(synthetic_splicing_adata):
    import scsplice  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=6, n_cells=10, n_groups=2)
    a.var["group_id"] = np.array([0, -1, 2, 2, 2, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        scsplice.tl.make_m2(a)
