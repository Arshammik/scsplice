"""Tests for splikit.tl.make_m2.

Tolerance choice: bit-exact (np.array_equal) on CSC indptr/indices/data arrays.
The kernel is purely additive (no transcendentals, no IRLS), parallelises over
columns with disjoint output writes (no reductions, no critical sections), and
uses an indexed dense workspace (not a hash map) so floating-point operations
execute in identical order regardless of thread count or platform. See the
R/Rcpp port spec for the determinism guarantee.

The R-equivalence test is gated behind pytest.mark.r_required and skipped when
tests/data/r_reference.h5 is missing. Regenerate with:

    Rscript tests/r_export/export_reference.R tests/data/r_reference.h5
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _openmp_enabled() -> bool:
    try:
        from splikit import _splikit_cpp  # noqa: PLC0415
        return bool(_splikit_cpp.__openmp__)
    except ImportError:
        return False


def test_make_m2_smoke(synthetic_splicing_adata):
    import splikit  # noqa: PLC0415

    adata = synthetic_splicing_adata(n_events=20, n_cells=50, n_groups=4)
    splikit.tl.make_m2(adata)

    assert "M2" in adata.layers
    M2 = adata.layers["M2"]
    assert sp.issparse(M2)
    assert M2.format == "csc"
    assert M2.dtype == np.float64
    assert M2.shape == adata.shape
    assert adata.uns["splikit"]["m2_valid"] is True
    assert adata.uns["splikit"]["params"]["make_m2"]["n_threads"] == 1


def test_make_m2_singleton_group_produces_zero_row(synthetic_splicing_adata):
    import splikit  # noqa: PLC0415

    adata = synthetic_splicing_adata(n_events=5, n_cells=20, n_groups=2)
    # Make event 0 a singleton in its own group; renumber remaining events to dense 0..G-1.
    adata.var["group_id"] = np.array([2, 0, 0, 1, 1], dtype=np.int32)
    adata.var["group_count"] = np.array([1, 2, 2, 2, 2], dtype=np.int32)

    splikit.tl.make_m2(adata)

    # Event 0 (var index 0) is alone in group 2; its row in M2 (cells x events => col 0)
    # must have zero explicit nonzeros.
    M2 = adata.layers["M2"]
    assert M2.getcol(0).nnz == 0


@pytest.mark.openmp
def test_make_m2_thread_determinism(synthetic_splicing_adata):
    if not _openmp_enabled():
        pytest.skip("OpenMP not enabled in this build")
    import splikit  # noqa: PLC0415

    a1 = synthetic_splicing_adata(n_events=200, n_cells=500, n_groups=20, density=0.2,
                                  seed=42)
    a4 = a1.copy()

    splikit.tl.make_m2(a1, n_threads=1)
    splikit.tl.make_m2(a4, n_threads=4)

    M2_1 = a1.layers["M2"]
    M2_4 = a4.layers["M2"]
    assert np.array_equal(M2_1.indptr, M2_4.indptr)
    assert np.array_equal(M2_1.indices, M2_4.indices)
    assert np.array_equal(M2_1.data, M2_4.data)


def test_make_m2_copy_semantic(synthetic_splicing_adata):
    import splikit  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=8, n_cells=20, n_groups=2)
    result = splikit.tl.make_m2(a, copy=True)
    assert result is not a
    assert "M2" in result.layers
    # Original adata must remain untouched (m2 not added, m2_valid still False).
    assert "M2" not in a.layers
    assert a.uns["splikit"]["m2_valid"] is False

    # In-place: returns None, mutates the original.
    a2 = synthetic_splicing_adata(n_events=8, n_cells=20, n_groups=2)
    out = splikit.tl.make_m2(a2)
    assert out is None
    assert "M2" in a2.layers
    assert a2.uns["splikit"]["m2_valid"] is True


def test_make_m2_validates_inputs(synthetic_splicing_adata):
    import splikit  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    del a.layers["M1"]
    with pytest.raises(KeyError, match="M1"):
        splikit.tl.make_m2(a)

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    a.layers["M1"] = a.layers["M1"].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        splikit.tl.make_m2(a)

    a = synthetic_splicing_adata(n_events=4, n_cells=10, n_groups=2)
    a.var = a.var.drop(columns=["group_id"])
    with pytest.raises(KeyError, match="group_id"):
        splikit.tl.make_m2(a)


def test_make_m2_rejects_sparse_group_ids(synthetic_splicing_adata):
    import splikit  # noqa: PLC0415

    a = synthetic_splicing_adata(n_events=6, n_cells=10, n_groups=2)
    # Skip group 1: 0..2 with 1 missing.
    a.var["group_id"] = np.array([0, 0, 2, 2, 2, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="dense"):
        splikit.tl.make_m2(a)


@pytest.mark.r_required
@pytest.mark.slow
def test_make_m2_bit_exact_vs_r(r_reference_path: Path):
    if not r_reference_path.exists():
        pytest.skip(
            f"R reference fixture not found at {r_reference_path}. "
            "Regenerate via Rscript tests/r_export/export_reference.R"
        )

    from tests.load_r_ref import load_reference  # noqa: PLC0415
    import splikit  # noqa: PLC0415

    ref = load_reference(r_reference_path)

    # R reference is events x cells; AnnData is cells x events. Transpose.
    M1_T = ref.m1.T.tocsc().astype(np.float64)
    n_cells, n_events = M1_T.shape

    var = pd.DataFrame(
        {
            "chr": pd.Categorical(
                ref.chr_arr.astype(str) if ref.chr_arr is not None
                else np.array(["chr0"] * n_events)
            ),
            "start": (ref.start.astype(np.int64) if ref.start is not None
                      else np.arange(n_events, dtype=np.int64)),
            "end": (ref.end.astype(np.int64) if ref.end is not None
                    else np.arange(n_events, dtype=np.int64) + 1),
            "strand": pd.Categorical(
                ref.strand.astype(str) if ref.strand is not None
                else np.array(["+"] * n_events)
            ),
            "row_names_mtx": ref.row_names_mtx,
            "group_id": ref.group_id.astype(np.int32),
            "group_kind": pd.Categorical(ref.group_kind, categories=["S", "E"]),
            "group_count": ref.group_count.astype(np.int32),
        },
        index=[f"{name}_{kind}" for name, kind in
               zip(ref.row_names_mtx, ref.group_kind, strict=False)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"bc{i}" for i in range(n_cells)],
         "sample_id": ["s1"] * n_cells},
        index=[f"bc{i}-s1" for i in range(n_cells)],
    )
    adata = ad.AnnData(layers={"M1": M1_T}, obs=obs, var=var)
    adata.uns["splikit"] = {"m2_valid": False}

    splikit.tl.make_m2(adata, n_threads=1)

    M2_py = adata.layers["M2"].T.tocsc()
    M2_r = ref.m2.tocsc()

    indptr_eq = np.array_equal(M2_py.indptr, M2_r.indptr)
    indices_eq = np.array_equal(M2_py.indices, M2_r.indices)
    data_eq = np.array_equal(M2_py.data, M2_r.data)

    assert indptr_eq, "indptr arrays differ"
    assert indices_eq, "indices arrays differ"

    if not data_eq:
        diff = np.abs(M2_py.data - M2_r.data)
        idx = int(np.argmax(diff))
        max_diff = float(diff[idx])
        if os.environ.get("SPLIKIT_M2_TOLERANCE_FALLBACK") == "1":
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"Bit-exact data comparison failed (max abs diff {max_diff:.3e} "
                f"at index {idx}); falling back to atol=1e-15.",
                stacklevel=2,
            )
            assert np.allclose(M2_py.data, M2_r.data, rtol=0, atol=1e-15)
        else:
            raise AssertionError(
                f"M2.data differs from R reference. Max abs diff {max_diff:.3e} "
                f"at index {idx} (py={M2_py.data[idx]!r}, r={M2_r.data[idx]!r}). "
                "Set SPLIKIT_M2_TOLERANCE_FALLBACK=1 to fall back to allclose for "
                "diagnostic runs."
            )
