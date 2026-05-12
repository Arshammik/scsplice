"""Tests for scsplice.tl.pseudo_correlation.

Per-event results are bit-identical across thread counts (disjoint scalar
writes; no cross-event reductions in the kernel).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _openmp_enabled() -> bool:
    try:
        from scsplice import _scsplice_cpp  # noqa: PLC0415
        return bool(_scsplice_cpp.__openmp__)
    except ImportError:
        return False


def _make_adata(n_events: int = 25, n_cells: int = 200, *, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    M1_dense = (rng.random((n_cells, n_events)) < 0.5).astype(np.float64)
    M1_dense *= rng.integers(1, 30, size=M1_dense.shape).astype(np.float64)
    M2_dense = (rng.random((n_cells, n_events)) < 0.5).astype(np.float64)
    M2_dense *= rng.integers(1, 30, size=M2_dense.shape).astype(np.float64)
    M1 = sp.csc_matrix(M1_dense)
    M2 = sp.csc_matrix(M2_dense)

    var = pd.DataFrame(
        {
            "chr": pd.Categorical(["chr1"] * n_events),
            "start": np.arange(n_events, dtype=np.int64) * 100,
            "end": np.arange(n_events, dtype=np.int64) * 100 + 50,
            "strand": pd.Categorical(["+"] * n_events),
            "row_names_mtx": [f"chr1:{i*100}-{i*100+50}" for i in range(n_events)],
            "group_id": np.arange(n_events, dtype=np.int32) % 5,
            "group_kind": pd.Categorical(["S"] * n_events, categories=["S", "E"]),
            "group_count": np.full(n_events, 5, dtype=np.int32),
        },
        index=[f"chr1:{i*100}-{i*100+50}_S" for i in range(n_events)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"bc{i}" for i in range(n_cells)],
         "sample_id": ["s1"] * n_cells},
        index=[f"bc{i}-s1" for i in range(n_cells)],
    )
    a = ad.AnnData(layers={"M1": M1, "M2": M2}, obs=obs, var=var)
    a.uns["scsplice"] = {"m2_valid": True}
    return a


def test_pseudo_correlation_smoke():
    import scsplice  # noqa: PLC0415

    a = _make_adata(seed=42)
    rng = np.random.default_rng(0)
    zdb = rng.normal(size=(a.n_vars, a.n_obs))
    scsplice.tl.pseudo_correlation(a, zdb, metric="CoxSnell")
    assert "pseudo_correlation" in a.var.columns
    r = a.var["pseudo_correlation"].to_numpy()
    finite = np.isfinite(r)
    # Most events should converge on this dense synthetic data.
    assert finite.sum() > a.n_vars // 2
    # Signed pseudo-correlation in [-1, 1] when finite.
    assert (r[finite] >= -1.0 - 1e-9).all()
    assert (r[finite] <= 1.0 + 1e-9).all()


def test_pseudo_correlation_metric_dispatch():
    import scsplice  # noqa: PLC0415

    a = _make_adata(n_events=15, seed=7)
    rng = np.random.default_rng(0)
    zdb = rng.normal(size=(a.n_vars, a.n_obs))
    a_cox = scsplice.tl.pseudo_correlation(a, zdb, metric="CoxSnell", inplace=False)
    a_nag = scsplice.tl.pseudo_correlation(a, zdb, metric="Nagelkerke", inplace=False)
    cox = a_cox.var["pseudo_correlation"].to_numpy()
    nag = a_nag.var["pseudo_correlation"].to_numpy()
    finite = np.isfinite(cox) & np.isfinite(nag)
    assert finite.sum() > 0
    # Nagelkerke >= CoxSnell in absolute value (it normalises by max possible).
    assert (np.abs(nag[finite]) + 1e-9 >= np.abs(cox[finite])).all()


def test_pseudo_correlation_unknown_metric():
    import scsplice  # noqa: PLC0415

    a = _make_adata()
    zdb = np.zeros((a.n_vars, a.n_obs))
    with pytest.raises(ValueError, match="metric"):
        scsplice.tl.pseudo_correlation(a, zdb, metric="bogus")


def test_pseudo_correlation_zdb_shape_validation():
    import scsplice  # noqa: PLC0415

    a = _make_adata(n_events=5, n_cells=20)
    bad_zdb = np.zeros((a.n_vars, a.n_obs + 3))
    with pytest.raises(ValueError, match="zdb shape"):
        scsplice.tl.pseudo_correlation(a, bad_zdb)


def test_pseudo_correlation_requires_m2_valid():
    import scsplice  # noqa: PLC0415

    a = _make_adata()
    a.uns["scsplice"]["m2_valid"] = False
    zdb = np.zeros((a.n_vars, a.n_obs))
    with pytest.raises(RuntimeError, match="m2_valid"):
        scsplice.tl.pseudo_correlation(a, zdb)


@pytest.mark.openmp
def test_pseudo_correlation_thread_determinism():
    if not _openmp_enabled():
        pytest.skip("OpenMP not enabled in this build")
    import scsplice  # noqa: PLC0415

    a1 = _make_adata(n_events=40, n_cells=300, seed=11)
    a4 = a1.copy()
    rng = np.random.default_rng(99)
    zdb = rng.normal(size=(a1.n_vars, a1.n_obs))
    scsplice.tl.pseudo_correlation(a1, zdb, n_threads=1)
    scsplice.tl.pseudo_correlation(a4, zdb, n_threads=4)
    r1 = a1.var["pseudo_correlation"].to_numpy()
    r4 = a4.var["pseudo_correlation"].to_numpy()
    nan1 = np.isnan(r1)
    assert np.array_equal(nan1, np.isnan(r4))
    assert np.array_equal(r1[~nan1], r4[~nan1])


def test_pseudo_correlation_null_distribution():
    import scsplice  # noqa: PLC0415

    a = _make_adata(n_events=20, n_cells=150, seed=3)
    rng = np.random.default_rng(0)
    zdb = rng.normal(size=(a.n_vars, a.n_obs))
    scsplice.tl.pseudo_correlation(a, zdb, n_permutations=5, seed=42)
    null = a.varm["pseudo_correlation_null"]
    assert null.shape == (a.n_vars, 5)
    assert null.dtype == np.float64


def test_pseudo_correlation_seeded_null_reproducible():
    import scsplice  # noqa: PLC0415

    a1 = _make_adata(n_events=15, n_cells=100, seed=5)
    a2 = a1.copy()
    rng = np.random.default_rng(0)
    zdb = rng.normal(size=(a1.n_vars, a1.n_obs))
    scsplice.tl.pseudo_correlation(a1, zdb, n_permutations=3, seed=42)
    scsplice.tl.pseudo_correlation(a2, zdb, n_permutations=3, seed=42)
    assert np.array_equal(
        np.nan_to_num(a1.varm["pseudo_correlation_null"], nan=-9.0),
        np.nan_to_num(a2.varm["pseudo_correlation_null"], nan=-9.0),
    )


def test_pseudo_correlation_sign_follows_slope():
    """A perfectly increasing predictor relative to a constant ratio should
    yield a non-negative correlation; a flipped predictor should flip the sign.
    """
    import scsplice  # noqa: PLC0415

    a = _make_adata(n_events=8, n_cells=120, seed=1)
    rng = np.random.default_rng(0)
    zdb_pos = rng.normal(size=(a.n_vars, a.n_obs))

    a_pos = scsplice.tl.pseudo_correlation(a, zdb_pos, inplace=False)
    a_neg = scsplice.tl.pseudo_correlation(a, -zdb_pos, inplace=False)
    p = a_pos.var["pseudo_correlation"].to_numpy()
    n = a_neg.var["pseudo_correlation"].to_numpy()
    finite = np.isfinite(p) & np.isfinite(n)
    # Sign flip must be exact for converged events.
    assert np.allclose(p[finite], -n[finite], atol=1e-12)


def test_pseudo_correlation_inplace_vs_copy():
    import scsplice  # noqa: PLC0415

    a = _make_adata(n_events=10, n_cells=80)
    zdb = np.zeros((a.n_vars, a.n_obs))
    out = scsplice.tl.pseudo_correlation(a, zdb, inplace=True)
    assert out is None
    assert "pseudo_correlation" in a.var.columns

    a2 = _make_adata(n_events=10, n_cells=80)
    out2 = scsplice.tl.pseudo_correlation(a2, zdb, inplace=False)
    assert out2 is not None
    assert "pseudo_correlation" not in a2.var.columns
    assert "pseudo_correlation" in out2.var.columns
