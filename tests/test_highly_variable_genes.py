"""Tests for scsplice.pp.highly_variable_genes (Seurat/R-splikit VST method).

Both C++ passes (row mean/variance, standardization) are deterministic across
thread counts (disjoint per-row writes, no cross-row reductions); the loess
fit in between runs single-threaded in Python via skmisc.loess.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("skmisc", reason="scsplice.pp.highly_variable_genes requires scikit-misc")


def _openmp_enabled() -> bool:
    try:
        from scsplice import _scsplice_cpp  # noqa: PLC0415
        return bool(_scsplice_cpp.__openmp__)
    except ImportError:
        return False


def _make_gene_adata(
    *,
    n_genes: int = 120,
    n_cells: int = 250,
    seed: int = 0,
) -> ad.AnnData:
    """Build a synthetic cell x gene AnnData with Poisson-lognormal counts."""
    rng = np.random.default_rng(seed)
    mu = rng.lognormal(mean=1.0, sigma=1.2, size=n_genes)
    X_dense = rng.poisson(mu[None, :], size=(n_cells, n_genes)).astype(np.float64)
    X = sp.csc_matrix(X_dense)

    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"bc{i}" for i in range(n_cells)]},
        index=[f"bc{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def test_hvg_smoke():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata()
    scsplice.pp.highly_variable_genes(adata)

    assert "standardize_variance" in adata.var.columns
    assert "highly_variable" in adata.var.columns
    assert adata.var["standardize_variance"].dtype == np.float64
    assert adata.var["highly_variable"].dtype == bool
    assert np.isfinite(adata.var["standardize_variance"]).all()
    assert (adata.var["standardize_variance"] >= 0).all()
    assert adata.var["highly_variable"].sum() > 0

    params = adata.uns["scsplice"]["params"]["highly_variable_genes"]
    assert params["method"] == "vst"
    assert params["n_genes_input"] == adata.n_vars


def test_hvg_row_mean_var_matches_numpy():
    """The C++ first pass must match a plain numpy mean/variance computation."""
    from scsplice import _scsplice_cpp as cpp  # noqa: PLC0415

    rng = np.random.default_rng(1)
    X_dense = rng.poisson(3.0, size=(80, 40)).astype(np.float64)
    X_dense[:, 5] = 0.0
    X = sp.csc_matrix(X_dense)
    X_T = X.T.tocsc()  # genes x cells

    mean, var = cpp.hvg_row_mean_var(X_T, 1)
    expected_mean = X_dense.T.mean(axis=1)
    expected_var = (X_dense.T ** 2).mean(axis=1) - expected_mean ** 2

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)


def test_hvg_standardize_variance_hand_computed():
    """Second pass matches a manual clamp-and-accumulate for a tiny example."""
    from scsplice import _scsplice_cpp as cpp  # noqa: PLC0415

    X_dense = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 3.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    X = sp.csc_matrix(X_dense)
    mean = X_dense.mean(axis=1)
    sd = np.array([2.0, 1.0])

    result = np.asarray(cpp.hvg_standardize_variance(X, mean, sd, 1))

    ncol = X_dense.shape[1]
    vmax = np.sqrt(ncol)
    manual = np.empty(2)
    for i in range(2):
        z = np.clip((X_dense[i] - mean[i]) / sd[i], -vmax, vmax)
        manual[i] = np.sum(z ** 2) / (ncol - 1)

    assert np.allclose(result, manual)


def test_hvg_all_zero_and_constant_genes_get_zero_not_nan():
    """R splikit computes a literal (non-NaN) value for degenerate genes."""
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=40, n_cells=100, seed=2)
    X = adata.X.toarray()
    X[:, 0] = 0.0  # all-zero
    X[:, 1] = 4.0  # constant nonzero
    adata.X = sp.csc_matrix(X)

    scsplice.pp.highly_variable_genes(adata)

    assert adata.var["standardize_variance"].iloc[0] == 0.0
    assert adata.var["standardize_variance"].iloc[1] == 0.0
    assert not np.isnan(adata.var["standardize_variance"].iloc[0])
    assert not np.isnan(adata.var["standardize_variance"].iloc[1])
    # Degenerate genes are excluded from selection when n_top is None.
    assert not adata.var["highly_variable"].iloc[0]
    assert not adata.var["highly_variable"].iloc[1]


def test_hvg_n_top_selection():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=60, seed=3)
    scsplice.pp.highly_variable_genes(adata, n_top=8)

    assert int(adata.var["highly_variable"].sum()) == 8
    selected = adata.var.loc[adata.var["highly_variable"], "standardize_variance"]
    rest = adata.var.loc[~adata.var["highly_variable"], "standardize_variance"]
    assert selected.min() >= rest.max()


@pytest.mark.openmp
def test_hvg_thread_determinism():
    if not _openmp_enabled():
        pytest.skip("OpenMP not enabled in this build")
    import scsplice  # noqa: PLC0415

    a1 = _make_gene_adata(n_genes=100, n_cells=300, seed=7)
    a4 = a1.copy()
    scsplice.pp.highly_variable_genes(a1, n_threads=1)
    scsplice.pp.highly_variable_genes(a4, n_threads=4)

    assert np.allclose(
        a1.var["standardize_variance"].to_numpy(),
        a4.var["standardize_variance"].to_numpy(),
    )
    assert np.array_equal(
        a1.var["highly_variable"].to_numpy(), a4.var["highly_variable"].to_numpy()
    )


def test_hvg_layer_argument():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=30, n_cells=80, seed=4)
    adata.layers["counts"] = adata.X.copy()
    adata.X = None

    scsplice.pp.highly_variable_genes(adata, layer="counts")
    assert "standardize_variance" in adata.var.columns


def test_hvg_inplace_vs_copy():
    import scsplice  # noqa: PLC0415

    a = _make_gene_adata(n_genes=20, n_cells=60, seed=5)
    out = scsplice.pp.highly_variable_genes(a, inplace=True)
    assert out is None
    assert "standardize_variance" in a.var.columns

    a2 = _make_gene_adata(n_genes=20, n_cells=60, seed=5)
    out2 = scsplice.pp.highly_variable_genes(a2, inplace=False)
    assert out2 is not None
    assert "standardize_variance" not in a2.var.columns
    assert "standardize_variance" in out2.var.columns


def test_hvg_requires_sparse_input():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=10, n_cells=20, seed=6)
    adata.X = adata.X.toarray()
    with pytest.raises(TypeError, match="scipy.sparse"):
        scsplice.pp.highly_variable_genes(adata)


def test_hvg_no_x_or_layer_raises():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=10, n_cells=20, seed=6)
    adata.X = None
    with pytest.raises(ValueError, match="adata.X is None"):
        scsplice.pp.highly_variable_genes(adata)


def test_hvg_all_degenerate_genes_raises():
    import scsplice  # noqa: PLC0415

    n_genes, n_cells = 5, 20
    X = sp.csc_matrix(np.zeros((n_cells, n_genes)))
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    obs = pd.DataFrame(index=[f"bc{i}" for i in range(n_cells)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    with pytest.raises(ValueError, match="No genes have both"):
        scsplice.pp.highly_variable_genes(adata)


def test_hvg_invalid_method_raises():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=10, n_cells=20, seed=8)
    with pytest.raises(ValueError, match="method must be 'vst'"):
        scsplice.pp.highly_variable_genes(adata, method="sum_deviance")


def test_hvg_invalid_n_top_raises():
    import scsplice  # noqa: PLC0415

    adata = _make_gene_adata(n_genes=10, n_cells=20, seed=9)
    with pytest.raises(ValueError, match="n_top must be positive"):
        scsplice.pp.highly_variable_genes(adata, n_top=0)
