"""End-to-end integration test: splikit-py AnnData -> gedi2py.

Loads the h5ad produced by ``run_py_pipeline.py``, applies an HVE pre-filter
to keep peak densification memory under ~8 GB, then runs ``gd.tl.gedi`` in
M_paired mode with ``layer='M1', layer2='M2', batch_key='sample_id'``.

The point of the test is to confirm that splikit-py's AnnData layout is
**directly consumable** by gedi2py with no adapter, on real data. Asserts:

  * gedi2py runs without exception on the splikit-py output.
  * obsm['X_gedi'] / varm['gedi_Z'] / uns['gedi'] all populated with the
    expected shapes and finite values.
  * params written to uns['gedi'] reflect the call we made.
  * Embedding rank is non-degenerate (>= 8 of 10 latent dims).

Memory math (per scverse-python-architect review on real adata,
14570 cells x 281735 events):
  Per-sample dense densify in gedi2py = 2 layers x 8 bytes x cells x events.
  Yi list retained for both samples adds (5604+8966) * N * 8 B per axis.
  Peak ~ 260K * N bytes. n_top=20000 -> ~5.2 GB densification peak.

Run on c170:
  cd /home/arsham79/projects/rrg-hsn/arsham79/splikitpy
  module load eigen/3.4.0
  /home/arsham79/projects/rrg-hsn/arsham79/multigedipy_pkg/.venv/bin/pytest \\
      validation/e2e/test_gedi2py_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest


pytestmark = [pytest.mark.slow]


HERE = Path(__file__).resolve().parent
H5AD = HERE / "data" / "py_pipeline.h5ad"
N_TOP = 20_000
N_LATENT = 10
MAX_ITER = 20


@pytest.fixture(scope="module")
def adata_full() -> ad.AnnData:
    if not H5AD.exists():
        pytest.skip(
            f"{H5AD} not found. Run validation/e2e/run_py_pipeline.py first "
            "(or bash validation/e2e/run_all.sh)."
        )
    a = ad.read_h5ad(H5AD)
    return a


@pytest.fixture(scope="module")
def adata_filtered(adata_full):
    """HVE-filtered subset; keeps gedi2py densification peak ~5 GB.

    ``min_row_sum=10`` is set deliberately low so this real-dataset slice
    has a large enough HVE pool that ``n_top=N_TOP`` is achievable. The
    default of 50 leaves only ~5,000 events on the ``A01 + B01`` slice.
    """
    import splikit  # noqa: PLC0415

    a = adata_full.copy()
    splikit.pp.highly_variable_events(
        a, min_row_sum=10, n_top=N_TOP, sample_key="sample_id",
        n_threads=4, inplace=True,
    )
    a = a[:, a.var["highly_variable"]].copy()
    n_kept = int(a.n_vars)
    if n_kept < 1000:
        pytest.skip(
            f"HVE kept only {n_kept} events; integration test needs >=1000 "
            "for a non-trivial run. Lower min_row_sum further or expand "
            "the slice."
        )
    # M2 is invalidated by the var-axis subset; recompute.
    splikit.tl.make_m2(a, n_threads=4)
    return a


def test_gedi2py_module_importable():
    pytest.importorskip("gedi2py")


def test_input_layout_matches_gedi2py_contract(adata_full):
    """Sanity-check the splikit-py output against gedi2py's runtime expectations
    BEFORE we hit gedi2py itself, so a layout regression on the splikit-py side
    fails here with a clear message instead of deep inside gedi2py's C++ kernel."""
    import scipy.sparse as sp  # noqa: PLC0415

    M1 = adata_full.layers["M1"]
    M2 = adata_full.layers["M2"]
    assert sp.issparse(M1), "layers['M1'] must be sparse"
    assert sp.issparse(M2), "layers['M2'] must be sparse"
    assert M1.shape == M2.shape == adata_full.shape, (
        f"layers shapes {M1.shape}, {M2.shape} must match adata.shape "
        f"{adata_full.shape}"
    )
    assert M1.dtype == np.float64 == M2.dtype
    assert "sample_id" in adata_full.obs.columns
    assert adata_full.obs["sample_id"].astype(str).nunique() >= 2, (
        "gedi2py needs at least two batches in batch_key for M_paired mode"
    )
    assert adata_full.uns["splikit"]["m2_valid"] is True


def test_gedi2py_runs_on_splikit_output(adata_filtered):
    """The integration test that justifies this whole branch."""
    pytest.importorskip("gedi2py")
    import gedi2py as gd  # noqa: PLC0415

    gd.tl.gedi(
        adata_filtered,
        batch_key="sample_id",
        layer="M1",
        layer2="M2",
        n_latent=N_LATENT,
        max_iterations=MAX_ITER,
        mode="Bsphere",
        n_jobs=-1,
        random_state=0,
        verbose=False,
    )

    # Embedding presence + shape.
    assert "X_gedi" in adata_filtered.obsm, "obsm['X_gedi'] missing after gedi"
    Xg = adata_filtered.obsm["X_gedi"]
    assert Xg.shape == (adata_filtered.n_obs, N_LATENT), (
        f"obsm['X_gedi'] shape {Xg.shape} != "
        f"({adata_filtered.n_obs}, {N_LATENT})"
    )
    assert np.isfinite(Xg).all(), "obsm['X_gedi'] has non-finite entries"

    # Gene loadings.
    assert "gedi_Z" in adata_filtered.varm, "varm['gedi_Z'] missing after gedi"
    Z = adata_filtered.varm["gedi_Z"]
    assert Z.shape == (adata_filtered.n_vars, N_LATENT), (
        f"varm['gedi_Z'] shape {Z.shape} != ({adata_filtered.n_vars}, {N_LATENT})"
    )
    assert np.isfinite(Z).all(), "varm['gedi_Z'] has non-finite entries"

    # Run metadata.
    assert "gedi" in adata_filtered.uns
    g = adata_filtered.uns["gedi"]
    assert "params" in g and "model" in g
    assert g["params"]["batch_key"] == "sample_id"
    assert g["params"]["layer"] == "M1"
    assert g["params"]["layer2"] == "M2"

    # Embedding non-degeneracy: with n_latent=10 we expect rank close to 10.
    # Allow a small tolerance for stochastic rank loss; require >= 8 / 10.
    rank = int(np.linalg.matrix_rank(Xg))
    assert rank >= 8, (
        f"obsm['X_gedi'] has rank {rank} (< 8); the embedding is degenerate"
    )


def test_documented_schema_divergences_persist(adata_full):
    """Surface (not enforce) the two schema divergences from R splikit so
    a future regression is visible in the test report:

    - var['row_names_mtx'] is un-suffixed.
    - var['group_count'] is post-filter (matches the matrix on disk).
    """
    rn = adata_full.var["row_names_mtx"].astype(str).to_numpy()
    has_suffix = np.array([n.endswith(("_S", "_E")) for n in rn])
    assert not has_suffix.any(), (
        "row_names_mtx leaked _S / _E suffix; the suffix should live in "
        "var_names + var['group_kind'] only"
    )
    # group_count is consistent with the partition implied by group_id.
    import pandas as pd  # noqa: PLC0415

    gid = adata_full.var["group_id"].to_numpy()
    sizes = pd.Series(gid).groupby(gid).transform("size").to_numpy()
    np.testing.assert_array_equal(
        adata_full.var["group_count"].astype(np.int32).to_numpy(),
        sizes.astype(np.int32),
        err_msg="group_count is not the post-filter LJV size from group_id",
    )
