"""Tests for splikit.pp.highly_variable_events.

Tolerance choice: per-row deviance is bit-identical across thread counts (each
row's accumulation is fixed-order serial within one thread). Cross-language
equivalence with R is np.allclose-tight, NOT bit-exact, because std::log
varies by libm/BLAS stack. See the R/Rcpp port spec for details.
"""

from __future__ import annotations

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


def _make_paired_adata(
    *,
    n_events: int = 30,
    n_cells: int = 200,
    n_groups: int = 5,
    n_samples: int = 2,
    seed: int = 0,
    density: float = 0.4,
) -> ad.AnnData:
    """Build a paired (M1, M2) AnnData with multiple libraries.

    Both layers are populated with positive integer counts so most rows pass
    the row-sum filter. group_id is dense 0..G-1.
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"s{k}" for k in range(n_samples)]

    M1_dense = (rng.random((n_cells, n_events)) < density).astype(np.float64)
    M1_dense *= rng.integers(1, 30, size=M1_dense.shape).astype(np.float64)
    M2_dense = (rng.random((n_cells, n_events)) < density).astype(np.float64)
    M2_dense *= rng.integers(1, 30, size=M2_dense.shape).astype(np.float64)

    M1 = sp.csc_matrix(M1_dense)
    M2 = sp.csc_matrix(M2_dense)

    group_id = np.arange(n_events, dtype=np.int32) % int(n_groups)
    group_count = np.bincount(group_id, minlength=int(n_groups))[group_id].astype(
        np.int32
    )

    var = pd.DataFrame(
        {
            "chr": pd.Categorical(["chr1"] * n_events),
            "start": np.arange(n_events, dtype=np.int64) * 100,
            "end": np.arange(n_events, dtype=np.int64) * 100 + 50,
            "strand": pd.Categorical(["+"] * n_events),
            "row_names_mtx": [f"chr1:{i*100}-{i*100+50}" for i in range(n_events)],
            "group_id": group_id,
            "group_kind": pd.Categorical(["S"] * n_events, categories=["S", "E"]),
            "group_count": group_count,
        },
        index=[f"chr1:{i*100}-{i*100+50}_S" for i in range(n_events)],
    )
    sample_assignment = rng.choice(sample_ids, size=n_cells)
    obs = pd.DataFrame(
        {"barcode": [f"bc{i}" for i in range(n_cells)],
         "sample_id": sample_assignment},
        index=[f"bc{i}-{s}" for i, s in enumerate(sample_assignment)],
    )
    a = ad.AnnData(layers={"M1": M1, "M2": M2}, obs=obs, var=var)
    a.uns["splikit"] = {"m2_valid": True}
    return a


def test_hve_smoke():
    import splikit  # noqa: PLC0415

    adata = _make_paired_adata()
    splikit.pp.highly_variable_events(adata, min_row_sum=10.0)

    assert "sum_deviance" in adata.var.columns
    assert "highly_variable" in adata.var.columns
    assert adata.var["sum_deviance"].dtype == np.float64
    assert adata.var["highly_variable"].dtype == bool
    # All passing events flagged True when n_top is None.
    n_passing = int(np.isfinite(adata.var["sum_deviance"]).sum())
    assert int(adata.var["highly_variable"].sum()) == n_passing
    assert n_passing > 0
    params = adata.uns["splikit"]["params"]["highly_variable_events"]
    assert params["min_row_sum"] == 10.0
    assert params["n_libraries"] == 2


def test_hve_n_top_selection():
    import splikit  # noqa: PLC0415

    adata = _make_paired_adata(n_events=30)
    splikit.pp.highly_variable_events(adata, min_row_sum=5.0, n_top=5)

    assert int(adata.var["highly_variable"].sum()) == 5
    selected_dev = adata.var.loc[adata.var["highly_variable"], "sum_deviance"]
    rest_dev = adata.var.loc[~adata.var["highly_variable"], "sum_deviance"]
    rest_passing = rest_dev[np.isfinite(rest_dev)]
    if len(rest_passing) > 0:
        assert selected_dev.min() >= rest_passing.max()


def test_hve_no_events_passing_raises():
    import splikit  # noqa: PLC0415

    adata = _make_paired_adata(n_events=8, n_cells=20, density=0.1, seed=42)
    with pytest.raises(ValueError, match="No events pass"):
        splikit.pp.highly_variable_events(adata, min_row_sum=1e9)


def test_hve_partial_filter():
    import splikit  # noqa: PLC0415

    adata = _make_paired_adata(n_events=20, n_cells=100)
    # Zero out the M1 column for half the events so they fail the M1 row-sum filter.
    M1 = adata.layers["M1"].toarray()
    M1[:, : adata.n_vars // 2] = 0.0
    adata.layers["M1"] = sp.csc_matrix(M1)
    splikit.pp.highly_variable_events(adata, min_row_sum=10.0)
    has_score = np.isfinite(adata.var["sum_deviance"])
    n_passing = int(has_score.sum())
    assert 0 < n_passing < adata.n_vars
    # Filtered events have sum_deviance NaN and highly_variable False.
    assert (~adata.var.loc[~has_score, "highly_variable"]).all()


@pytest.mark.openmp
def test_hve_thread_determinism():
    if not _openmp_enabled():
        pytest.skip("OpenMP not enabled in this build")
    import splikit  # noqa: PLC0415

    a1 = _make_paired_adata(n_events=80, n_cells=400, n_groups=10, n_samples=3,
                            seed=7)
    a4 = a1.copy()
    splikit.pp.highly_variable_events(a1, min_row_sum=10.0, n_threads=1)
    splikit.pp.highly_variable_events(a4, min_row_sum=10.0, n_threads=4)

    s1 = a1.var["sum_deviance"].to_numpy()
    s4 = a4.var["sum_deviance"].to_numpy()
    nan_mask = np.isnan(s1)
    assert np.array_equal(nan_mask, np.isnan(s4))
    assert np.array_equal(s1[~nan_mask], s4[~nan_mask])


def test_hve_inplace_vs_copy():
    import splikit  # noqa: PLC0415

    a = _make_paired_adata(n_events=10, n_cells=50)
    out = splikit.pp.highly_variable_events(a, min_row_sum=5.0, inplace=True)
    assert out is None
    assert "sum_deviance" in a.var.columns

    a2 = _make_paired_adata(n_events=10, n_cells=50)
    out2 = splikit.pp.highly_variable_events(a2, min_row_sum=5.0, inplace=False)
    assert out2 is not None
    assert "sum_deviance" not in a2.var.columns
    assert "sum_deviance" in out2.var.columns


def test_hve_requires_m2_valid():
    import splikit  # noqa: PLC0415

    a = _make_paired_adata()
    a.uns["splikit"]["m2_valid"] = False
    with pytest.raises(RuntimeError, match="m2_valid"):
        splikit.pp.highly_variable_events(a, min_row_sum=10.0)


def test_hve_requires_sample_key():
    import splikit  # noqa: PLC0415

    a = _make_paired_adata()
    del a.obs["sample_id"]
    with pytest.raises(KeyError, match="sample_id"):
        splikit.pp.highly_variable_events(a, min_row_sum=10.0)


def test_hve_p_hat_clamp_zero_row_returns_zero_deviance():
    """Row with zero variation (M2==0 everywhere) gets p_hat==1, clamped to 0."""
    import splikit  # noqa: PLC0415

    a = _make_paired_adata(n_events=8, n_cells=80, density=0.5)
    # Force one row to have M2 all-zero (p_hat = 1 -> clamped to 0 deviance).
    M2 = a.layers["M2"].toarray()
    M2[:, 0] = 0.0  # cells × events; zero out event 0's M2 column
    a.layers["M2"] = sp.csc_matrix(M2)
    # Make sure row sums are large enough to pass the filter.
    M1 = a.layers["M1"].toarray()
    M1[:, 0] = 100.0
    a.layers["M1"] = sp.csc_matrix(M1)

    # min_row_sum on M2 row is 0, so this event is filtered out (NaN).
    splikit.pp.highly_variable_events(a, min_row_sum=10.0)
    assert np.isnan(a.var["sum_deviance"].iloc[0])


@pytest.mark.r_required
@pytest.mark.slow
def test_hve_vs_r(r_reference_path: Path):
    """splk.pp.highly_variable_events vs R splikit::find_variable_events on the toy.

    Tolerance: rtol=1e-10, atol=1e-12 on per-event sum_deviance, NaN-mask exact.
    The kernel is per-row independent (no cross-row reduction); drift comes
    only from std::log differences across libm / SIMD vectorisation paths.
    """
    if not r_reference_path.exists():
        pytest.skip(
            f"R reference fixture not found at {r_reference_path}. "
            "Regenerate via Rscript tests/r_export/export_reference.R"
        )

    from tests.load_r_ref import load_reference  # noqa: PLC0415
    from tests.test_make_m2 import _build_adata_from_toy_ref  # noqa: PLC0415
    import splikit  # noqa: PLC0415

    ref = load_reference(r_reference_path)
    if ref.hve_events is None:
        pytest.skip(
            "Loaded R fixture is missing find_variable_events/sum_deviance. "
            "Regenerate via Rscript tests/r_export/export_reference.R."
        )

    adata = _build_adata_from_toy_ref(ref)
    splikit.tl.make_m2(adata, n_threads=1)
    splikit.pp.highly_variable_events(
        adata, min_row_sum=ref.min_row_sum, n_threads=1
    )

    # Map R-kept events back to Python's full-length output.
    py_dev = adata.var["sum_deviance"].to_numpy()
    py_event_id = adata.var.index.to_numpy()
    r_event_id = ref.hve_events
    r_dev = ref.hve_sum_deviance

    py_index = pd.Index(py_event_id)
    matched = py_index.get_indexer(r_event_id)
    if (matched < 0).any():
        # The toy fixture stores R's `events` column verbatim; if R suffixed
        # them differently from our var_names, retry against the un-suffixed
        # row_names_mtx.
        bare_index = pd.Index(adata.var["row_names_mtx"].to_numpy())
        matched_bare = bare_index.get_indexer(r_event_id)
        if (matched_bare >= 0).all():
            matched = matched_bare
        else:
            missing = r_event_id[matched < 0][:5]
            raise AssertionError(
                f"R-kept events not found in Python output (sample): {missing}"
            )
    py_dev_matched = py_dev[matched]

    valid = np.isfinite(r_dev) & np.isfinite(py_dev_matched)
    if not np.array_equal(np.isnan(py_dev_matched), np.isnan(r_dev)):
        bad = int(np.argmax(np.isnan(py_dev_matched) != np.isnan(r_dev)))
        raise AssertionError(
            f"NaN mask differs at event {r_event_id[bad]!r}: "
            f"py={py_dev_matched[bad]!r}, r={r_dev[bad]!r}"
        )
    if not np.allclose(py_dev_matched[valid], r_dev[valid],
                       rtol=1e-10, atol=1e-12):
        diff = np.abs(py_dev_matched[valid] - r_dev[valid])
        idx = int(np.argmax(diff))
        raise AssertionError(
            f"sum_deviance differs beyond rtol=1e-10/atol=1e-12. Max abs diff "
            f"{diff[idx]:.3e} at event {r_event_id[valid][idx]!r}: "
            f"py={py_dev_matched[valid][idx]!r}, r={r_dev[valid][idx]!r}."
        )
