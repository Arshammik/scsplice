"""Cross-language equivalence tests against R splikit on real data.

The fixture is built on c170 from a saved MultiGEDI model:

    ssh c170 'cd <repo> && module load r/4.4.0 && \
        Rscript tests/r_export/export_merfish_reference.R \
        --out tests/data/r_reference_merfish.h5'

Single sample (smallest one in the model, default ``SRR26528546``), 12,774
events × 835 cells. The fixture is too large to commit (~80 MB ZDB alone)
and never lands in git; tests skip cleanly when it's missing.

Tolerance bands (all proposed; refine on first run):

* ``make_m2``      — bit-exact (``np.array_equal`` on ``indptr/indices/data``).
* ``hve``          — ``np.allclose(rtol=1e-10, atol=1e-12)`` on per-event
  ``sum_deviance``; NaN mask must match exactly.
* ``pseudo_corr``  — ``np.allclose(rtol=1e-7, atol=1e-9)`` on signed pseudo-r;
  sign comparison floored when ``|β_slope|`` is near machine zero (the
  fixture doesn't expose β so we just allow rtol-band sign flips); NaN mask
  must match exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_PATH = REPO_ROOT / "tests" / "data" / "r_reference_merfish.h5"


pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        f"MERFISH R reference fixture not found at {FIXTURE_PATH}. "
        "Regenerate on c170 via "
        "Rscript tests/r_export/export_merfish_reference.R."
    ),
)


def _build_adata_from_ref(ref) -> ad.AnnData:
    """Build a splicing AnnData matching the schema validators expect."""
    # R reference is events × cells; AnnData is cells × events.
    M1_T = ref.m1.T.tocsc().astype(np.float64)
    n_cells, n_events = M1_T.shape

    # row_names_mtx (un-suffixed) — strip the trailing _S/_E for the var column.
    bare_names = np.array(
        [n[:-2] if (n.endswith("_S") or n.endswith("_E")) else n
         for n in ref.row_names_mtx]
    )

    var = pd.DataFrame(
        {
            "chr": pd.Categorical(ref.chr_arr.astype(str)),
            "start": ref.start.astype(np.int64),
            "end": ref.end.astype(np.int64),
            "strand": pd.Categorical(
                ref.strand.astype(str), categories=[".", "+", "-", "*"]
            ),
            "row_names_mtx": bare_names,
            "group_id": ref.group_id_dense.astype(np.int32),
            "group_kind": pd.Categorical(
                ref.group_kind.astype(str), categories=["S", "E"]
            ),
            "group_count": ref.group_count.astype(np.int32),
        },
        index=pd.Index(ref.row_names_mtx),
    )

    # cells: barcodes are "<bc>-<sample_id>" already; sample_id is uniform.
    obs = pd.DataFrame(
        {
            "barcode": np.array([n.rsplit("-", 1)[0] for n in ref.zdb_colnames]),
            "sample_id": np.array([ref.sample_id] * n_cells, dtype=object),
        },
        index=pd.Index(ref.zdb_colnames),
    )
    obs["sample_id"] = obs["sample_id"].astype("category")

    a = ad.AnnData(layers={"M1": M1_T}, obs=obs, var=var)
    a.uns["splikit"] = {
        "version": 1,
        "m2_valid": False,
        "ljv_kind": "start_end",
        "source": "merfish_fixture",
        "params": {},
    }
    return a


@pytest.fixture(scope="module")
def merfish_ref():
    from tests.load_r_ref import load_merfish_reference  # noqa: PLC0415
    return load_merfish_reference(FIXTURE_PATH)


@pytest.fixture(scope="module")
def staleness_warning(merfish_ref):
    """Emit a warning if the on-c170 splikit version differs from the
    splikit-py kernel's algorithm-spec target. Non-fatal."""
    import warnings  # noqa: PLC0415

    target = "2.2.1"  # the version present at fixture-generation time
    if merfish_ref.splikit_version and merfish_ref.splikit_version != target:
        warnings.warn(
            f"Fixture was generated against R splikit {merfish_ref.splikit_version}; "
            f"splikit-py kernels were written against {target}. Tolerance bands "
            "may need adjustment.",
            stacklevel=2,
        )
    return merfish_ref


@pytest.mark.r_required
@pytest.mark.slow
def test_make_m2_vs_r_real(merfish_ref):
    """splk.tl.make_m2 must reproduce R splikit::make_m2 bit-for-bit on
    the subset eventdata. The kernel is purely additive over a deterministic
    CSC traversal; any drift is a real bug."""
    import splikit  # noqa: PLC0415

    a = _build_adata_from_ref(merfish_ref)
    splikit.tl.make_m2(a, n_threads=1)

    M2_py = a.layers["M2"].T.tocsc()
    M2_r = merfish_ref.m2

    assert M2_py.shape == M2_r.shape, f"shape {M2_py.shape} != {M2_r.shape}"
    assert np.array_equal(M2_py.indptr, M2_r.indptr.astype(M2_py.indptr.dtype)), (
        f"indptr arrays differ; first divergence at "
        f"{int(np.argmax(M2_py.indptr.astype(np.int64) != M2_r.indptr.astype(np.int64)))}"
    )
    assert np.array_equal(
        M2_py.indices, M2_r.indices.astype(M2_py.indices.dtype)
    ), "indices arrays differ"
    if not np.array_equal(M2_py.data, M2_r.data):
        diff = np.abs(M2_py.data - M2_r.data)
        idx = int(np.argmax(diff))
        raise AssertionError(
            f"M2.data differs from R reference. Max abs diff {diff[idx]:.3e} "
            f"at flat index {idx} (py={M2_py.data[idx]!r}, r={M2_r.data[idx]!r})."
        )


@pytest.mark.r_required
@pytest.mark.slow
def test_hve_vs_r_real(merfish_ref):
    """splk.pp.highly_variable_events must reproduce R splikit::find_variable_events
    on the same subset. Tolerance is rtol=1e-10, atol=1e-12 (per-row sums of
    log terms; libm/SIMD log differences). NaN mask must match."""
    import splikit  # noqa: PLC0415

    a = _build_adata_from_ref(merfish_ref)
    splikit.tl.make_m2(a, n_threads=1)
    splikit.pp.highly_variable_events(
        a, min_row_sum=merfish_ref.min_row_sum, n_threads=1
    )

    py_dev = a.var["sum_deviance"].to_numpy()
    py_event_id = a.var.index.to_numpy()  # var_names are suffixed

    r_event_id = merfish_ref.hve_events
    r_dev = merfish_ref.hve_sum_deviance

    # R drops filtered events. For every R-event, the Python value must match.
    py_index = pd.Index(py_event_id)
    matched = py_index.get_indexer(r_event_id)
    if (matched < 0).any():
        missing = r_event_id[matched < 0][:5]
        raise AssertionError(
            f"R-kept events not found in Python output: {missing.tolist()}..."
        )
    py_dev_matched = py_dev[matched]

    # Both finite where R is finite.
    r_finite = np.isfinite(r_dev)
    if not np.array_equal(np.isnan(py_dev_matched), np.isnan(r_dev) | ~r_finite):
        bad_idx = int(np.argmax(np.isnan(py_dev_matched) != np.isnan(r_dev)))
        raise AssertionError(
            f"NaN mask differs at event {r_event_id[bad_idx]!r}: "
            f"py={py_dev_matched[bad_idx]!r}, r={r_dev[bad_idx]!r}"
        )

    valid = r_finite & np.isfinite(py_dev_matched)
    if not np.allclose(py_dev_matched[valid], r_dev[valid], rtol=1e-10, atol=1e-12):
        diff = np.abs(py_dev_matched[valid] - r_dev[valid])
        idx = int(np.argmax(diff))
        raise AssertionError(
            f"sum_deviance differs beyond rtol=1e-10/atol=1e-12. Max abs diff "
            f"{diff[idx]:.3e} at event {r_event_id[valid][idx]!r}: "
            f"py={py_dev_matched[valid][idx]!r}, r={r_dev[valid][idx]!r}."
        )

    # Events R filtered out (not in r_event_id) must be NaN in Python.
    py_filtered_mask = ~np.isin(py_event_id, r_event_id)
    if py_filtered_mask.any():
        py_filtered = py_dev[py_filtered_mask]
        if not np.all(np.isnan(py_filtered)):
            n_bad = int(np.sum(~np.isnan(py_filtered)))
            raise AssertionError(
                f"{n_bad} events that R filtered out have non-NaN sum_deviance in Python"
            )


@pytest.mark.r_required
@pytest.mark.slow
@pytest.mark.parametrize("metric", ["CoxSnell", "Nagelkerke"])
def test_pseudo_correlation_vs_r_real(merfish_ref, metric: str):
    """splk.tl.pseudo_correlation must reproduce R splikit::get_pseudo_correlation
    within IRLS-path tolerance (rtol=1e-7, atol=1e-9). NaN mask must match."""
    import splikit  # noqa: PLC0415

    a = _build_adata_from_ref(merfish_ref)
    splikit.tl.make_m2(a, n_threads=1)

    # ZDB from the R fixture is events × cells; splikit-py expects (n_var, n_obs)
    # which is also events × cells in AnnData var/obs orientation.
    zdb = merfish_ref.zdb.astype(np.float64)
    assert zdb.shape == (a.n_vars, a.n_obs), (
        f"ZDB shape {zdb.shape} != (n_var={a.n_vars}, n_obs={a.n_obs})"
    )

    splikit.tl.pseudo_correlation(a, zdb, metric=metric, n_threads=1)

    py = a.var["pseudo_correlation"].to_numpy()
    r = (merfish_ref.pcor_coxsnell if metric == "CoxSnell"
         else merfish_ref.pcor_nagelkerke)

    if not np.array_equal(np.isnan(py), np.isnan(r)):
        py_nan = int(np.isnan(py).sum())
        r_nan = int(np.isnan(r).sum())
        diff_idx = np.where(np.isnan(py) != np.isnan(r))[0]
        sample = diff_idx[:5].tolist()
        raise AssertionError(
            f"NaN masks differ ({py_nan} vs {r_nan}); first divergent indices "
            f"{sample}: py={py[sample]!r}, r={r[sample]!r}"
        )

    valid = np.isfinite(py) & np.isfinite(r)
    if not valid.any():
        pytest.skip("No event has a valid pseudo-correlation in either impl.")

    if not np.allclose(py[valid], r[valid], rtol=1e-7, atol=1e-9):
        diff = np.abs(py[valid] - r[valid])
        idx = int(np.argmax(diff))
        # Locate the absolute index in the full event vector for diagnostic.
        valid_idx = np.where(valid)[0]
        abs_idx = int(valid_idx[idx])
        raise AssertionError(
            f"pseudo_correlation ({metric}) differs beyond rtol=1e-7/atol=1e-9. "
            f"Max abs diff {diff[idx]:.3e} at event index {abs_idx}: "
            f"py={py[abs_idx]!r}, r={r[abs_idx]!r}"
        )
