"""Cross-language equivalence: scsplice.pp.highly_variable_genes(method="vst")
vs. R splikit's find_variable_genes(method = "vst").

Skipped automatically unless tests/data/r_reference_hvg.h5 is present.
Regenerate it with:

    module load r/4.4.0   # or any r/4.x with splikit + rhdf5 installed
    Rscript tests/r_export/export_hvg_reference.R

See tests/r_export/export_hvg_reference.R for what the fixture contains and
tests/load_hvg_r_ref.py for the loader.

Tolerance rationale
--------------------
The row mean/variance and standardization passes are plain floating-point
arithmetic (np.allclose-tight vs. R, not bit-exact — summation order
differs). The loess fit uses skmisc.loess, which wraps the same
netlib/Cleveland-Grosse loess Fortran/C code R's stats::loess() calls, with
identical defaults (degree=2, family="gaussian", surface="interpolate",
cell=0.2) -- so the dominant source of any residual discrepancy is ordinary
cross-platform/compiler floating-point drift, not an algorithmic difference.
rtol=1e-6/atol=1e-8 is deliberately tight; widen it here (and explain why) if
a legitimate new platform combination needs more slack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skmisc", reason="scsplice.pp.highly_variable_genes requires scikit-misc")

sys.path.insert(0, str(Path(__file__).parent))
from load_hvg_r_ref import load_hvg_reference  # noqa: E402


def test_highly_variable_genes_vst_matches_r(hvg_r_reference_path):
    import anndata as ad  # noqa: PLC0415

    import scsplice  # noqa: PLC0415

    ref = load_hvg_reference(hvg_r_reference_path)

    # AnnData is cells x genes; the fixture is genes x cells (R convention).
    X = ref.gene_expression.T.tocsc()
    var = pd.DataFrame(index=pd.Index(ref.gene_names))
    obs = pd.DataFrame(index=[f"bc{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    scsplice.pp.highly_variable_genes(adata, loess_span=ref.loess_span)

    # R's vst branch never filters or reorders rows (asserted at export time),
    # so events line up positionally with gene_names/adata.var_names already;
    # still realign explicitly by name to be robust to that invariant ever
    # changing on the R side.
    assert list(ref.events) == list(ref.gene_names)
    py_values = adata.var.loc[ref.events, "standardize_variance"].to_numpy()
    r_values = ref.standardize_variance

    assert np.allclose(py_values, r_values, rtol=1e-6, atol=1e-8), (
        f"max abs diff={np.max(np.abs(py_values - r_values)):.3e}, "
        f"max rel diff={np.max(np.abs((py_values - r_values) / np.where(r_values != 0, r_values, 1))):.3e}"
    )
