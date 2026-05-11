"""Per-sample splitting of paired (M1, M2) layers.

Mirrors the pattern in ``multigedipy_pkg`` (``_multi_model.py:_split_by_sample``):
the kernel receives one CSC matrix per library so the per-library deviance loop
inside C++ can iterate without any further Python-side bookkeeping.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


__all__ = ["split_paired_by_sample"]


def split_paired_by_sample(
    adata: ad.AnnData,
    sample_key: str = "sample_id",
) -> tuple[list[sp.csc_matrix], list[sp.csc_matrix], list[str]]:
    """Slice ``layers['M1']`` and ``layers['M2']`` along ``obs[sample_key]``.

    Returns
    -------
    M1i, M2i, sample_ids
        Per-sample CSC matrices in AnnData layout (rows = cells of that sample,
        cols = events) and the ordered list of sample ids. Order is the
        ``Categorical`` category order if ``obs[sample_key]`` is categorical;
        otherwise first-seen order in the obs dataframe.

    Notes
    -----
    Output matrices are float64 CSC; transposition to ``events x cells`` for
    Eigen happens at the C++ binding boundary, not here, so this function stays
    pure-Python and trivially testable.
    """
    if sample_key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs[{sample_key!r}] is required for per-sample splitting. "
            "scsplice.io.read_starsolo populates it from its sample_ids= argument; "
            "for AnnDatas built another way, set adata.obs['sample_id'] manually."
        )

    M1 = adata.layers["M1"]
    M2 = adata.layers["M2"]
    sample_vec = adata.obs[sample_key]

    if isinstance(sample_vec.dtype, pd.CategoricalDtype):
        sample_ids = list(sample_vec.cat.categories)
    else:
        sample_ids = list(dict.fromkeys(sample_vec.tolist()))

    sample_arr = np.asarray(sample_vec)
    M1i: list[sp.csc_matrix] = []
    M2i: list[sp.csc_matrix] = []
    for sid in sample_ids:
        idx = np.where(sample_arr == sid)[0]
        if idx.size == 0:
            continue
        M1i.append(sp.csc_matrix(M1[idx, :], dtype=np.float64))
        M2i.append(sp.csc_matrix(M2[idx, :], dtype=np.float64))

    sample_ids = [s for s, m in zip(sample_ids, M1i, strict=False) if m is not None]
    return M1i, M2i, sample_ids
