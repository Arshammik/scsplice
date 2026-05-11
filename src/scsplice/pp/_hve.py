"""Highly variable splicing-event selection via per-library binomial deviance."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scsplice._core._validators import (
    validate_paired_layers,
    validate_var_schema,
)


__all__ = ["highly_variable_events"]


def _import_extension():
    try:
        from scsplice import _scsplice_cpp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scsplice's C++ extension (_scsplice_cpp) is not built. "
            "Run `pip install -e .` (or install a wheel) in an environment "
            "with Eigen3 available."
        ) from exc
    return _scsplice_cpp


def highly_variable_events(
    adata: ad.AnnData,
    *,
    min_row_sum: float = 50.0,
    n_top: int | None = None,
    n_threads: int = 1,
    sample_key: str = "sample_id",
    key_added: str = "highly_variable",
    inplace: bool = True,
) -> ad.AnnData | None:
    """Identify highly variable splicing events via ratio binomial deviance.

    For every event, compute the per-library deviance of the (M1, M2) ratio
    against its library-aggregate p_hat, then sum across libraries. Events
    that fail the row-sum filter (``M1.sum > min_row_sum AND M2.sum > min_row_sum``)
    receive ``NaN`` deviance and are excluded from the top-N selection.

    Parameters
    ----------
    adata
        Splicing AnnData with ``layers['M1']``, ``layers['M2']``, and
        ``obs[sample_key]``. M2 must be valid (``uns['splikit']['m2_valid']``).
    min_row_sum
        Minimum row sum required on **both** M1 and M2 (computed on the
        full data, before per-library splitting). Events failing the filter
        are excluded from deviance computation entirely.
    n_top
        If set, mark the top-N events by ``sum_deviance`` as
        ``var[key_added] = True``. Otherwise mark all passing events.
    n_threads
        OpenMP thread count. Per-row work is independent so output is
        bit-identical regardless of n_threads.
    sample_key
        ``adata.obs`` column identifying libraries; replaces R splikit's
        regex on barcode strings.
    key_added
        Boolean column in ``var`` flagging selected events.
    inplace
        Mutate ``adata`` in place and return ``None`` (default), or return
        a copy with the same modifications.

    Returns
    -------
    ``None`` when ``inplace=True``; otherwise a copy.

    Notes
    -----
    Writes ``var['sum_deviance']`` (float64, ``NaN`` for filtered-out events)
    and ``var[key_added]`` (bool). Stores call params under
    ``uns['splikit']['params']['highly_variable_events']``.
    """
    cpp = _import_extension()

    if not inplace:
        adata = adata.copy()

    validate_paired_layers(adata, require_m2_valid=True)
    validate_var_schema(adata)
    if sample_key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs[{sample_key!r}] is required for per-library deviance "
            "(scsplice replaces R splikit's barcode-regex split with an "
            "explicit sample_id column)."
        )

    M1 = adata.layers["M1"]
    M2 = adata.layers["M2"]
    if not isinstance(M1, sp.csc_matrix):
        M1 = sp.csc_matrix(M1)
    if not isinstance(M2, sp.csc_matrix):
        M2 = sp.csc_matrix(M2)
    if M1.dtype != np.float64:
        M1 = M1.astype(np.float64)
    if M2.dtype != np.float64:
        M2 = M2.astype(np.float64)

    # AnnData layout is cells × events; the R kernel works in events × cells.
    # Filter on full-data row sums BEFORE splitting (matches R, see spec).
    M1_T = M1.T.tocsc()
    M2_T = M2.T.tocsc()

    n_events = M1_T.shape[0]
    m1_row_sums = np.asarray(M1_T.sum(axis=1)).ravel()
    m2_row_sums = np.asarray(M2_T.sum(axis=1)).ravel()
    keep_mask = (m1_row_sums > float(min_row_sum)) & (m2_row_sums > float(min_row_sum))
    if not keep_mask.any():
        raise ValueError(
            f"No events pass min_row_sum={min_row_sum}. Lower the threshold or "
            f"check the data. Current row-sum range: M1=[{m1_row_sums.min():.1f},"
            f"{m1_row_sums.max():.1f}], M2=[{m2_row_sums.min():.1f},"
            f"{m2_row_sums.max():.1f}]."
        )

    keep_idx = np.where(keep_mask)[0]
    M1_filt = M1_T[keep_idx, :].tocsc()
    M2_filt = M2_T[keep_idx, :].tocsc()
    n_kept = M1_filt.shape[0]

    # Per-library splitting (replaces R's barcode regex).
    sample_vec = np.asarray(adata.obs[sample_key])
    if hasattr(adata.obs[sample_key].dtype, "categories"):
        sample_ids = list(adata.obs[sample_key].cat.categories)
    else:
        sample_ids = list(dict.fromkeys(sample_vec.tolist()))

    summed = np.zeros(n_kept, dtype=np.float64)
    for sid in sample_ids:
        mask = sample_vec == sid
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        M1_lib = sp.csc_matrix(M1_filt[:, idx], dtype=np.float64)
        M2_lib = sp.csc_matrix(M2_filt[:, idx], dtype=np.float64)
        dev_lib = cpp.calc_deviances_ratio(M1_lib, M2_lib, int(n_threads))
        summed += np.asarray(dev_lib, dtype=np.float64).ravel()

    sum_deviance = np.full(n_events, np.nan, dtype=np.float64)
    sum_deviance[keep_idx] = summed

    selected = np.zeros(n_events, dtype=bool)
    valid_idx = keep_idx
    if n_top is None:
        selected[valid_idx] = True
    else:
        if n_top <= 0:
            raise ValueError(f"n_top must be positive, got {n_top}")
        valid_dev = sum_deviance[valid_idx]
        order = np.argsort(-valid_dev, kind="stable")
        top_local = order[: int(n_top)]
        selected[valid_idx[top_local]] = True

    adata.var["sum_deviance"] = sum_deviance
    adata.var[key_added] = selected

    ns = adata.uns.setdefault("splikit", {})
    ns.setdefault("params", {})["highly_variable_events"] = {
        "min_row_sum": float(min_row_sum),
        "n_top": None if n_top is None else int(n_top),
        "n_threads": int(n_threads),
        "sample_key": str(sample_key),
        "key_added": str(key_added),
        "n_passing": int(keep_mask.sum()),
        "n_libraries": int(len(sample_ids)),
    }

    return adata if not inplace else None
