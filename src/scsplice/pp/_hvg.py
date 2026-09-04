"""Highly variable gene selection via Seurat/R-splikit variance-stabilizing transformation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scsplice._core._validators import setdefault_scsplice_ns

__all__ = ["highly_variable_genes"]


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


def _import_loess():
    try:
        from skmisc.loess import loess  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scsplice.pp.highly_variable_genes(method='vst') requires "
            "scikit-misc. Install it with `pip install scikit-misc` (or "
            "`pip install \"scsplice[hvg]\"`). scikit-misc wraps the same "
            "netlib/Cleveland-Grosse loess Fortran code that R's "
            "stats::loess() calls, which is what makes this function's "
            "output numerically comparable to R splikit's "
            "find_variable_genes(method='vst')."
        ) from exc
    return loess


def highly_variable_genes(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    method: str = "vst",
    loess_span: float = 0.3,
    n_top: int | None = None,
    n_threads: int = 1,
    key_added: str = "highly_variable",
    inplace: bool = True,
) -> ad.AnnData | None:
    """Identify highly variable genes via Seurat-style variance-stabilizing transformation.

    Fits a mean-variance trend (``log10(variance) ~ log10(mean)``, degree-2
    local regression with tricube weights, ``span=loess_span``) across genes,
    then computes each gene's clipped, standardized residual variance —
    Hafemeister & Satija (2019)'s VST, as implemented by R splikit's
    ``find_variable_genes(method = "vst")`` /
    ``standardizeSparse_variance_vst()``.

    Parameters
    ----------
    adata
        Gene-expression AnnData, e.g. from :func:`scsplice.io.read_starsolo_gene`
        or :func:`scsplice.io.read_starsolo_velocyto`. Genes are columns
        (``var``), cells are rows (``obs``), matching AnnData convention.
    layer
        Read counts from ``adata.layers[layer]`` instead of ``adata.X``.
    method
        Only ``"vst"`` is implemented. R splikit's other method,
        ``"sum_deviance"``, is not ported — its algorithm (per-library NB
        deviance) has no dependency on R-specific numerics and can be
        approximated with :func:`scanpy.pp.highly_variable_genes` /
        :func:`scanpy.pp.calculate_qc_metrics` if needed.
    loess_span
        Span (neighborhood fraction) passed to the loess fit. Matches R
        splikit's hardcoded ``span = 0.3`` (and Seurat's ``loess.span``
        default) when left at the default.
    n_top
        If set, mark the top-N genes by ``var["standardize_variance"]`` as
        ``var[key_added] = True``. Otherwise mark every gene that
        contributed to the mean-variance fit (``mean > 0`` and ``var > 0``
        before standardization).
    n_threads
        OpenMP thread count for the two C++ passes (row mean/variance and
        standardization). The loess fit itself is single-threaded (it runs
        in Python via ``skmisc.loess``, not the C++ extension). Output is
        deterministic regardless of ``n_threads`` (disjoint per-row writes).
    key_added
        Boolean column in ``var`` flagging selected genes.
    inplace
        Mutate ``adata`` in place and return ``None`` (default), or return
        a copy with the same modifications.

    Returns
    -------
    ``None`` when ``inplace=True``; otherwise a copy.

    Notes
    -----
    Writes ``var["standardize_variance"]`` (float64; this is R splikit's own
    column name, kept verbatim for direct comparison) and ``var[key_added]``
    (bool). Stores call params under
    ``uns['scsplice']['params']['highly_variable_genes']``.

    Genes that fail the ``mean > 0 and variance > 0`` filter (constant or
    all-zero genes) do **not** receive ``NaN``: R splikit computes their
    standardized variance using a hardcoded fallback trend (``sd = 1.0``),
    and this port replicates that behaviour exactly rather than "fixing" it,
    so results match R splikit row-for-row. Such genes are excluded from
    ``var[key_added]`` selection when ``n_top`` is ``None``.

    Numerical equivalence with R
    -----------------------------
    The row mean/variance pass and the standardization pass are plain
    floating-point arithmetic ported from
    ``splikit/src/hvf_gene_expression.cpp`` and are ``np.allclose``-tight
    (not bit-exact) vs. R, the same as scsplice's other ported kernels — see
    the C++ header comments in ``hvg_vst.hpp``. The loess step uses
    ``skmisc.loess``, which wraps the *same* netlib/Cleveland-Grosse loess
    Fortran/C implementation that R's ``stats::loess()`` calls (both default
    to ``degree=2``, ``family="gaussian"``, ``surface="interpolate"``,
    ``cell=0.2``), so this is genuine — not merely approximate — parity with
    R, modulo ordinary cross-platform floating-point differences.
    """
    if method != "vst":
        raise ValueError(
            f"method must be 'vst' (the only method scsplice ports), got {method!r}"
        )
    if not isinstance(loess_span, int | float) or loess_span <= 0:
        raise ValueError(f"loess_span must be positive, got {loess_span!r}")
    if isinstance(n_threads, bool) or not isinstance(n_threads, int | np.integer):
        raise ValueError("n_threads must be a positive integer")
    if n_threads < 1:
        raise ValueError(f"n_threads must be positive, got {n_threads}")
    if not isinstance(key_added, str) or not key_added:
        raise ValueError("key_added must be a non-empty string")
    if n_top is not None and n_top <= 0:
        raise ValueError(f"n_top must be positive, got {n_top}")

    cpp = _import_extension()
    loess = _import_loess()

    if not inplace:
        adata = adata.copy()

    X = adata.layers[layer] if layer is not None else adata.X
    if X is None:
        raise ValueError(
            "adata.X is None and no layer was given; pass layer=... or "
            "populate adata.X with gene counts."
        )
    if not sp.issparse(X):
        raise TypeError(
            f"Gene expression matrix must be scipy.sparse, got {type(X).__name__}"
        )
    if not isinstance(X, sp.csc_matrix):
        X = sp.csc_matrix(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # AnnData is cells x genes; the kernel works in genes x cells.
    X_T = X.T.tocsc()
    n_genes = X_T.shape[0]
    n_cells = X_T.shape[1]

    mean, var = cpp.hvg_row_mean_var(X_T, int(n_threads))
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    good = (mean > 0) & (var > 0)
    if not good.any():
        raise ValueError(
            "No genes have both a positive mean and positive variance; "
            "cannot fit a mean-variance trend. Check that adata.X (or the "
            "given layer) holds raw, non-degenerate gene counts."
        )

    log_mean_good = np.log10(mean[good])
    log_var_good = np.log10(var[good])

    model = loess(log_mean_good, log_var_good, span=float(loess_span), degree=2)
    model.fit()
    fitted_log_var = np.asarray(model.outputs.fitted_values, dtype=np.float64)

    # R defaults: expectedVar=0.0, sd=1.0 for every gene, overwritten only for
    # "good" genes by the fitted trend. Replicated verbatim (see docstring).
    sd = np.ones(n_genes, dtype=np.float64)
    expected_var_good = np.power(10.0, fitted_log_var)
    sd_good = np.where(expected_var_good > 0, np.sqrt(expected_var_good), 1.0)
    sd[good] = sd_good

    standardize_variance = np.asarray(
        cpp.hvg_standardize_variance(X_T, mean, sd, int(n_threads)),
        dtype=np.float64,
    )

    adata.var["standardize_variance"] = standardize_variance

    selected = np.zeros(n_genes, dtype=bool)
    if n_top is None:
        selected[good] = True
    else:
        good_idx = np.where(good)[0]
        order = np.argsort(-standardize_variance[good_idx], kind="stable")
        top_local = order[: int(n_top)]
        selected[good_idx[top_local]] = True
    adata.var[key_added] = selected

    ns = setdefault_scsplice_ns(adata)
    ns.setdefault("params", {})["highly_variable_genes"] = {
        "method": method,
        "layer": layer,
        "loess_span": float(loess_span),
        "n_top": None if n_top is None else int(n_top),
        "n_threads": int(n_threads),
        "key_added": str(key_added),
        "n_genes_input": int(n_genes),
        "n_genes_fitted": int(good.sum()),
        "n_cells": int(n_cells),
    }

    return adata if not inplace else None
