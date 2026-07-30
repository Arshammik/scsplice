"""Per-event signed pseudo-correlation against an external (events x cells) matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scsplice._core._validators import (
    get_scsplice_ns,
    setdefault_scsplice_ns,
    validate_paired_layers,
    validate_var_schema,
)

__all__ = [
    "PseudoCorrelationResult",
    "get_pseudo_correlation_result",
    "pseudo_correlation",
]


@dataclass(frozen=True)
class PseudoCorrelationResult:
    """Export-friendly view of a pseudo-correlation computation.

    Attributes
    ----------
    statistics
        One row per retained event with observed and empirical-null summaries.
    null_distribution
        Long table with one row per retained event and permutation.
    metadata
        Computation parameters and retained event/null-draw counts.
    """

    statistics: pd.DataFrame
    null_distribution: pd.DataFrame
    metadata: dict[str, object]


def _import_extension():
    try:
        from scsplice import _scsplice_cpp
    except ImportError as exc:
        raise ImportError(
            "scsplice's C++ extension (_scsplice_cpp) is not built. "
            "Run `pip install -e .` (or install a wheel) in an environment "
            "with Eigen3 available."
        ) from exc
    return _scsplice_cpp


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment without an additional dependency."""
    adjusted = np.full(pvalues.shape, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(~np.isnan(pvalues))
    if valid_idx.size == 0:
        return adjusted

    valid = pvalues[valid_idx]
    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    scale = valid.size / np.arange(1, valid.size + 1, dtype=np.float64)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.minimum(ranked_adjusted, 1.0)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    adjusted[valid_idx] = ranked_adjusted[inverse]
    return adjusted


def _result_keys(key: str) -> dict[str, str]:
    return {
        "observed": key,
        "null_draws": f"{key}_null",
        "null_mean": f"{key}_null_mean",
        "null_sd": f"{key}_null_sd",
        "n_valid": f"{key}_n_perm_valid",
        "pvalue": f"{key}_emp_pvalue",
        "padj": f"{key}_emp_padj",
    }


def get_pseudo_correlation_result(
    adata: ad.AnnData,
    *,
    key: str = "pseudo_correlation",
) -> PseudoCorrelationResult:
    """Materialize exportable pseudo-correlation tables from ``adata``.

    The computation itself remains AnnData-native. This helper reconstructs
    the structured result used by R splikit 2.3.3 without persisting a second,
    long-form copy of the null draws in the ``.h5ad`` file.

    Parameters
    ----------
    adata
        AnnData previously processed by :func:`pseudo_correlation`.
    key
        ``key_added`` used for the computation.

    Returns
    -------
    PseudoCorrelationResult
        Per-event statistics, long event/permutation null draws, and metadata.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")

    keys = _result_keys(key)
    required = (
        keys["observed"],
        keys["null_mean"],
        keys["null_sd"],
        keys["n_valid"],
        keys["pvalue"],
        keys["padj"],
    )
    missing = [column for column in required if column not in adata.var]
    if missing:
        raise KeyError(
            f"No complete pseudo-correlation result for key {key!r}; "
            f"missing adata.var columns: {missing}"
        )

    ns = get_scsplice_ns(adata)
    metadata_by_key = ns.get("pseudo_correlation", {})
    if key not in metadata_by_key:
        raise KeyError(
            f"No pseudo-correlation metadata for key {key!r} in "
            "adata.uns['scsplice']['pseudo_correlation']"
        )
    metadata = dict(metadata_by_key[key])
    n_permutations = int(metadata["permutation_count"])

    observed = np.asarray(adata.var[key], dtype=np.float64)
    null_mean = np.asarray(adata.var[keys["null_mean"]], dtype=np.float64)
    keep = ~np.isnan(observed) & ~np.isnan(null_mean)
    events = np.asarray(adata.var_names.astype(str), dtype=object)

    statistics = pd.DataFrame(
        {
            "event": events[keep],
            "pseudo_correlation": observed[keep],
            "null_distribution": null_mean[keep],
            "null_sd": np.asarray(adata.var[keys["null_sd"]], dtype=np.float64)[keep],
            "n_perm_valid": np.asarray(adata.var[keys["n_valid"]], dtype=np.int64)[keep],
            "emp_pvalue": np.asarray(adata.var[keys["pvalue"]], dtype=np.float64)[keep],
            "emp_padj": np.asarray(adata.var[keys["padj"]], dtype=np.float64)[keep],
        }
    )

    n_retained = int(keep.sum())
    if n_permutations == 0:
        null_distribution = pd.DataFrame(
            {
                "event": pd.Series(dtype=object),
                "permutation": pd.Series(dtype=np.int64),
                "null_pseudo_correlation": pd.Series(dtype=np.float64),
            }
        )
    else:
        if keys["null_draws"] not in adata.varm:
            raise KeyError(
                f"Missing adata.varm[{keys['null_draws']!r}] for a result with "
                f"{n_permutations} permutations"
            )
        nulls = np.asarray(adata.varm[keys["null_draws"]], dtype=np.float64)
        expected_shape = (adata.n_vars, n_permutations)
        if nulls.shape != expected_shape:
            raise ValueError(
                f"adata.varm[{keys['null_draws']!r}] has shape {nulls.shape}; "
                f"expected {expected_shape} from stored metadata"
            )
        retained_nulls = nulls[keep, :]
        null_distribution = pd.DataFrame(
            {
                "event": np.tile(events[keep], n_permutations),
                "permutation": np.repeat(
                    np.arange(1, n_permutations + 1, dtype=np.int64),
                    n_retained,
                ),
                "null_pseudo_correlation": retained_nulls.T.reshape(-1),
            }
        )

    return PseudoCorrelationResult(
        statistics=statistics,
        null_distribution=null_distribution,
        metadata=metadata,
    )


def pseudo_correlation(
    adata: ad.AnnData,
    zdb: np.ndarray,
    *,
    metric: Literal["CoxSnell", "Nagelkerke"] = "CoxSnell",
    n_permutations: int = 100,
    seed: int | None = None,
    n_threads: int = 1,
    key_added: str = "pseudo_correlation",
    inplace: bool = True,
) -> ad.AnnData | None:
    """Compute per-event signed pseudo-correlation against ``zdb``.

    For each event, fit a binomial GLM (logistic link) via IRLS with design
    matrix ``[intercept | zdb[event, valid_cells]]`` against the M1/(M1+M2)
    response, where ``valid_cells`` are cells with ``M1+M2 > 0``. Return
    ``sqrt(R^2) * sign(slope)`` where ``R^2`` is Cox-Snell or Nagelkerke
    pseudo-R^2. Events with fewer than two valid cells, all-zero M1 or M2,
    singular Hessian, or negative R^2 receive ``NaN``.

    Parameters
    ----------
    adata
        Splicing AnnData with valid M1 / M2 layers (``uns['scsplice']['m2_valid']``).
    zdb
        Dense ``(n_var, n_obs)`` numpy array — one predictor value per
        ``(event, cell)``. Note: this is **events x cells**, NOT
        ``(events x K_latent_dims)``. Callers with a K-dim latent
        embedding must compute ``zdb`` per dim separately and call this
        function once per dim.
    metric
        ``"CoxSnell"`` (default) or ``"Nagelkerke"``.
    n_permutations
        Number of event-wise null draws, generated by column-permuting ``zdb``
        and recomputing the kernel. Defaults to 100, matching R splikit 2.3.2.
        Use 0 for an observed-only computation. Raw draws are stored in
        ``adata.varm[key_added + '_null']`` with shape
        ``(n_var, n_permutations)``; null summaries and empirical inference are
        stored in ``adata.var``.
    seed
        Seed for the column permutation RNG (``numpy.random.default_rng``).
        Cross-language bit-equivalence with R is impossible (PCG64 vs
        Mersenne Twister); permutations are reproducible across Python
        runs given the same seed.
    n_threads
        OpenMP thread count for the per-event outer loop. Per-event
        results are bit-identical regardless of n_threads (disjoint scalar
        writes).
    key_added
        Output column name in ``adata.var``.
    inplace
        Mutate ``adata`` in place and return ``None`` (default), or
        operate on a copy and return it.

    Returns
    -------
    ``None`` when ``inplace=True``; otherwise the modified copy.

    Notes
    -----
    Use :func:`get_pseudo_correlation_result` to materialize R-compatible
    statistics and long null-distribution tables for export. The pooled null
    table is descriptive; empirical p-values always use each event's own draws.
    """
    cpp = _import_extension()

    if metric not in ("CoxSnell", "Nagelkerke"):
        raise ValueError(f"metric must be 'CoxSnell' or 'Nagelkerke', got {metric!r}")
    zdb = np.asarray(zdb, dtype=np.float64)
    if zdb.ndim != 2:
        raise ValueError(f"zdb must be 2D, got shape {zdb.shape}")
    if zdb.shape != (adata.n_vars, adata.n_obs):
        raise ValueError(
            f"zdb shape {zdb.shape} != (n_var={adata.n_vars}, n_obs={adata.n_obs}); "
            "zdb is events x cells, not events x K. See docstring."
        )
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int | np.integer):
        raise ValueError("n_permutations must be a non-negative integer")
    if n_permutations < 0:
        raise ValueError(f"n_permutations must be non-negative, got {n_permutations}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int | np.integer)):
        raise ValueError("seed must be None or an integer")
    if isinstance(n_threads, bool) or not isinstance(n_threads, int | np.integer):
        raise ValueError("n_threads must be a positive integer")
    if n_threads < 1:
        raise ValueError(f"n_threads must be positive, got {n_threads}")
    if not isinstance(key_added, str) or not key_added:
        raise ValueError("key_added must be a non-empty string")

    n_permutations = int(n_permutations)
    n_threads = int(n_threads)
    seed = None if seed is None else int(seed)

    if not inplace:
        adata = adata.copy()

    validate_paired_layers(adata, require_m2_valid=True)
    validate_var_schema(adata)

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

    # AnnData layout is cells x events; the kernel works in events x cells.
    M1_T = M1.T.tocsc()
    M2_T = M2.T.tocsc()

    # Eigen MatrixXd is column-major; pybind11 will copy from numpy's C-order.
    # The cost is O(n_events * n_cells * 8) bytes — acceptable for typical sizes.
    Z = np.asfortranarray(zdb)

    point = np.asarray(
        cpp.pseudo_correlation(Z, M1_T, M2_T, str(metric), int(n_threads)),
        dtype=np.float64,
    ).ravel()
    adata.var[key_added] = point

    keys = _result_keys(key_added)
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        nulls = np.empty((adata.n_vars, n_permutations), dtype=np.float64)
        for k in range(n_permutations):
            perm = rng.permutation(adata.n_obs)
            Z_perm = np.asfortranarray(zdb[:, perm])
            nulls[:, k] = np.asarray(
                cpp.pseudo_correlation(Z_perm, M1_T, M2_T, str(metric), int(n_threads)),
                dtype=np.float64,
            ).ravel()
        adata.varm[keys["null_draws"]] = nulls

        valid = ~np.isnan(nulls)
        n_valid = valid.sum(axis=1, dtype=np.int64)
        null_sum = np.nansum(nulls, axis=1)
        null_sum_sq = np.nansum(nulls * nulls, axis=1)

        null_mean = np.full(adata.n_vars, np.nan, dtype=np.float64)
        np.divide(null_sum, n_valid, out=null_mean, where=n_valid > 0)

        null_variance = np.full(adata.n_vars, np.nan, dtype=np.float64)
        eligible_sd = n_valid > 1
        null_variance[eligible_sd] = (
            null_sum_sq[eligible_sd] - (null_sum[eligible_sd] ** 2) / n_valid[eligible_sd]
        ) / (n_valid[eligible_sd] - 1)
        tiny_negative = np.isfinite(null_variance) & (null_variance < 0)
        null_variance[tiny_negative] = 0.0
        null_sd = np.sqrt(null_variance)

        exceed = np.sum(
            valid & (np.abs(nulls) >= np.abs(point)[:, None]),
            axis=1,
            dtype=np.int64,
        )
        emp_pvalue = (exceed + 1) / (n_valid + 1)
        emp_pvalue = np.asarray(emp_pvalue, dtype=np.float64)
        emp_pvalue[np.isnan(point) | (n_valid == 0)] = np.nan
    else:
        if keys["null_draws"] in adata.varm:
            del adata.varm[keys["null_draws"]]
        n_valid = np.zeros(adata.n_vars, dtype=np.int64)
        null_mean = np.full(adata.n_vars, np.nan, dtype=np.float64)
        null_sd = np.full(adata.n_vars, np.nan, dtype=np.float64)
        emp_pvalue = np.full(adata.n_vars, np.nan, dtype=np.float64)

    retained = ~np.isnan(point) & ~np.isnan(null_mean)
    emp_padj = np.full(adata.n_vars, np.nan, dtype=np.float64)
    emp_padj[retained] = _bh_adjust(emp_pvalue[retained])

    adata.var[keys["null_mean"]] = null_mean
    adata.var[keys["null_sd"]] = null_sd
    adata.var[keys["n_valid"]] = n_valid
    adata.var[keys["pvalue"]] = emp_pvalue
    adata.var[keys["padj"]] = emp_padj

    ns = setdefault_scsplice_ns(adata)
    params = {
        "metric": str(metric),
        "n_permutations": n_permutations,
        "seed": seed,
        "n_threads": n_threads,
        "key_added": str(key_added),
    }
    ns.setdefault("params", {})["pseudo_correlation"] = params

    n_retained = int(retained.sum())
    n_null_valid = 0
    if n_permutations > 0:
        n_null_valid = int(np.sum((~np.isnan(nulls))[retained, :]))
    ns.setdefault("pseudo_correlation", {})[key_added] = {
        "metric": str(metric),
        "permutation_count": n_permutations,
        "permutation_seed": seed,
        "n_threads": n_threads,
        "key_added": str(key_added),
        "n_events_input": int(adata.n_vars),
        "n_events_retained": n_retained,
        "n_null_draws": n_retained * n_permutations,
        "n_null_valid": n_null_valid,
        "pooled_null_usage": "descriptive",
    }

    return adata if not inplace else None
