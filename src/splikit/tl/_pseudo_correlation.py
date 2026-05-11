"""Per-event signed pseudo-correlation against an external (events x cells) matrix."""

from __future__ import annotations

from typing import Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp

from splikit._core._validators import (
    validate_paired_layers,
    validate_var_schema,
)


__all__ = ["pseudo_correlation"]


def _import_extension():
    try:
        from splikit import _scsplice_cpp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "splikit's C++ extension (_scsplice_cpp) is not built. "
            "Run `pip install -e .` (or install a wheel) in an environment "
            "with Eigen3 available."
        ) from exc
    return _scsplice_cpp


def pseudo_correlation(
    adata: ad.AnnData,
    zdb: np.ndarray,
    *,
    metric: Literal["CoxSnell", "Nagelkerke"] = "CoxSnell",
    n_permutations: int = 0,
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
        Splicing AnnData with valid M1 / M2 layers (``uns['splikit']['m2_valid']``).
    zdb
        Dense ``(n_var, n_obs)`` numpy array — one predictor value per
        ``(event, cell)``. Note: this is **events x cells**, NOT
        ``(events x K_latent_dims)``. Callers with a K-dim latent
        embedding must compute ``zdb`` per dim separately and call this
        function once per dim.
    metric
        ``"CoxSnell"`` (default) or ``"Nagelkerke"``.
    n_permutations
        If > 0, also compute ``n_permutations`` null distributions by
        column-permuting ``zdb`` (cell axis) and recomputing the kernel.
        Stored in ``adata.varm[key_added + '_null']`` of shape
        ``(n_var, n_permutations)``.
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
    """
    cpp = _import_extension()

    if metric not in ("CoxSnell", "Nagelkerke"):
        raise ValueError(
            f"metric must be 'CoxSnell' or 'Nagelkerke', got {metric!r}"
        )
    zdb = np.asarray(zdb, dtype=np.float64)
    if zdb.ndim != 2:
        raise ValueError(f"zdb must be 2D, got shape {zdb.shape}")
    if zdb.shape != (adata.n_vars, adata.n_obs):
        raise ValueError(
            f"zdb shape {zdb.shape} != (n_var={adata.n_vars}, n_obs={adata.n_obs}); "
            "zdb is events x cells, not events x K. See docstring."
        )
    if n_permutations < 0:
        raise ValueError(f"n_permutations must be non-negative, got {n_permutations}")

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

    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        nulls = np.empty((adata.n_vars, int(n_permutations)), dtype=np.float64)
        for k in range(int(n_permutations)):
            perm = rng.permutation(adata.n_obs)
            Z_perm = np.asfortranarray(zdb[:, perm])
            nulls[:, k] = np.asarray(
                cpp.pseudo_correlation(
                    Z_perm, M1_T, M2_T, str(metric), int(n_threads)
                ),
                dtype=np.float64,
            ).ravel()
        adata.varm[key_added + "_null"] = nulls

    ns = adata.uns.setdefault("splikit", {})
    ns.setdefault("params", {})["pseudo_correlation"] = {
        "metric": str(metric),
        "n_permutations": int(n_permutations),
        "seed": None if seed is None else int(seed),
        "n_threads": int(n_threads),
        "key_added": str(key_added),
    }

    return adata if not inplace else None
