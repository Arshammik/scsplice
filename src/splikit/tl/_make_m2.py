"""Build the LJV exclusion matrix M2 from M1 + var['group_id']."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import scipy.sparse as sp

from splikit._core._validators import (
    mark_m2_valid,
    validate_m1_layer,
    validate_var_schema,
)

if TYPE_CHECKING:
    pass


__all__ = ["make_m2"]


def _import_extension():
    """Defer the C++ extension import so ``import splikit.tl`` works without a build."""
    try:
        from splikit import _splikit_cpp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "splikit's C++ extension (_splikit_cpp) is not built. "
            "Run `pip install -e .` (or `pip install splikit-py`) in an "
            "environment with Eigen3 available."
        ) from exc
    return _splikit_cpp


def _validate_dense_zero_indexed(group_ids: np.ndarray) -> None:
    """Assert ``group_ids`` is dense 0..G-1 contiguous (the C++ kernel precondition)."""
    if group_ids.size == 0:
        return
    g_min = int(group_ids.min())
    g_max = int(group_ids.max())
    if g_min < 0:
        raise ValueError(f"var['group_id'] must be non-negative, got min={g_min}")
    expected = set(range(g_max + 1))
    actual = set(group_ids.tolist())
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(
            f"var['group_id'] must be a dense 0..{g_max} mapping, missing: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )


def make_m2(
    adata: ad.AnnData,
    *,
    n_threads: int = 1,
    copy: bool = False,
) -> ad.AnnData | None:
    """Build the exclusion matrix M2 and store it in ``adata.layers['M2']``.

    For every event ``i`` (var) and cell ``j`` (obs):

        M2[j, i] = sum(M1[j, k] for k in same LJV group as i) - M1[j, i]

    Entries that are exactly zero are dropped (no epsilon threshold). M2 has the
    same shape, sparse format (CSC), and dtype (``float64``) as M1.

    Parameters
    ----------
    adata
        Splicing AnnData built by :func:`splikit.io.read_starsolo` or equivalent.
        Must have ``layers['M1']`` (cells × events, sparse CSC, ``float64``) and
        the var columns required by :func:`splikit._core._validators.validate_var_schema`.
    n_threads
        OpenMP thread count for the kernel. ``n_threads > 1`` requires a build
        with OpenMP enabled (check ``splikit._splikit_cpp.__openmp__``). The
        kernel produces byte-identical output regardless of thread count.
    copy
        Mutate ``adata`` in place and return ``None`` (default), or operate on a
        deep copy and return the new AnnData.

    Returns
    -------
    ``None`` when ``copy=False``; the new AnnData when ``copy=True``.

    Notes
    -----
    Sets ``adata.uns['splikit']['m2_valid'] = True`` and records the call params
    under ``adata.uns['splikit']['params']['make_m2']``.

    The kernel reads ``layers['M1']``, transposes to events × cells at the C++
    boundary (the kernel works in that layout), and transposes the result back
    to AnnData layout (cells × events) before storing in ``layers['M2']``.
    """
    cpp = _import_extension()

    if copy:
        adata = adata.copy()

    validate_m1_layer(adata)
    validate_var_schema(adata)

    M1 = adata.layers["M1"]
    if not isinstance(M1, sp.csc_matrix):
        M1 = sp.csc_matrix(M1)
    if M1.dtype != np.float64:
        M1 = M1.astype(np.float64)

    # AnnData is cells × events; the kernel works in events × cells.
    M1_T = M1.T.tocsc()

    group_ids = np.asarray(adata.var["group_id"], dtype=np.int32)
    if group_ids.shape != (adata.n_vars,):
        raise ValueError(
            f"var['group_id'] shape {group_ids.shape} != (n_vars={adata.n_vars},)"
        )
    _validate_dense_zero_indexed(group_ids)

    M2_T = cpp.make_m2(M1_T, group_ids, int(n_threads))

    M2 = sp.csc_matrix(M2_T).T.tocsc()
    if M2.dtype != np.float64:
        M2 = M2.astype(np.float64)
    adata.layers["M2"] = M2

    mark_m2_valid(adata)
    ns = adata.uns.setdefault("splikit", {})
    ns.setdefault("params", {})["make_m2"] = {"n_threads": int(n_threads)}

    return adata if copy else None
