"""Schema and invariant checks for the splicing AnnData.

These helpers are called at the top of every public ``tl`` / ``pp`` function so
the kernels can rely on a known-good layout. Keep them cheap (O(1) checks where
possible, no copies).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp


__all__ = [
    "REQUIRED_VAR_COLUMNS",
    "invalidate_m2",
    "mark_m2_valid",
    "validate_m1_layer",
    "validate_paired_layers",
    "validate_var_schema",
]


REQUIRED_VAR_COLUMNS: tuple[str, ...] = (
    "chr",
    "start",
    "end",
    "strand",
    "row_names_mtx",
    "group_id",
    "group_kind",
    "group_count",
)


def validate_m1_layer(adata: ad.AnnData) -> None:
    """Assert that ``layers['M1']`` is present, sparse, float64, shape-aligned."""
    if "M1" not in adata.layers:
        raise KeyError(
            "adata.layers['M1'] is required. Build the splicing AnnData via "
            "splikit.io.read_starsolo()."
        )
    M1 = adata.layers["M1"]
    if not sp.issparse(M1):
        raise TypeError(
            f"layers['M1'] must be scipy.sparse, got {type(M1).__name__}. "
            "splikit-py never densifies M1; convert before assigning."
        )
    if M1.shape != adata.shape:
        raise ValueError(f"layers['M1'] shape {M1.shape} != adata.shape {adata.shape}")
    if M1.dtype != np.float64:
        raise TypeError(
            f"layers['M1'] must be float64 (matches R double), got {M1.dtype}"
        )


def validate_paired_layers(adata: ad.AnnData, *, require_m2_valid: bool = True) -> None:
    """Assert M1 and M2 are both present, sparse, float64, and shape-aligned.

    When ``require_m2_valid=True`` (the default) also assert
    ``adata.uns['splikit']['m2_valid']`` so callers see a clear error rather than
    silently consuming a stale exclusion matrix.
    """
    validate_m1_layer(adata)
    if "M2" not in adata.layers:
        raise KeyError(
            "adata.layers['M2'] is required; call splikit.tl.make_m2(adata) first"
        )
    M1, M2 = adata.layers["M1"], adata.layers["M2"]
    if not sp.issparse(M2):
        raise TypeError(
            f"layers['M2'] must be scipy.sparse, got {type(M2).__name__}"
        )
    if M2.shape != M1.shape:
        raise ValueError(
            f"layers['M2'] shape {M2.shape} != layers['M1'] shape {M1.shape}"
        )
    if M2.dtype != np.float64:
        raise TypeError(
            f"layers['M2'] must be float64 (matches R double), got {M2.dtype}"
        )
    if require_m2_valid:
        ns = adata.uns.get("splikit", {})
        if not ns.get("m2_valid", False):
            raise RuntimeError(
                "M2 is stale (uns['splikit']['m2_valid'] is False). "
                "Any operation that mutated layers['M1'] or subsetted the var "
                "axis invalidates M2. Call splikit.tl.make_m2(adata) again."
            )


def validate_var_schema(adata: ad.AnnData) -> None:
    """Assert the var-axis invariants the kernels rely on.

    Checked:
      - all of ``REQUIRED_VAR_COLUMNS`` are present
      - ``var_names`` are globally unique (no ``var_names_make_unique`` mangling)
      - ``var['group_kind']`` only contains ``'S'`` and ``'E'``
    """
    missing = [c for c in REQUIRED_VAR_COLUMNS if c not in adata.var.columns]
    if missing:
        raise KeyError(
            f"adata.var is missing required columns: {missing}. "
            "Required: " + ", ".join(REQUIRED_VAR_COLUMNS)
        )
    if not adata.var_names.is_unique:
        n_dup = int(adata.var_names.duplicated(keep=False).sum())
        raise ValueError(
            f"adata.var_names must be globally unique ({n_dup} duplicate entries). "
            "Do NOT call var_names_make_unique() — it appends -1/-2 suffixes that "
            "destroy the load-bearing _S / _E LJV-grouping scheme."
        )
    kinds = {str(k) for k in adata.var["group_kind"].unique()}
    bad = sorted(kinds - {"S", "E"})
    if bad:
        raise ValueError(f"var['group_kind'] must be 'S' or 'E', found: {bad}")


def mark_m2_valid(adata: ad.AnnData) -> None:
    """Set ``uns['splikit']['m2_valid'] = True``. Called by ``splk.tl.make_m2``."""
    ns = adata.uns.setdefault("splikit", {})
    ns["m2_valid"] = True


def invalidate_m2(adata: ad.AnnData) -> None:
    """Set ``uns['splikit']['m2_valid'] = False``.

    Call this from any operation that mutates ``layers['M1']`` or subsets the var
    axis — both invalidate the LJV co-membership relationship between M1 and M2.
    """
    ns = adata.uns.setdefault("splikit", {})
    ns["m2_valid"] = False
