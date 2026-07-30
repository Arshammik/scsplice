"""Whitelist + spatial-tissue resolution shared by all STARsolo readers.

The ``read_starsolo*`` family supports three sources of barcode subsetting,
in strict precedence order *per sample*:

1. ``tissue_positions[i]`` — Space Ranger ``tissue_positions[_list].csv``;
   barcodes used as the whitelist AND populate spatial obs/obsm/uns.
2. ``barcode_whitelists[i]`` — explicit per-sample list / file / sequence.
3. ``use_internal_whitelist=True`` — fall back to the STARsolo internal
   ``Solo.out/Gene/filtered/barcodes.tsv`` file when present.
4. Otherwise — no whitelist; retain the full selected matrix barcode set.

When (1) or (2) fires, the readers historically source counts from ``raw/``
(a strict superset) and intersect with the supplied whitelist. The gene reader
now selects its matrix independently: its default and compatibility ``auto``
mode still use raw for external/spatial whitelists, while an explicit
``matrix_source="filtered"`` can intentionally restrict the available cells.
SJ and Velocyto only have ``raw/`` to begin with.

This module is intentionally side-effect-free: it parses inputs into
plain Python objects (sets of barcodes; spatial DataFrames). The reader
modules apply those to MTX matrices.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "ResolvedWhitelist",
    "load_barcode_whitelist",
    "load_tissue_positions",
    "normalize_per_sample_arg",
    "resolve_whitelist",
]


_NT_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


# Space Ranger 2.x column order with header. Space Ranger 1.x is the same
# columns but no header (file is named ``tissue_positions_list.csv``).
_TISSUE_POSITIONS_COLS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)


@dataclass(frozen=True)
class ResolvedWhitelist:
    """The result of resolving per-sample whitelist precedence.

    Attributes
    ----------
    barcodes
        ``set[str]`` of barcodes to keep, or ``None`` to keep all.
    source
        Which precedence rule fired: ``"tissue_positions"`` /
        ``"explicit"`` / ``"internal"`` / ``"none"``.
    spatial
        Optional spatial DataFrame keyed by barcode with columns
        ``in_tissue``, ``array_row``, ``array_col``,
        ``pxl_row_in_fullres``, ``pxl_col_in_fullres``. Only set when
        ``source == "tissue_positions"``.
    library_id
        Squidpy library_id for ``uns["spatial"][library_id]``. Only set
        when ``spatial is not None``.
    """

    barcodes: set[str] | None
    source: Literal["tissue_positions", "explicit", "internal", "none"]
    spatial: pd.DataFrame | None = None
    library_id: str | None = None


def normalize_per_sample_arg(
    arg, n_samples: int, *, name: str
) -> list:
    """Validate and pad a per-sample sequence-or-None argument.

    The reader public API accepts ``None`` (broadcast to all samples) or a
    sequence whose length matches ``len(sample_dirs)``. Returns a list of
    length ``n_samples``; raises ``ValueError`` otherwise.
    """
    if arg is None:
        return [None] * n_samples
    arg = list(arg)
    if len(arg) != n_samples:
        raise ValueError(
            f"len({name})={len(arg)} must equal n_samples={n_samples}"
        )
    return arg


def load_barcode_whitelist(
    spec: str | Path | Sequence[str],
) -> set[str]:
    """Materialize a whitelist spec into a ``set[str]``.

    ``spec`` is either a path to a one-barcode-per-line text file (gzip
    transparently supported via ``compression="infer"``) or an in-memory
    sequence of barcodes.
    """
    if isinstance(spec, (str, Path)):
        # Read first column only; some users hand us multi-column files.
        df = pd.read_csv(
            spec, sep="\t", header=None, dtype=str, compression="infer",
            usecols=[0], names=["barcode"],
        )
        out = set(df["barcode"].astype(str).tolist())
    else:
        out = set(map(str, spec))
    # Sanity-check: at least ONE entry should look like a nucleotide sequence.
    # If the user passed a CSV-formatted whitelist by mistake, the entire
    # first line becomes one "barcode" containing commas — catch that here.
    if out and not any(_NT_RE.match(b) for b in out):
        warnings.warn(
            "load_barcode_whitelist: none of the loaded entries look like a "
            "nucleotide barcode (regex ^[ACGTN]+$). The input may be a CSV "
            "or contain malformed lines; whitelist intersection will likely "
            "be empty.",
            stacklevel=3,
        )
    return out


def load_tissue_positions(path: str | Path) -> pd.DataFrame:
    """Read a Space Ranger ``tissue_positions[_list].csv``.

    Detects v2 (with header) vs v1 (no header) by sniffing the first line.
    Returns a DataFrame indexed by ``barcode`` with columns matching
    ``_TISSUE_POSITIONS_COLS[1:]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tissue_positions file not found: {path}")

    # Sniff first line for header. v2 files write the canonical column
    # names; v1 files start with a barcode like "AACAGTC...".
    with path.open() as fh:
        first = fh.readline()
    has_header = (
        "barcode" in first.lower()
        and ("in_tissue" in first.lower() or "in_tissue" in first)
    )
    if has_header:
        # In the header branch, key dtype by column NAME ("barcode") instead
        # of positional 0; the first column may not be at position 0 if the
        # file has been re-ordered, and named keys are robust.
        df = pd.read_csv(path, dtype={"barcode": str})
        # Allow trivial column-name variations.
        rename_map = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(
            path, header=None, names=list(_TISSUE_POSITIONS_COLS),
            dtype={0: str},
        )

    needed = list(_TISSUE_POSITIONS_COLS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"tissue_positions {path} missing columns: {missing}; "
            f"saw: {list(df.columns)}"
        )

    df = df.loc[:, needed].copy()
    # Unconditional str cast: handles all-numeric barcodes (legitimate for
    # some custom spatial assays) that would otherwise round-trip as int64
    # and silently fail set-intersection with str MTX barcodes.
    df["barcode"] = df["barcode"].astype(str)
    df["in_tissue"] = df["in_tissue"].astype(np.int8)
    df["array_row"] = df["array_row"].astype(np.int32)
    df["array_col"] = df["array_col"].astype(np.int32)
    df["pxl_row_in_fullres"] = df["pxl_row_in_fullres"].astype(np.float64)
    df["pxl_col_in_fullres"] = df["pxl_col_in_fullres"].astype(np.float64)
    df = df.set_index("barcode", drop=True)
    return df


def resolve_whitelist(
    *,
    sample_id: str,
    barcode_whitelist: str | Path | Sequence[str] | None,
    tissue_positions: str | Path | None,
    use_internal_whitelist: bool,
    internal_whitelist_path: Path | None,
    spatial_library_id: str | None,
) -> ResolvedWhitelist:
    """Implement the per-sample precedence rule documented at module top.

    ``internal_whitelist_path`` may be ``None`` if the caller (e.g. the SJ
    reader) does not have a filtered/ companion at all; in that case the
    "internal" branch is silently skipped.
    """
    # 1. tissue_positions wins
    if tissue_positions is not None:
        if barcode_whitelist is not None:
            warnings.warn(
                f"Sample {sample_id!r}: both tissue_positions and "
                "barcode_whitelists provided; tissue_positions takes "
                "precedence (the spatial barcode set is more specific).",
                stacklevel=3,
            )
        spatial_df = load_tissue_positions(tissue_positions)
        bc_set = set(spatial_df.index.tolist())
        lib_id = spatial_library_id if spatial_library_id is not None else sample_id
        return ResolvedWhitelist(
            barcodes=bc_set,
            source="tissue_positions",
            spatial=spatial_df,
            library_id=lib_id,
        )
    # 2. explicit whitelist
    if barcode_whitelist is not None:
        return ResolvedWhitelist(
            barcodes=load_barcode_whitelist(barcode_whitelist),
            source="explicit",
        )
    # 3. internal whitelist fallback
    if use_internal_whitelist and internal_whitelist_path is not None:
        if internal_whitelist_path.exists():
            return ResolvedWhitelist(
                barcodes=load_barcode_whitelist(internal_whitelist_path),
                source="internal",
            )
        warnings.warn(
            f"Sample {sample_id!r}: use_internal_whitelist=True but "
            f"{internal_whitelist_path} not found; proceeding without whitelist.",
            stacklevel=3,
        )
    # 4. no whitelist
    return ResolvedWhitelist(barcodes=None, source="none")
