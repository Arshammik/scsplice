"""STARsolo SJ ingestion + LJV grouping → splicing AnnData.

The reader walks one or more ``Solo.out/SJ/`` directories, reads each sample's
``matrix.mtx + barcodes.tsv + ../../SJ.out.tab`` triplet, applies optional
multi-mapped / whitelist / min-counts filters, unions junctions across samples
with zero-padding, and finally expands each junction into the ``_S`` / ``_E``
LJV-grouped event rows that the downstream kernels expect.

The output AnnData has:

- ``layers["M1"]`` (cells × events, sparse CSC float64).
- ``var_names`` ``f"{chr}:{start}-{end}_S"`` or ``..._E``; globally unique
  by construction. Never call ``var_names_make_unique`` on the result.
- ``var`` columns: ``chr``, ``start``, ``end``, ``strand``, ``intron_motif``,
  ``is_annot``, ``unique_mapped`` (when present), ``row_names_mtx``,
  ``group_id`` (int32 dense ``0..G-1``), ``group_kind`` (categorical S/E),
  ``group_count`` (int32, ≥ 2 by construction).
- ``obs`` columns: ``barcode`` (raw 16-mer), ``sample_id`` (user-supplied).
- ``obs_names``: ``f"{barcode}-{sample_id}"``.
- ``uns["splikit"]``: ``version`` (currently 1), ``m2_valid`` (False),
  ``ljv_kind``, ``source`` ("starsolo"), ``params``.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from splikit._core._validators import (
    invalidate_m2,
    validate_m1_layer,
    validate_var_schema,
)
from splikit.io._whitelist import (
    ResolvedWhitelist,
    normalize_per_sample_arg,
    resolve_whitelist,
)


__all__ = ["read_starsolo"]


# STARsolo encodes strand as 0 (undefined) / 1 (+) / 2 (-).
_STAR_STRAND_DECODE = {0: ".", 1: "+", 2: "-"}


# Defensive caps on mmread inputs. A malformed matrix.mtx header that
# declares ``nnz=1e12`` would OOM scipy before any shape check fires.
# Both can be overridden via environment variable for users with truly
# enormous (but well-formed) inputs.
_MAX_MTX_NNZ: int = int(os.environ.get("SPLIKIT_MAX_MTX_NNZ", 5_000_000_000))
_MAX_MTX_SHAPE: int = int(os.environ.get("SPLIKIT_MAX_MTX_SHAPE", 10_000_000_000_000))


def _safe_mmread(path: Path | str, *, what: str = "matrix.mtx") -> sp.spmatrix:
    """``scipy.io.mmread`` with a header-only pre-check.

    Calls :func:`scipy.io.mminfo` first to read the ``(rows cols nnz)``
    header without allocating; if the file's declared size exceeds
    ``_MAX_MTX_NNZ`` or ``rows * cols > _MAX_MTX_SHAPE`` raises early with
    a clear message instead of OOMing inside scipy.
    """
    path = Path(path)
    info = sio.mminfo(str(path))
    rows, cols, nnz = int(info[0]), int(info[1]), int(info[2])
    if nnz > _MAX_MTX_NNZ:
        raise ValueError(
            f"{what} at {path} declares nnz={nnz:,} > _MAX_MTX_NNZ="
            f"{_MAX_MTX_NNZ:,}. Refusing to allocate. If this is a "
            "legitimate but very large MTX, raise the limit via the "
            "SPLIKIT_MAX_MTX_NNZ environment variable."
        )
    if rows * cols > _MAX_MTX_SHAPE:
        raise ValueError(
            f"{what} at {path} declares shape {rows}x{cols} (product "
            f"{rows * cols:,}) > _MAX_MTX_SHAPE={_MAX_MTX_SHAPE:,}. "
            "Refusing to allocate. Raise SPLIKIT_MAX_MTX_SHAPE to override."
        )
    return sio.mmread(str(path))


@dataclass(frozen=True)
class _SJPaths:
    matrix: Path
    barcodes: Path
    sj_tab: Path
    internal_whitelist: Path  # may not exist


@dataclass(frozen=True)
class _SampleArtifacts:
    features: pd.DataFrame  # one row per junction in this sample (post-filter)
    barcodes: np.ndarray  # raw 16-mers, length n_cells_kept
    mtx: sp.csc_matrix  # junctions × cells_kept, float64
    spatial: pd.DataFrame | None = None  # aligned to barcodes, when tissue_positions
    library_id: str | None = None
    whitelist_source: str = "none"


def _resolve_sj_paths(sj_dir: str | Path) -> _SJPaths:
    """Normalize ``sj_dir`` and locate the four input files.

    Accepts either ``<...>/Solo.out/SJ/`` (with a ``raw/`` subdir) or
    ``<...>/Solo.out/SJ/raw/`` directly.
    """
    sj_dir = Path(sj_dir).expanduser().resolve()
    if (sj_dir / "raw" / "matrix.mtx").exists() or (sj_dir / "raw" / "matrix.mtx.gz").exists():
        raw = sj_dir / "raw"
    elif (sj_dir / "matrix.mtx").exists() or (sj_dir / "matrix.mtx.gz").exists():
        raw = sj_dir
    else:
        raise FileNotFoundError(
            f"Cannot locate matrix.mtx (or .gz) under {sj_dir}. "
            f"Tried: {sj_dir / 'raw' / 'matrix.mtx'} and {sj_dir / 'matrix.mtx'}."
        )

    matrix = raw / "matrix.mtx" if (raw / "matrix.mtx").exists() else raw / "matrix.mtx.gz"
    barcodes = raw / "barcodes.tsv" if (raw / "barcodes.tsv").exists() else raw / "barcodes.tsv.gz"

    # SJ.out.tab is written by STAR alongside Solo.out, NOT inside it. So
    # for raw = <sample_root>/Solo.out/SJ/raw/, the file lives THREE levels up.
    # Some users may also drop it inside Solo.out (e.g., synthetic fixtures or
    # post-hoc reorganisations); accept those two locations as fallbacks.
    sj_tab_candidates = [
        raw.parent.parent.parent / "SJ.out.tab",  # canonical: sample root
        raw.parent.parent / "SJ.out.tab",          # legacy: inside Solo.out
        raw.parent / "SJ.out.tab",                 # user pointed at Solo.out/SJ
    ]
    sj_tab = next((p for p in sj_tab_candidates if p.exists()), sj_tab_candidates[0])
    if not sj_tab.exists():
        raise FileNotFoundError(
            f"Cannot locate SJ.out.tab. Tried: "
            f"{', '.join(str(p) for p in sj_tab_candidates)}."
        )

    # Internal whitelist (Gene/filtered/barcodes.tsv) lives inside Solo.out,
    # which is two levels up from raw/.
    internal_wl = raw.parent.parent / "Gene" / "filtered" / "barcodes.tsv"
    return _SJPaths(matrix=matrix, barcodes=barcodes, sj_tab=sj_tab,
                    internal_whitelist=internal_wl)


def _read_one_sample(
    sj_dir: str | Path,
    sample_id: str,
    *,
    keep_multi_mapped: bool,
    barcode_whitelist,
    tissue_positions,
    use_internal_whitelist: bool,
    spatial_library_id: str | None,
    verbose: bool,
) -> _SampleArtifacts:
    """Read one STARsolo SJ sample, apply per-sample filters, decode strand."""
    paths = _resolve_sj_paths(sj_dir)

    if verbose:
        print(f"[splikit.io] Reading sample {sample_id!r} from {paths.matrix.parent}")

    # MTX is junctions × cells per STARsolo convention.
    mtx = _safe_mmread(paths.matrix, what="SJ matrix.mtx").tocsc().astype(np.float64)
    n_jx_mtx, n_cells_mtx = mtx.shape

    barcodes = pd.read_csv(
        paths.barcodes, sep="\t", header=None, names=["barcode"],
        dtype=str, compression="infer",
    )["barcode"].to_numpy()

    # ``unique_mapped`` is a per-junction unique-read count; deeply sequenced
    # libraries can plausibly exceed int8 / int32 — use int64 to be safe.
    sj_cols = ["chr", "start", "end", "strand",
               "intron_motif", "is_annot", "unique_mapped"]
    features = pd.read_csv(
        paths.sj_tab, sep="\t", header=None,
        usecols=range(len(sj_cols)), names=sj_cols,
        dtype={"chr": str, "start": "int64", "end": "int64",
               "strand": "int8", "intron_motif": "int8",
               "is_annot": "int8", "unique_mapped": "int64"},
        compression="infer",
    )

    if mtx.shape[0] != len(features) or mtx.shape[1] != len(barcodes):
        raise ValueError(
            f"STARsolo dimension mismatch in sample {sample_id!r}:\n"
            f"  matrix.mtx header says    {n_jx_mtx} junctions x {n_cells_mtx} cells (source of truth)\n"
            f"  SJ.out.tab parsed to     {len(features)} rows\n"
            f"  barcodes.tsv parsed to   {len(barcodes)} lines\n"
            "Verify SJ.out.tab is well-formed (no comment rows) and from the "
            "same STARsolo run as matrix.mtx."
        )

    # Decode strand 0/1/2 → '.' / '+' / '-'.
    bad = ~features["strand"].isin(_STAR_STRAND_DECODE.keys())
    if bad.any():
        bad_rows = features.loc[bad, ["chr", "start", "end", "strand"]].head(5)
        raise ValueError(
            f"SJ.out.tab in sample {sample_id!r} has unrecognised strand values "
            f"(expected 0/1/2). First offending rows:\n{bad_rows}"
        )
    features = features.copy()
    features["strand"] = features["strand"].map(_STAR_STRAND_DECODE).astype(
        pd.CategoricalDtype(categories=[".", "+", "-"])
    )
    features["chr"] = features["chr"].astype("category")

    # Per-sample filters.
    if not keep_multi_mapped:
        keep_jx = features["unique_mapped"].to_numpy() > 0
        if not keep_jx.any():
            warnings.warn(
                f"Sample {sample_id!r}: no junctions survive keep_multi_mapped=False; "
                "this sample will contribute zero events to the output.",
                stacklevel=3,
            )
        mtx = mtx[keep_jx, :]
        features = features.loc[keep_jx].reset_index(drop=True)

    # Barcode whitelist via the shared resolver (raw 16-mers, before suffixing).
    resolved: ResolvedWhitelist = resolve_whitelist(
        sample_id=sample_id,
        barcode_whitelist=barcode_whitelist,
        tissue_positions=tissue_positions,
        use_internal_whitelist=use_internal_whitelist,
        internal_whitelist_path=paths.internal_whitelist,
        spatial_library_id=spatial_library_id,
    )
    wl_set = resolved.barcodes
    if wl_set is not None:
        # ``pd.Index(...).isin(set)`` is ~50× faster than a Python list-comp
        # on 6.7M barcodes and dtype-safe across str / numpy.str_ / arrow.
        keep_cells = np.asarray(pd.Index(barcodes).isin(wl_set), dtype=bool)
        if not keep_cells.any():
            warnings.warn(
                f"Sample {sample_id!r}: whitelist intersection is empty; "
                "this sample will contribute zero cells to the output.",
                stacklevel=3,
            )
        mtx = mtx[:, keep_cells]
        barcodes = barcodes[keep_cells]

    spatial_aligned = None
    if resolved.spatial is not None:
        spatial_aligned = resolved.spatial.reindex(barcodes)
        if spatial_aligned.isna().any().any():
            n_missing = int(spatial_aligned["in_tissue"].isna().sum())
            raise ValueError(
                f"SJ sample {sample_id!r}: {n_missing} kept barcodes are not "
                "present in tissue_positions; verify the tissue_positions file "
                "is from the same library as the MTX."
            )

    return _SampleArtifacts(
        features=features, barcodes=barcodes, mtx=mtx.tocsc(),
        spatial=spatial_aligned, library_id=resolved.library_id,
        whitelist_source=resolved.source,
    )


def _global_junction_table(
    artifacts: list[_SampleArtifacts],
) -> pd.DataFrame:
    """Build the global, deduplicated junction table across all samples.

    Per-junction QC flags (``is_annot``, ``unique_mapped``) are aggregated
    with ``max`` across samples so the strongest signal wins (a junction
    annotated in at least one sample is treated as annotated globally;
    its global ``unique_mapped`` count is the max observed across samples).
    Other columns take the first observed value (``chr``, ``start``,
    ``end``, ``strand`` are identity keys, ``intron_motif`` is constant
    per-junction in well-formed SJ.out.tab).
    """
    non_empty = [a.features for a in artifacts if len(a.features) > 0]
    if not non_empty:
        raise ValueError(
            "No junctions detected across any sample. Check that "
            "keep_multi_mapped wasn't filtering everything out and that "
            "SJ.out.tab is well-formed."
        )
    pooled = pd.concat(non_empty, axis=0, ignore_index=True)

    keys = ["chr", "start", "end", "strand"]
    # Cast categoricals back to plain types so groupby preserves them
    # without observed-cardinality warnings, then aggregate.
    pooled_for_agg = pooled.copy()
    if isinstance(pooled_for_agg["chr"].dtype, pd.CategoricalDtype):
        pooled_for_agg["chr"] = pooled_for_agg["chr"].astype(str)
    if isinstance(pooled_for_agg["strand"].dtype, pd.CategoricalDtype):
        pooled_for_agg["strand"] = pooled_for_agg["strand"].astype(str)

    agg_map: dict[str, str] = {}
    for col in pooled_for_agg.columns:
        if col in keys:
            continue
        if col in ("is_annot", "unique_mapped"):
            agg_map[col] = "max"
        else:
            agg_map[col] = "first"

    grouped = (
        pooled_for_agg.groupby(keys, as_index=False, sort=False, observed=True)
        .agg(agg_map)
        .reset_index(drop=True)
    )
    # Restore categorical dtypes that callers downstream rely on.
    grouped["chr"] = grouped["chr"].astype("category")
    grouped["strand"] = grouped["strand"].astype(
        pd.CategoricalDtype(categories=[".", "+", "-"])
    )
    # Reorder columns to match the original layout for downstream stability.
    grouped = grouped.loc[:, list(pooled.columns)]
    # Validate uniqueness on the natural key post-aggregation.
    if grouped.duplicated(subset=keys).any():
        raise RuntimeError(
            "Internal error: junction table not unique on (chr, start, end, "
            "strand) after aggregation — please file a bug with input details."
        )
    return grouped


def _concat_samples(
    artifacts: list[_SampleArtifacts],
    sample_ids: Sequence[str],
) -> tuple[sp.csc_matrix, pd.DataFrame, pd.DataFrame, dict | None, np.ndarray | None]:
    """Union junctions across samples (zero-pad) and stack cell axes.

    Returns ``(mtx_concat, var_pre, obs, spatial_uns, spatial_arr)`` where
    the last two are only non-None when at least one sample had a
    ``tissue_positions`` whitelist applied.
    """
    var_pre = _global_junction_table(artifacts)

    # (chr, start, end, strand) → global row index.
    key_to_idx: dict[tuple, int] = {
        (str(r.chr), int(r.start), int(r.end), str(r.strand)): i
        for i, r in enumerate(var_pre.itertuples(index=False))
    }
    n_global = len(var_pre)

    remapped_mtxs: list[sp.csc_matrix] = []
    obs_rows: list[pd.DataFrame] = []
    spatial_obs_rows: list[pd.DataFrame] = []
    spatial_uns: dict[str, dict] = {}
    for art, sid in zip(artifacts, sample_ids, strict=True):
        n_cells = art.mtx.shape[1]
        if len(art.features) == 0 or art.mtx.nnz == 0:
            empty = sp.csc_matrix((n_global, n_cells), dtype=np.float64)
            remapped_mtxs.append(empty)
        else:
            row_remap = np.asarray(
                [key_to_idx[(str(r.chr), int(r.start), int(r.end), str(r.strand))]
                 for r in art.features.itertuples(index=False)],
                dtype=np.int64,
            )
            coo = art.mtx.tocoo()
            new_rows = row_remap[coo.row]
            remapped = sp.csr_matrix(
                (coo.data.astype(np.float64), (new_rows, coo.col)),
                shape=(n_global, n_cells),
            ).tocsc()
            remapped_mtxs.append(remapped)
        obs_rows.append(pd.DataFrame({
            "barcode": art.barcodes,
            "sample_id": np.full(n_cells, sid, dtype=object),
        }))
        if art.spatial is not None:
            sp_df = art.spatial.reset_index(drop=False).copy()
            if "barcode" not in sp_df.columns and "index" in sp_df.columns:
                sp_df = sp_df.rename(columns={"index": "barcode"})
            sp_df["sample_id"] = sid
            spatial_obs_rows.append(sp_df)
            if art.library_id is not None:
                spatial_uns[art.library_id] = {
                    "metadata": {"source": "starsolo+tissue_positions"},
                }

    mtx_concat = sp.hstack(remapped_mtxs, format="csc").astype(np.float64)
    obs = pd.concat(obs_rows, axis=0, ignore_index=True)
    obs["sample_id"] = obs["sample_id"].astype("category")
    obs.index = pd.Index(
        [f"{bc}-{sid}" for bc, sid in zip(obs["barcode"], obs["sample_id"], strict=True)]
    )

    spatial_arr: np.ndarray | None = None
    if spatial_obs_rows:
        sp_concat = pd.concat(spatial_obs_rows, axis=0, ignore_index=True)
        sp_concat["barcode"] = sp_concat["barcode"].astype(str)
        sp_concat["sample_id"] = sp_concat["sample_id"].astype(str)
        obs_key = pd.DataFrame({
            "barcode": obs["barcode"].astype(str).to_numpy(),
            "sample_id": obs["sample_id"].astype(str).to_numpy(),
        })
        merged = obs_key.merge(sp_concat, on=["barcode", "sample_id"], how="left")
        obs["in_tissue"] = merged["in_tissue"].fillna(-1).astype(np.int8).to_numpy()
        obs["array_row"] = merged["array_row"].fillna(-1).astype(np.int32).to_numpy()
        obs["array_col"] = merged["array_col"].fillna(-1).astype(np.int32).to_numpy()
        spatial_arr = np.stack(
            [merged["pxl_row_in_fullres"].to_numpy(dtype=np.float64),
             merged["pxl_col_in_fullres"].to_numpy(dtype=np.float64)],
            axis=1,
        )
        spatial_arr = np.where(np.isnan(spatial_arr), -1.0, spatial_arr)

    return mtx_concat, var_pre, obs, (spatial_uns if spatial_uns else None), spatial_arr


def _apply_ljv_grouping(
    mtx_concat: sp.csc_matrix,
    var_pre: pd.DataFrame,
    ljv_kind: Literal["start_end", "start", "end"],
) -> tuple[sp.csc_matrix, pd.DataFrame]:
    """Expand junctions into ``_S`` / ``_E`` event rows with dense ``group_id``."""
    var_pre = var_pre.copy()
    var_pre["start_count"] = var_pre.groupby(
        ["chr", "start", "strand"], observed=False
    )["end"].transform("size").astype(np.int32)
    var_pre["end_count"] = var_pre.groupby(
        ["chr", "end", "strand"], observed=False
    )["start"].transform("size").astype(np.int32)

    do_S = ljv_kind in ("start_end", "start")
    do_E = ljv_kind in ("start_end", "end")
    mask_S = (var_pre["start_count"] > 1).to_numpy() if do_S else np.zeros(len(var_pre), dtype=bool)
    mask_E = (var_pre["end_count"] > 1).to_numpy() if do_E else np.zeros(len(var_pre), dtype=bool)

    if not (mask_S.any() or mask_E.any()):
        raise ValueError(
            f"No LJV events detected after grouping (ljv_kind={ljv_kind!r}). "
            "Every junction is a singleton in both its start- and end-coordinate "
            "groups. Check that SJ.out.tab contains alternative junctions."
        )

    row_idx_S = np.where(mask_S)[0]
    row_idx_E = np.where(mask_E)[0]
    row_remap_events = np.concatenate([row_idx_S, row_idx_E]).astype(np.int64)

    # Duplicate the global matrix rows for events; a junction qualifying for
    # both _S and _E lands at two distinct event positions.
    mtx_grouped = mtx_concat[row_remap_events, :].tocsc()

    # Build per-event metadata.
    pre_S = var_pre.iloc[row_idx_S].copy()
    pre_E = var_pre.iloc[row_idx_E].copy()

    chr_S = pre_S["chr"].astype(str).to_numpy()
    chr_E = pre_E["chr"].astype(str).to_numpy()
    start_S = pre_S["start"].to_numpy(dtype=np.int64)
    start_E = pre_E["start"].to_numpy(dtype=np.int64)
    end_S = pre_S["end"].to_numpy(dtype=np.int64)
    end_E = pre_E["end"].to_numpy(dtype=np.int64)

    row_names_S = np.array(
        [f"{c}:{s}-{e}" for c, s, e in zip(chr_S, start_S, end_S, strict=True)]
    )
    row_names_E = np.array(
        [f"{c}:{s}-{e}" for c, s, e in zip(chr_E, start_E, end_E, strict=True)]
    )

    var_S = pd.DataFrame({
        "chr": pd.Categorical(chr_S),
        "start": start_S,
        "end": end_S,
        "strand": pre_S["strand"].astype(str).to_numpy(),
        "intron_motif": pre_S["intron_motif"].to_numpy(),
        "is_annot": pre_S["is_annot"].to_numpy(),
        "unique_mapped": pre_S["unique_mapped"].to_numpy(),
        "row_names_mtx": row_names_S,
        "group_kind": "S",
        "group_count": pre_S["start_count"].astype(np.int32).to_numpy(),
    })
    var_E = pd.DataFrame({
        "chr": pd.Categorical(chr_E),
        "start": start_E,
        "end": end_E,
        "strand": pre_E["strand"].astype(str).to_numpy(),
        "intron_motif": pre_E["intron_motif"].to_numpy(),
        "is_annot": pre_E["is_annot"].to_numpy(),
        "unique_mapped": pre_E["unique_mapped"].to_numpy(),
        "row_names_mtx": row_names_E,
        "group_kind": "E",
        "group_count": pre_E["end_count"].astype(np.int32).to_numpy(),
    })

    parts = [df for df in (var_S, var_E) if len(df) > 0]
    var_grouped = pd.concat(parts, axis=0, ignore_index=True) if parts else var_S
    var_grouped["group_kind"] = pd.Categorical(
        var_grouped["group_kind"], categories=["S", "E"]
    )
    var_grouped["strand"] = pd.Categorical(
        var_grouped["strand"], categories=[".", "+", "-"]
    )

    # Dense int32 0..G-1 group_id over the disjoint S- and E-group key spaces.
    # Format the key as `S|chr|start|strand` for S-rows, `E|chr|end|strand` for E-rows
    # so factorize sees them as distinct namespaces.
    keys_S = (
        "S|" + chr_S
        + "|" + start_S.astype(str)
        + "|" + var_S["strand"].astype(str).to_numpy()
    )
    keys_E = (
        "E|" + chr_E
        + "|" + end_E.astype(str)
        + "|" + var_E["strand"].astype(str).to_numpy()
    )
    keys = np.concatenate([keys_S, keys_E])
    codes, _ = pd.factorize(keys, use_na_sentinel=False)
    var_grouped["group_id"] = codes.astype(np.int32)

    var_grouped.index = pd.Index(
        [f"{rn}_{kind}" for rn, kind in
         zip(var_grouped["row_names_mtx"], var_grouped["group_kind"], strict=True)]
    )

    return mtx_grouped, var_grouped


def _filter_min_counts(
    mtx_grouped: sp.csc_matrix,
    var_grouped: pd.DataFrame,
    min_counts: int,
) -> tuple[sp.csc_matrix, pd.DataFrame]:
    """Drop event rows whose total count across all cells is below ``min_counts``.

    Re-factorizes ``group_id`` after filtering so the dense ``0..G-1`` invariant
    documented for ``var["group_id"]`` (and required by the C++ kernel) holds
    on the returned AnnData. Dropping events can leave gaps in the group_id
    range when an entire LJV group's members all fail the filter.

    Also drops any LJV group that has fallen to size 1 after filtering — the
    LJV semantics require ``group_count >= 2``. We iterate to a fixed point
    because removing a singleton leaves no surviving members of its group, but
    in principle never cascades (we don't add events back).
    """
    if min_counts <= 0:
        return mtx_grouped, var_grouped
    row_sums = np.asarray(mtx_grouped.sum(axis=1)).ravel()
    keep = row_sums >= float(min_counts)
    if not keep.any():
        raise ValueError(
            f"min_counts={min_counts} removed all {len(var_grouped)} events; "
            f"row-sum range was [{row_sums.min():.1f}, {row_sums.max():.1f}]."
        )
    out_var = var_grouped.loc[keep].copy().reset_index(drop=False)
    # Preserve the original var_names index column for the final rebuild.
    if "index" in out_var.columns:
        out_var = out_var.rename(columns={"index": "_orig_index"})
    out_mtx = mtx_grouped[keep, :].tocsc()

    # Iterate: factorize → compute group_count → drop singletons → repeat
    # until no singletons remain. This is bounded; each pass shrinks the set.
    n_singleton_total = 0
    while True:
        new_codes, _ = pd.factorize(out_var["group_id"].to_numpy(), use_na_sentinel=False)
        out_var["group_id"] = new_codes.astype(np.int32)
        sizes = (
            out_var.groupby("group_id")["group_id"].transform("size").astype(np.int32)
        )
        out_var["group_count"] = sizes.to_numpy()
        survivors = (sizes >= 2).to_numpy()
        if survivors.all():
            break
        n_singleton_total += int((~survivors).sum())
        out_var = out_var.loc[survivors].reset_index(drop=True)
        out_mtx = out_mtx[survivors, :].tocsc()
        if len(out_var) == 0:
            # Emit the warning BEFORE raising so callers using pytest.warns
            # or warnings.catch_warnings can observe both signals.
            warnings.warn(
                f"min_counts={min_counts} dropped {n_singleton_total} event "
                "row(s) whose LJV group fell to size 1 after filtering "
                "(group_count must be >= 2).",
                stacklevel=3,
            )
            raise ValueError(
                f"min_counts={min_counts} left no LJV-valid events (every "
                "surviving event's group_count fell to 1). Lower min_counts "
                "or check the input."
            )

    if n_singleton_total:
        warnings.warn(
            f"min_counts={min_counts} dropped {n_singleton_total} event row(s) "
            "whose LJV group fell to size 1 after filtering (group_count must "
            "be >= 2).",
            stacklevel=3,
        )

    # Restore the original event-name index.
    if "_orig_index" in out_var.columns:
        out_var.index = pd.Index(out_var["_orig_index"].to_numpy())
        out_var = out_var.drop(columns=["_orig_index"])
    return out_mtx, out_var


def read_starsolo(
    sj_dirs: str | Path | Sequence[str | Path],
    sample_ids: str | Sequence[str],
    *,
    barcode_whitelists: Sequence[str | Path | Sequence[str] | None] | None = None,
    use_internal_whitelist: bool = True,
    keep_multi_mapped: bool = False,
    min_counts: int = 1,
    ljv_kind: Literal["start_end", "start", "end"] = "start_end",
    tissue_positions: Sequence[str | Path | None] | None = None,
    spatial_library_ids: Sequence[str | None] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Read STARsolo splice-junction output and assemble a splicing AnnData.

    Parameters
    ----------
    sj_dirs
        One or more ``Solo.out/SJ/`` (or ``Solo.out/SJ/raw/``) directories.
        A single string or :class:`pathlib.Path` is auto-promoted to a
        one-element sequence.
    sample_ids
        Per-directory unique sample identifier; appears in
        ``adata.obs["sample_id"]`` and as the ``-{sample_id}`` suffix on
        ``adata.obs_names``. Must have the same length as ``sj_dirs``.
        **Must not contain ``'-'``**: the ``{barcode}-{sample_id}``
        ``obs_names`` convention becomes unparseable if the sample_id has
        an embedded hyphen, so the reader rejects such inputs up-front.
    barcode_whitelists
        Optional per-sample whitelist of raw 16-mer barcodes. Each entry
        may be a path to a one-barcode-per-line file, a sequence of
        barcodes, or ``None`` to defer to ``use_internal_whitelist``.
    use_internal_whitelist
        When no explicit whitelist is supplied for a sample, fall back to
        STARsolo's per-sample whitelist at ``../Gene/filtered/barcodes.tsv``.
        If that file does not exist, a warning is emitted and no whitelist
        is applied.
    keep_multi_mapped
        Default ``False`` drops junctions whose ``unique_mapped`` flag in
        ``SJ.out.tab`` is zero (multi-mapped-only junctions). Set ``True``
        to keep all junctions.
    min_counts
        Minimum row-sum (across all cells of all samples) required for an
        event row to survive in the final AnnData. Applied **after** LJV
        grouping. Default ``1`` drops zero-count events.
    ljv_kind
        Which LJV-grouped event rows to emit: ``"start_end"`` (default;
        matches R splikit), ``"start"`` (only ``_S``), or ``"end"`` (only
        ``_E``). The single-half modes are advanced overrides for users who
        only care about alternative 5' or 3' splice sites.
    tissue_positions
        Per-sample optional path to a Space Ranger
        ``tissue_positions[_list].csv``. When provided for a sample, the
        file's barcodes act as the whitelist (raw/ source) and populate
        ``obs["in_tissue"]`` / ``obs["array_row"]`` / ``obs["array_col"]``,
        ``obsm["spatial"]``, and ``uns["spatial"][library_id]`` per the
        squidpy contract. Header (Space Ranger 2.x) and headerless
        ``tissue_positions_list.csv`` (v1) are auto-detected.
    spatial_library_ids
        Per-sample squidpy library_id key for ``uns["spatial"]``. Defaults
        to ``sample_ids[i]`` when not given.
    verbose
        Print per-sample progress to stdout.

    Notes
    -----
    Whitelist precedence per sample:

    1. ``tissue_positions[i]`` (when given)
    2. ``barcode_whitelists[i]`` (when given)
    3. ``use_internal_whitelist=True`` and ``Gene/filtered/barcodes.tsv`` exists
    4. otherwise no whitelist (use all raw barcodes)

    The SJ feature has only ``raw/`` available (no per-feature filtered
    dir), so the "read raw and intersect" step is the only mode regardless
    of which precedence rule fires; the rule still affects WHICH barcodes
    are kept.

    Returns
    -------
    AnnData
        Cells × events. ``layers["M1"]`` populated; ``layers["M2"]`` absent
        and ``uns["splikit"]["m2_valid"] is False`` — call
        :func:`splikit.tl.make_m2` next.
    """
    if isinstance(sj_dirs, (str, Path)):
        sj_dirs = [sj_dirs]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    sj_dirs = list(sj_dirs)
    sample_ids = list(sample_ids)
    if len(sj_dirs) != len(sample_ids):
        raise ValueError(
            f"len(sj_dirs)={len(sj_dirs)} must equal len(sample_ids)={len(sample_ids)}"
        )
    if len(sj_dirs) == 0:
        raise ValueError("At least one sj_dir / sample_id pair is required.")
    if len(set(sample_ids)) != len(sample_ids):
        dup = sorted({s for s, c in Counter(sample_ids).items() if c > 1})
        raise ValueError(f"sample_ids must be unique; duplicates: {dup}")
    bad_sid = [s for s in sample_ids if "-" in s]
    if bad_sid:
        raise ValueError(
            f"sample_ids must not contain '-' (the reader uses 'obs_names = "
            f"{{barcode}}-{{sample_id}}' so embedded hyphens make the suffix "
            f"ambiguous to parse). Offending sample_ids: {bad_sid}"
        )
    if min_counts < 0:
        raise ValueError(f"min_counts must be non-negative, got {min_counts}")
    if ljv_kind not in ("start_end", "start", "end"):
        raise ValueError(
            f"ljv_kind must be 'start_end', 'start', or 'end'; got {ljv_kind!r}"
        )

    n = len(sj_dirs)
    bcw = normalize_per_sample_arg(barcode_whitelists, n, name="barcode_whitelists")
    tp = normalize_per_sample_arg(tissue_positions, n, name="tissue_positions")
    libs = normalize_per_sample_arg(spatial_library_ids, n, name="spatial_library_ids")

    artifacts: list[_SampleArtifacts] = []
    for sj_dir, sid, bw, tps, lib in zip(
        sj_dirs, sample_ids, bcw, tp, libs, strict=True,
    ):
        artifacts.append(
            _read_one_sample(
                sj_dir, sid,
                keep_multi_mapped=keep_multi_mapped,
                barcode_whitelist=bw,
                tissue_positions=tps,
                use_internal_whitelist=use_internal_whitelist,
                spatial_library_id=lib,
                verbose=verbose,
            )
        )

    mtx_concat, var_pre, obs, spatial_uns, spatial_arr = _concat_samples(
        artifacts, sample_ids,
    )
    mtx_grouped, var_grouped = _apply_ljv_grouping(mtx_concat, var_pre, ljv_kind)
    mtx_grouped, var_grouped = _filter_min_counts(mtx_grouped, var_grouped, min_counts)

    # Transpose junctions × cells → cells × events for AnnData.
    M1 = mtx_grouped.T.tocsc().astype(np.float64)

    adata = ad.AnnData(
        layers={"M1": M1},
        obs=obs,
        var=var_grouped,
    )
    adata.uns["splikit"] = {
        "version": 1,
        "m2_valid": False,
        "ljv_kind": ljv_kind,
        "source": "starsolo",
        "params": {
            "read_starsolo": {
                "n_samples": len(sj_dirs),
                "keep_multi_mapped": bool(keep_multi_mapped),
                "min_counts": int(min_counts),
                "ljv_kind": ljv_kind,
                "use_internal_whitelist": bool(use_internal_whitelist),
                "any_explicit_whitelist": any(b is not None for b in bcw),
                "any_tissue_positions": any(t is not None for t in tp),
            }
        },
    }
    if spatial_arr is not None:
        adata.obsm["spatial"] = spatial_arr.astype(np.float64, copy=False)
    if spatial_uns is not None:
        adata.uns["spatial"] = spatial_uns
    invalidate_m2(adata)

    # Final sanity check: the schema validators must pass on the output.
    validate_m1_layer(adata)
    validate_var_schema(adata)

    if verbose:
        print(
            f"[splikit.io] Built AnnData: {adata.n_obs} cells × "
            f"{adata.n_vars} events ({len(sj_dirs)} samples, "
            f"M1 nnz={adata.layers['M1'].nnz})"
        )

    return adata
