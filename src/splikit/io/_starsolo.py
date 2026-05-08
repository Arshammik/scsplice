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

import warnings
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


__all__ = ["read_starsolo"]


# STARsolo encodes strand as 0 (undefined) / 1 (+) / 2 (-).
_STAR_STRAND_DECODE = {0: ".", 1: "+", 2: "-"}


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
    whitelist: Sequence[str] | None,
    use_internal_whitelist: bool,
    verbose: bool,
) -> _SampleArtifacts:
    """Read one STARsolo SJ sample, apply per-sample filters, decode strand."""
    paths = _resolve_sj_paths(sj_dir)

    if verbose:
        print(f"[splikit.io] Reading sample {sample_id!r} from {paths.matrix.parent}")

    # MTX is junctions × cells per STARsolo convention.
    mtx = sio.mmread(str(paths.matrix)).tocsc().astype(np.float64)
    n_jx_mtx, n_cells_mtx = mtx.shape

    barcodes = pd.read_csv(
        paths.barcodes, sep="\t", header=None, names=["barcode"],
        dtype=str, compression="infer",
    )["barcode"].to_numpy()

    sj_cols = ["chr", "start", "end", "strand",
               "intron_motif", "is_annot", "unique_mapped"]
    features = pd.read_csv(
        paths.sj_tab, sep="\t", header=None,
        usecols=range(len(sj_cols)), names=sj_cols,
        dtype={"chr": str, "start": "int64", "end": "int64",
               "strand": "int8", "intron_motif": "int8",
               "is_annot": "int8", "unique_mapped": "int32"},
        compression="infer",
    )

    if mtx.shape[0] != len(features) or mtx.shape[1] != len(barcodes):
        raise ValueError(
            f"STARsolo dimension mismatch in sample {sample_id!r}:\n"
            f"  matrix.mtx declares {n_jx_mtx} junctions x {n_cells_mtx} cells\n"
            f"  SJ.out.tab has    {len(features)} rows\n"
            f"  barcodes.tsv has  {len(barcodes)} lines\n"
            f"Junction or cell count disagrees. Verify SJ.out.tab and matrix.mtx "
            f"came from the same STARsolo run."
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

    # Barcode whitelist (raw 16-mers, before any sample_id suffix).
    wl_set: set[str] | None = None
    if whitelist is not None:
        wl_set = set(map(str, whitelist))
    elif use_internal_whitelist and paths.internal_whitelist.exists():
        wl_set = set(
            pd.read_csv(
                paths.internal_whitelist, sep="\t", header=None,
                names=["barcode"], dtype=str, compression="infer",
            )["barcode"].tolist()
        )
    elif use_internal_whitelist:
        warnings.warn(
            f"Sample {sample_id!r}: use_internal_whitelist=True but "
            f"{paths.internal_whitelist} not found; proceeding without whitelist.",
            stacklevel=3,
        )

    if wl_set is not None:
        keep_cells = np.array([bc in wl_set for bc in barcodes], dtype=bool)
        if not keep_cells.any():
            warnings.warn(
                f"Sample {sample_id!r}: whitelist intersection is empty; "
                "this sample will contribute zero cells to the output.",
                stacklevel=3,
            )
        mtx = mtx[:, keep_cells]
        barcodes = barcodes[keep_cells]

    return _SampleArtifacts(features=features, barcodes=barcodes,
                            mtx=mtx.tocsc())


def _global_junction_table(
    artifacts: list[_SampleArtifacts],
) -> pd.DataFrame:
    """Build the global, deduplicated junction table across all samples."""
    non_empty = [a.features for a in artifacts if len(a.features) > 0]
    if not non_empty:
        raise ValueError(
            "No junctions detected across any sample. Check that "
            "keep_multi_mapped wasn't filtering everything out and that "
            "SJ.out.tab is well-formed."
        )
    pooled = pd.concat(non_empty, axis=0, ignore_index=True)
    # Stable order: (chr, start, end, strand) lexicographic.
    pooled = pooled.drop_duplicates(
        subset=["chr", "start", "end", "strand"], keep="first"
    ).reset_index(drop=True)
    return pooled


def _concat_samples(
    artifacts: list[_SampleArtifacts],
    sample_ids: Sequence[str],
) -> tuple[sp.csc_matrix, pd.DataFrame, pd.DataFrame]:
    """Union junctions across samples (zero-pad) and stack cell axes."""
    var_pre = _global_junction_table(artifacts)

    # (chr, start, end, strand) → global row index.
    key_to_idx: dict[tuple, int] = {
        (str(r.chr), int(r.start), int(r.end), str(r.strand)): i
        for i, r in enumerate(var_pre.itertuples(index=False))
    }
    n_global = len(var_pre)

    remapped_mtxs: list[sp.csc_matrix] = []
    obs_rows: list[pd.DataFrame] = []
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

    mtx_concat = sp.hstack(remapped_mtxs, format="csc").astype(np.float64)
    obs = pd.concat(obs_rows, axis=0, ignore_index=True)
    obs["sample_id"] = obs["sample_id"].astype("category")
    obs.index = pd.Index(
        [f"{bc}-{sid}" for bc, sid in zip(obs["barcode"], obs["sample_id"], strict=True)]
    )
    return mtx_concat, var_pre, obs


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
    out_var = var_grouped.loc[keep].copy()
    # Re-densify group_id so codes are 0..G_kept-1 with no gaps.
    new_codes, _ = pd.factorize(out_var["group_id"].to_numpy(), use_na_sentinel=False)
    out_var["group_id"] = new_codes.astype(np.int32)
    # group_count is no longer accurate after dropping members; rebuild.
    out_var["group_count"] = out_var.groupby("group_id")["group_id"].transform(
        "size"
    ).astype(np.int32)
    return mtx_grouped[keep, :].tocsc(), out_var


def read_starsolo(
    sj_dirs: str | Path | Sequence[str | Path],
    sample_ids: str | Sequence[str],
    *,
    barcode_whitelists: Sequence[str | Path | Sequence[str] | None] | None = None,
    use_internal_whitelist: bool = True,
    keep_multi_mapped: bool = False,
    min_counts: int = 1,
    ljv_kind: Literal["start_end", "start", "end"] = "start_end",
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
    verbose
        Print per-sample progress to stdout.

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
        dup = [s for s in sample_ids if sample_ids.count(s) > 1]
        raise ValueError(f"sample_ids must be unique; duplicates: {sorted(set(dup))}")
    if min_counts < 0:
        raise ValueError(f"min_counts must be non-negative, got {min_counts}")
    if ljv_kind not in ("start_end", "start", "end"):
        raise ValueError(
            f"ljv_kind must be 'start_end', 'start', or 'end'; got {ljv_kind!r}"
        )

    if barcode_whitelists is None:
        whitelists_per_sample: list[Sequence[str] | None] = [None] * len(sj_dirs)
    else:
        if len(barcode_whitelists) != len(sj_dirs):
            raise ValueError(
                f"len(barcode_whitelists)={len(barcode_whitelists)} must equal "
                f"len(sj_dirs)={len(sj_dirs)}"
            )
        whitelists_per_sample = []
        for wl in barcode_whitelists:
            if wl is None:
                whitelists_per_sample.append(None)
            elif isinstance(wl, (str, Path)):
                wl_arr = pd.read_csv(
                    wl, sep="\t", header=None, names=["barcode"],
                    dtype=str, compression="infer",
                )["barcode"].tolist()
                whitelists_per_sample.append(wl_arr)
            else:
                whitelists_per_sample.append(list(map(str, wl)))

    artifacts: list[_SampleArtifacts] = []
    for sj_dir, sid, wl in zip(sj_dirs, sample_ids, whitelists_per_sample, strict=True):
        artifacts.append(
            _read_one_sample(
                sj_dir, sid,
                keep_multi_mapped=keep_multi_mapped,
                whitelist=wl,
                use_internal_whitelist=use_internal_whitelist and wl is None,
                verbose=verbose,
            )
        )

    mtx_concat, var_pre, obs = _concat_samples(artifacts, sample_ids)
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
            }
        },
    }
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
