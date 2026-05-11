"""STARsolo Gene-feature ingestion → cell × gene AnnData.

Reads ``Solo.out/Gene/{raw,filtered}/{barcodes.tsv, features.tsv, matrix.mtx}``
for one or more samples. The reader applies the consistent whitelist
precedence implemented in :mod:`splikit.io._whitelist`:

    tissue_positions  >  explicit barcode_whitelist  >  internal filtered/  >  raw

When a custom whitelist is supplied (explicit or spatial), counts are read
from ``Gene/raw/`` so the reader can intersect with barcodes that may not
be in STARsolo's filtered set; otherwise the much-smaller ``filtered/`` is
used directly.

Multi-sample inputs are concatenated on the cell axis with
``obs_names = "<barcode>-<sample_id>"`` and the gene axis is unioned
(zero-padded) so samples whose ``features.tsv`` differ still produce a
clean AnnData.

Output schema is documented in
``docs/explanation/io-readers-and-data-layouts.md`` and validated at the
end of :func:`read_starsolo_gene`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from splikit.io._starsolo import _safe_mmread  # shared mmread guard
from splikit.io._whitelist import (
    ResolvedWhitelist,
    load_tissue_positions,  # re-exported for callers that want raw access
    normalize_per_sample_arg,
    resolve_whitelist,
)


__all__ = ["read_starsolo_gene"]


@dataclass(frozen=True)
class _GenePaths:
    matrix: Path
    barcodes: Path
    features: Path
    source_dir: Path  # raw/ or filtered/
    internal_whitelist: Path  # may not exist
    used_raw: bool


@dataclass(frozen=True)
class _GeneSampleArtifacts:
    features: pd.DataFrame  # gene_id, gene_name, feature_type
    barcodes: np.ndarray  # cells_kept
    mtx: sp.csc_matrix  # cells_kept × genes (already transposed)
    spatial: pd.DataFrame | None  # rows aligned to barcodes
    library_id: str | None
    whitelist_source: str


def _resolve_gene_paths(sample_dir: str | Path, *, prefer_raw: bool) -> _GenePaths:
    """Locate the four input files for one sample.

    Accepts:
      - ``<sample_root>`` (containing ``Solo.out/``)
      - ``<sample_root>/Solo.out/``
      - ``<sample_root>/Solo.out/Gene/``
      - ``<sample_root>/Solo.out/Gene/raw/`` or ``.../filtered/``

    ``prefer_raw=True`` forces use of ``raw/`` (when an external whitelist /
    tissue_positions is provided); otherwise prefers ``filtered/`` and falls
    back to ``raw/``.
    """
    p = Path(sample_dir).expanduser().resolve()

    # Walk down to the Gene dir.
    if (p / "Solo.out" / "Gene").is_dir():
        gene_dir = p / "Solo.out" / "Gene"
    elif p.name == "Solo.out" or (p / "Gene").is_dir():
        gene_dir = (p / "Gene") if p.name == "Solo.out" else p.parent / "Gene"
    elif p.name == "Gene":
        gene_dir = p
    elif (p / "raw").is_dir() or (p / "filtered").is_dir() or _has_mtx(p):
        # User pointed at raw/, filtered/, or directly at the dir holding mtx.
        gene_dir = p.parent if _has_mtx(p) else p
    else:
        raise FileNotFoundError(
            f"Cannot locate Solo.out/Gene under {sample_dir}. "
            f"Tried: {p / 'Solo.out' / 'Gene'}."
        )

    raw_dir = gene_dir / "raw"
    filt_dir = gene_dir / "filtered"

    # If user pointed directly at raw/ or filtered/.
    if _has_mtx(p) and p.name in ("raw", "filtered"):
        chosen = p
        used_raw = (p.name == "raw")
    else:
        if prefer_raw:
            if not _has_mtx(raw_dir):
                raise FileNotFoundError(
                    f"prefer_raw=True but {raw_dir}/matrix.mtx[.gz] not found."
                )
            chosen = raw_dir
            used_raw = True
        else:
            if _has_mtx(filt_dir):
                chosen = filt_dir
                used_raw = False
            elif _has_mtx(raw_dir):
                chosen = raw_dir
                used_raw = True
            else:
                raise FileNotFoundError(
                    f"Neither {filt_dir}/matrix.mtx[.gz] nor "
                    f"{raw_dir}/matrix.mtx[.gz] found."
                )

    matrix = chosen / "matrix.mtx" if (chosen / "matrix.mtx").exists() else chosen / "matrix.mtx.gz"
    barcodes = chosen / "barcodes.tsv" if (chosen / "barcodes.tsv").exists() else chosen / "barcodes.tsv.gz"
    features = chosen / "features.tsv" if (chosen / "features.tsv").exists() else chosen / "features.tsv.gz"
    internal_wl = filt_dir / "barcodes.tsv"
    if not internal_wl.exists() and (filt_dir / "barcodes.tsv.gz").exists():
        internal_wl = filt_dir / "barcodes.tsv.gz"
    return _GenePaths(
        matrix=matrix, barcodes=barcodes, features=features,
        source_dir=chosen, internal_whitelist=internal_wl, used_raw=used_raw,
    )


def _has_mtx(d: Path) -> bool:
    return (d / "matrix.mtx").exists() or (d / "matrix.mtx.gz").exists()


def _read_features(path: Path) -> pd.DataFrame:
    """Read 10x/STARsolo features.tsv. Handles 2-col (v2) and 3-col (v3) files.

    Returns columns ``gene_id``, ``gene_name``, ``feature_type`` (filled with
    ``"Gene Expression"`` for 2-col v2 files, which predate the multi-modal
    feature_type column).
    """
    # Sniff column count.
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, compression="infer")
    if df.shape[1] == 2:
        df.columns = ["gene_id", "gene_name"]
        df["feature_type"] = "Gene Expression"
    elif df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ["gene_id", "gene_name", "feature_type"]
    else:
        raise ValueError(
            f"features.tsv at {path} has {df.shape[1]} columns; "
            "expected 2 (Cell Ranger v2) or 3 (v3 / STARsolo)."
        )
    return df


def _read_one_gene_sample(
    sample_dir: str | Path,
    sample_id: str,
    *,
    barcode_whitelist,
    tissue_positions,
    use_internal_whitelist: bool,
    spatial_library_id: str | None,
    verbose: bool,
) -> _GeneSampleArtifacts:
    """Read one STARsolo Gene sample with the resolved whitelist applied."""
    # Decide whether we need raw/ before resolving paths (we need the
    # internal-whitelist path either way).
    prefer_raw = (tissue_positions is not None) or (barcode_whitelist is not None)
    paths = _resolve_gene_paths(sample_dir, prefer_raw=prefer_raw)

    if verbose:
        print(
            f"[splikit.io] Reading Gene sample {sample_id!r} from "
            f"{paths.source_dir} (used_raw={paths.used_raw})"
        )

    resolved: ResolvedWhitelist = resolve_whitelist(
        sample_id=sample_id,
        barcode_whitelist=barcode_whitelist,
        tissue_positions=tissue_positions,
        use_internal_whitelist=use_internal_whitelist,
        internal_whitelist_path=paths.internal_whitelist,
        spatial_library_id=spatial_library_id,
    )

    # If user pointed at filtered/ but tissue_positions/explicit whitelist
    # was provided, we silently re-resolve to raw/. (handled above by
    # prefer_raw=True; this is an extra safety net for in-memory whitelists
    # whose intersection might exceed filtered/'s coverage.)

    # MTX is genes × cells in 10x/STARsolo convention; AnnData wants cells × genes.
    mtx = _safe_mmread(paths.matrix, what="Gene matrix.mtx").tocsc().astype(np.float64)
    n_genes_mtx, n_cells_mtx = mtx.shape

    barcodes = pd.read_csv(
        paths.barcodes, sep="\t", header=None, names=["barcode"],
        dtype=str, compression="infer",
    )["barcode"].to_numpy()
    features = _read_features(paths.features)

    if mtx.shape[0] != len(features) or mtx.shape[1] != len(barcodes):
        raise ValueError(
            f"STARsolo dimension mismatch in Gene sample {sample_id!r}:\n"
            f"  matrix.mtx declares {n_genes_mtx} genes x {n_cells_mtx} cells\n"
            f"  features.tsv has {len(features)} rows\n"
            f"  barcodes.tsv has {len(barcodes)} lines"
        )

    # Apply whitelist.
    if resolved.barcodes is not None:
        wl_set = resolved.barcodes
        # ``pd.Index(...).isin(set)`` is ~50× faster than a Python comprehension
        # on 6.7M barcodes and dtype-safe across str / numpy.str_ / arrow.
        keep = np.asarray(pd.Index(barcodes).isin(wl_set), dtype=bool)
        if not keep.any():
            import warnings
            warnings.warn(
                f"Gene sample {sample_id!r}: whitelist intersection is empty; "
                "this sample contributes 0 cells.",
                stacklevel=4,
            )
        mtx = mtx[:, keep]
        barcodes = barcodes[keep]

    # Transpose to cells × genes.
    mtx_cell_gene = mtx.T.tocsc().astype(np.float64)

    # Align spatial DataFrame to the kept barcode order, if present.
    spatial_aligned: pd.DataFrame | None = None
    if resolved.spatial is not None:
        spatial_aligned = resolved.spatial.reindex(barcodes)
        if spatial_aligned.isna().any().any():
            missing = int(spatial_aligned["in_tissue"].isna().sum())
            raise ValueError(
                f"Gene sample {sample_id!r}: {missing} kept barcodes are not "
                "present in tissue_positions; verify the tissue_positions file "
                "is from the same library as the MTX."
            )

    return _GeneSampleArtifacts(
        features=features,
        barcodes=barcodes,
        mtx=mtx_cell_gene,
        spatial=spatial_aligned,
        library_id=resolved.library_id,
        whitelist_source=resolved.source,
    )


def _concat_gene_samples(
    artifacts: list[_GeneSampleArtifacts],
    sample_ids: Sequence[str],
    *,
    var_names: Literal["gene_ids", "gene_symbols"],
) -> tuple[sp.csc_matrix, pd.DataFrame, pd.DataFrame, dict | None, np.ndarray | None]:
    """Union genes (zero-pad) and stack cell axis.

    Returns ``(X, var, obs, spatial_uns, spatial_arr)``. ``spatial_arr`` is
    returned explicitly rather than smuggled through ``obs.attrs`` (which
    is experimental and not preserved across many pandas operations).
    """
    # Build the global gene index. Stable order: first appearance.
    # Use gene_id as the join key (always unique by design).
    seen: dict[str, dict] = {}
    for art in artifacts:
        for row in art.features.itertuples(index=False):
            gid = str(row.gene_id)
            if gid not in seen:
                seen[gid] = {
                    "gene_id": gid,
                    "gene_name": str(row.gene_name),
                    "feature_type": str(row.feature_type),
                }
    var_global = pd.DataFrame(list(seen.values()))
    gid_to_global = {gid: i for i, gid in enumerate(var_global["gene_id"])}
    n_global = len(var_global)

    # Build remapped per-sample CSC matrices, then hstack on the cell axis
    # — but our matrices are cells × genes, so we need vstack (axis=0) AFTER
    # remapping COLUMNS to the global gene order.
    remapped: list[sp.csc_matrix] = []
    obs_rows: list[pd.DataFrame] = []
    spatial_obs_rows: list[pd.DataFrame] = []
    spatial_uns: dict[str, dict] = {}

    for art, sid in zip(artifacts, sample_ids, strict=True):
        n_cells = art.mtx.shape[0]
        if n_cells == 0:
            empty = sp.csc_matrix((0, n_global), dtype=np.float64)
            remapped.append(empty)
        else:
            local_to_global = np.fromiter(
                (gid_to_global[str(g)] for g in art.features["gene_id"]),
                dtype=np.int64, count=len(art.features),
            )
            # mtx is cells × local_genes; need cells × n_global.
            coo = art.mtx.tocoo()
            new_cols = local_to_global[coo.col]
            remapped.append(
                sp.csr_matrix(
                    (coo.data.astype(np.float64), (coo.row, new_cols)),
                    shape=(n_cells, n_global),
                ).tocsc()
            )
        obs_rows.append(pd.DataFrame({
            "barcode": art.barcodes,
            "sample_id": np.full(n_cells, sid, dtype=object),
        }))
        if art.spatial is not None:
            sp_df = art.spatial.reset_index(drop=False).copy()
            # ``art.spatial`` was reindexed to ``art.barcodes`` so the index
            # name is "barcode"; reset_index promotes it back to a column.
            if "barcode" not in sp_df.columns and "index" in sp_df.columns:
                sp_df = sp_df.rename(columns={"index": "barcode"})
            sp_df["sample_id"] = sid
            spatial_obs_rows.append(sp_df)
            if art.library_id is not None:
                spatial_uns[art.library_id] = {
                    "metadata": {"source": "starsolo+tissue_positions"},
                }

    # vstack in CSR (cheap) then convert once to CSC; vstacking CSC directly
    # peaks at multi-GB on real data because CSC isn't the natural concat fmt.
    if remapped:
        X = sp.vstack(remapped, format="csr").tocsc().astype(np.float64)
    else:
        X = sp.csc_matrix((0, n_global), dtype=np.float64)
    obs = pd.concat(obs_rows, axis=0, ignore_index=True) if obs_rows else \
        pd.DataFrame({"barcode": [], "sample_id": []})
    obs["sample_id"] = obs["sample_id"].astype("category")
    obs.index = pd.Index(
        [f"{bc}-{sid}" for bc, sid in zip(obs["barcode"], obs["sample_id"], strict=True)]
    )

    # Var: choose var_names per option.
    var = var_global.copy()
    var = var.rename(columns={"gene_id": "gene_id", "gene_name": "gene_name"})
    if var_names == "gene_ids":
        var.index = pd.Index(var["gene_id"].astype(str))
    elif var_names == "gene_symbols":
        var.index = pd.Index(var["gene_name"].astype(str))
        # Symbols can collide; legal here (no LJV suffix scheme).
        # We let the AnnData object do `var_names_make_unique()` after build.
    else:
        raise ValueError(
            f"var_names must be 'gene_ids' or 'gene_symbols', got {var_names!r}"
        )

    # Spatial obs columns (stacked across all samples that had spatial).
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
        # Cells from non-spatial samples land as NaN; replace with -1.0 sentinel
        # so the array stays float64-clean. Squidpy users filter in_tissue==1.
        spatial_arr = np.where(np.isnan(spatial_arr), -1.0, spatial_arr)
    else:
        spatial_arr = None

    spatial_uns_or_none = spatial_uns if spatial_uns else None
    # Return spatial_arr explicitly rather than stashing it on obs.attrs
    # (pandas .attrs is experimental and not preserved through most ops).
    return X, var, obs, spatial_uns_or_none, spatial_arr


def read_starsolo_gene(
    sample_dirs: str | Path | Sequence[str | Path],
    sample_ids: str | Sequence[str],
    *,
    barcode_whitelists: Sequence[str | Path | Sequence[str] | None] | None = None,
    use_internal_whitelist: bool = True,
    var_names: Literal["gene_ids", "gene_symbols"] = "gene_ids",
    tissue_positions: Sequence[str | Path | None] | None = None,
    spatial_library_ids: Sequence[str | None] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Read STARsolo Gene-feature output and assemble a cell × gene AnnData.

    Parameters
    ----------
    sample_dirs
        One or more sample directories. Each may be the sample root (parent
        of ``Solo.out/``), the ``Solo.out/`` dir, ``Solo.out/Gene/``, or
        ``Solo.out/Gene/raw/`` / ``.../filtered/`` directly.
    sample_ids
        Per-sample unique identifier; appears in ``obs["sample_id"]`` and
        as the ``-{sample_id}`` suffix on ``obs_names``. **Must not contain
        ``'-'``**: the ``{barcode}-{sample_id}`` ``obs_names`` convention
        becomes unparseable if the sample_id has an embedded hyphen.
    barcode_whitelists
        Per-sample whitelist of barcodes. ``None`` to fall back to internal
        whitelist; otherwise a path or sequence. When provided, counts are
        read from ``Gene/raw/`` (so barcodes outside ``filtered/`` are
        captured) and intersected with the whitelist.
    use_internal_whitelist
        When neither an external whitelist nor ``tissue_positions`` is
        given, fall back to ``Solo.out/Gene/filtered/barcodes.tsv``. If
        that file is missing, a warning is emitted.
    var_names
        ``"gene_ids"`` (default) uses Ensembl/NCBI IDs (always unique);
        ``"gene_symbols"`` uses gene names. With symbols, the reader
        calls ``var_names_make_unique()`` because symbols collide for
        paralogs (e.g. ``IGH@``) — this is *only* legal here, never on
        the splicing reader's ``_S/_E`` axis.
    tissue_positions
        Per-sample optional path to a Space Ranger
        ``tissue_positions[_list].csv``. When provided, the file's
        barcodes act as the whitelist (raw/ source) AND populate
        ``obs["in_tissue"]``, ``obs["array_row"]``, ``obs["array_col"]``,
        ``obsm["spatial"]``, and ``uns["spatial"][library_id]`` per the
        squidpy contract. Header (Space Ranger 2.x) and headerless v1
        ``tissue_positions_list.csv`` are auto-detected.
    spatial_library_ids
        Per-sample squidpy library_id used as the key in
        ``uns["spatial"]``. Defaults to ``sample_ids[i]`` when not given.
    verbose
        Print per-sample diagnostics to stdout.

    Returns
    -------
    AnnData
        Cells × genes. ``X`` is sparse CSC float64 (raw counts). When any
        sample provides ``tissue_positions``, the squidpy-shaped fields
        are populated; cells from non-spatial samples have ``in_tissue=-1``
        and ``obsm["spatial"][i] = (-1.0, -1.0)`` as sentinel.

    Notes
    -----
    Whitelist precedence per sample:

    1. ``tissue_positions[i]`` (read raw/, intersect)
    2. ``barcode_whitelists[i]`` (read raw/, intersect)
    3. ``use_internal_whitelist=True`` and filtered/ exists (read filtered/)
    4. otherwise raw/ unfiltered

    The reader does not run ``min_counts`` filtering — gene expression
    workflows have their own QC (e.g. ``scanpy.pp.calculate_qc_metrics``).

    Validated against STARsolo 2.7.10+ and Cell Ranger 6/7/8 features.tsv
    (3-col); 2-col v2 features.tsv is supported via the column-count
    dispatch in ``_read_features``.
    """
    if isinstance(sample_dirs, (str, Path)):
        sample_dirs = [sample_dirs]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    sample_dirs = list(sample_dirs)
    sample_ids = list(sample_ids)

    if len(sample_dirs) != len(sample_ids):
        raise ValueError(
            f"len(sample_dirs)={len(sample_dirs)} must equal "
            f"len(sample_ids)={len(sample_ids)}"
        )
    if len(sample_dirs) == 0:
        raise ValueError("At least one sample_dir / sample_id pair is required.")
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

    n = len(sample_dirs)
    bcw = normalize_per_sample_arg(barcode_whitelists, n, name="barcode_whitelists")
    tp = normalize_per_sample_arg(tissue_positions, n, name="tissue_positions")
    libs = normalize_per_sample_arg(spatial_library_ids, n, name="spatial_library_ids")

    artifacts: list[_GeneSampleArtifacts] = []
    for sd, sid, bw, tps, lib in zip(sample_dirs, sample_ids, bcw, tp, libs, strict=True):
        artifacts.append(
            _read_one_gene_sample(
                sd, sid,
                barcode_whitelist=bw,
                tissue_positions=tps,
                use_internal_whitelist=use_internal_whitelist,
                spatial_library_id=lib,
                verbose=verbose,
            )
        )

    X, var, obs, spatial_uns, spatial_arr = _concat_gene_samples(
        artifacts, sample_ids, var_names=var_names,
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["splikit"] = {
        "version": 1,
        "source": "starsolo",
        "params": {
            "read_starsolo_gene": {
                "n_samples": n,
                "var_names": var_names,
                "use_internal_whitelist": bool(use_internal_whitelist),
                "any_explicit_whitelist": any(b is not None for b in bcw),
                "any_tissue_positions": any(t is not None for t in tp),
            }
        },
    }
    if var_names == "gene_symbols":
        # Legal here; gene symbols can collide and there is no LJV suffix scheme.
        adata.var_names_make_unique()
    else:
        if not adata.var_names.is_unique:
            raise RuntimeError(
                "var_names not unique with var_names='gene_ids' — duplicate "
                "gene_id seen across samples; this should be impossible."
            )

    if spatial_arr is not None:
        adata.obsm["spatial"] = spatial_arr.astype(np.float64, copy=False)
    if spatial_uns is not None:
        adata.uns["spatial"] = spatial_uns

    if verbose:
        print(
            f"[splikit.io] Built Gene AnnData: {adata.n_obs} cells × "
            f"{adata.n_vars} genes ({n} samples, X nnz={adata.X.nnz})"
        )
    return adata
