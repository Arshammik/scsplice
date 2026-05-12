"""STARsolo Velocyto-feature ingestion → cell × gene AnnData with three layers.

``Solo.out/Velocyto/raw/`` ships RNA-velocity input. Two on-disk formats
are seen in the wild and both are handled here:

- **Modern (split files; STARsolo 2.7.10b+):** three sibling matrices
  ``spliced.mtx``, ``unspliced.mtx``, ``ambiguous.mtx`` (each cells × genes
  per the standard STARsolo wire layout, i.e. genes-as-rows in the file).
- **Legacy (stacked):** a single ``matrix.mtx`` with ``3 * n_genes`` rows;
  rows ``[0, n_genes)`` are spliced, ``[n_genes, 2*n_genes)`` unspliced,
  ``[2*n_genes, 3*n_genes)`` ambiguous.

Detection is purely structural: presence of ``spliced.mtx`` chooses split.

Output schema is documented in
``docs/explanation/io-readers-and-data-layouts.md``. ``X`` is set to a
copy of ``layers["spliced"]`` so scvelo helpers that only know about
``X`` work without modification.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scsplice.io._starsolo import _safe_mmread  # shared mmread guard
from scsplice.io._starsolo_gene import _read_features
from scsplice.io._whitelist import (
    ResolvedWhitelist,
    normalize_per_sample_arg,
    resolve_whitelist,
)


__all__ = ["read_starsolo_velocyto"]


@dataclass(frozen=True)
class _VelPaths:
    barcodes: Path
    features: Path
    # Modern layout
    spliced: Path | None
    unspliced: Path | None
    ambiguous: Path | None
    # Legacy layout
    stacked_matrix: Path | None
    source_dir: Path
    internal_whitelist: Path  # may not exist


@dataclass(frozen=True)
class _VelSampleArtifacts:
    features: pd.DataFrame
    barcodes: np.ndarray
    spliced: sp.csc_matrix      # cells × genes
    unspliced: sp.csc_matrix
    ambiguous: sp.csc_matrix
    spatial: pd.DataFrame | None
    library_id: str | None
    whitelist_source: str
    wire_format: Literal["split", "stacked"]


def _resolve_velocyto_paths(sample_dir: str | Path) -> _VelPaths:
    """Locate raw/ Velocyto files and decide split vs stacked."""
    p = Path(sample_dir).expanduser().resolve()
    # Walk to Velocyto/raw/.
    if (p / "Solo.out" / "Velocyto" / "raw").is_dir():
        raw = p / "Solo.out" / "Velocyto" / "raw"
    elif p.name == "Solo.out" and (p / "Velocyto" / "raw").is_dir():
        raw = p / "Velocyto" / "raw"
    elif p.name == "Velocyto" and (p / "raw").is_dir():
        raw = p / "raw"
    elif p.name == "raw" and (p / "barcodes.tsv").exists():
        raw = p
    else:
        raise FileNotFoundError(
            f"Cannot locate Solo.out/Velocyto/raw/ under {sample_dir}."
        )

    bc = raw / "barcodes.tsv" if (raw / "barcodes.tsv").exists() else raw / "barcodes.tsv.gz"
    feat = raw / "features.tsv" if (raw / "features.tsv").exists() else raw / "features.tsv.gz"

    has_split = any((raw / f"{m}.mtx").exists() or (raw / f"{m}.mtx.gz").exists()
                    for m in ("spliced", "unspliced", "ambiguous"))
    has_stacked = (raw / "matrix.mtx").exists() or (raw / "matrix.mtx.gz").exists()

    # Internal whitelist sits next door at Gene/filtered/barcodes.tsv. Velocyto
    # itself only ships raw/; the filtered whitelist is the Gene/filtered/ set.
    # Note: ``raw.parent`` is Velocyto/, so we need to step up to Solo.out/.
    solo_out = raw.parent.parent
    internal_wl = solo_out / "Gene" / "filtered" / "barcodes.tsv"
    if not internal_wl.exists() and (solo_out / "Gene" / "filtered" / "barcodes.tsv.gz").exists():
        internal_wl = solo_out / "Gene" / "filtered" / "barcodes.tsv.gz"

    if has_split:
        def _pick(name: str) -> Path:
            p1 = raw / f"{name}.mtx"
            p2 = raw / f"{name}.mtx.gz"
            if p1.exists():
                return p1
            if p2.exists():
                return p2
            raise FileNotFoundError(
                f"Velocyto split layout missing {name}.mtx[.gz] in {raw}"
            )
        return _VelPaths(
            barcodes=bc, features=feat,
            spliced=_pick("spliced"), unspliced=_pick("unspliced"),
            ambiguous=_pick("ambiguous"),
            stacked_matrix=None, source_dir=raw, internal_whitelist=internal_wl,
        )
    if has_stacked:
        m = raw / "matrix.mtx" if (raw / "matrix.mtx").exists() else raw / "matrix.mtx.gz"
        return _VelPaths(
            barcodes=bc, features=feat,
            spliced=None, unspliced=None, ambiguous=None,
            stacked_matrix=m, source_dir=raw, internal_whitelist=internal_wl,
        )
    raise FileNotFoundError(
        f"Velocyto raw dir {raw} contains neither split (spliced/unspliced/"
        "ambiguous) nor stacked (matrix.mtx) layouts."
    )


def _load_split_layer(path: Path, expected_shape: tuple[int, int]) -> sp.csc_matrix:
    m = _safe_mmread(path, what=f"Velocyto {path.name}").tocsc().astype(np.float64)
    if m.shape != expected_shape:
        raise ValueError(
            f"Velocyto layer {path.name} shape {m.shape} != expected "
            f"{expected_shape} (genes × cells from features.tsv / barcodes.tsv)"
        )
    return m


def _read_one_velocyto_sample(
    sample_dir: str | Path,
    sample_id: str,
    *,
    barcode_whitelist,
    tissue_positions,
    use_internal_whitelist: bool,
    spatial_library_id: str | None,
    verbose: bool,
) -> _VelSampleArtifacts:
    paths = _resolve_velocyto_paths(sample_dir)
    if verbose:
        wf = "split" if paths.spliced is not None else "stacked"
        print(
            f"[scsplice.io] Reading Velocyto sample {sample_id!r} from "
            f"{paths.source_dir} (wire_format={wf})"
        )

    barcodes = pd.read_csv(
        paths.barcodes, sep="\t", header=None, names=["barcode"],
        dtype=str, compression="infer",
    )["barcode"].to_numpy()
    features = _read_features(paths.features)

    n_cells = len(barcodes)
    n_genes = len(features)

    # Read all three layers in genes × cells layout.
    if paths.spliced is not None:
        wire_format: Literal["split", "stacked"] = "split"
        spliced = _load_split_layer(paths.spliced, (n_genes, n_cells))
        unspliced = _load_split_layer(paths.unspliced, (n_genes, n_cells))
        ambiguous = _load_split_layer(paths.ambiguous, (n_genes, n_cells))
    else:
        wire_format = "stacked"
        stacked = _safe_mmread(
            paths.stacked_matrix, what="Velocyto stacked matrix.mtx"
        ).astype(np.float64)
        if stacked.shape[0] != 3 * n_genes:
            raise ValueError(
                f"Velocyto stacked matrix.mtx in sample {sample_id!r} has "
                f"{stacked.shape[0]} rows; expected 3 * n_genes = "
                f"{3 * n_genes}."
            )
        if stacked.shape[1] != n_cells:
            raise ValueError(
                f"Velocyto stacked matrix.mtx col count {stacked.shape[1]} "
                f"!= barcodes.tsv length {n_cells}"
            )
        # Row-slicing a CSC is worst-case; convert to CSR once, slice three
        # contiguous row blocks, then convert each layer back to CSC.
        stacked_csr = stacked.tocsr()
        spliced = stacked_csr[:n_genes, :].tocsc()
        unspliced = stacked_csr[n_genes:2 * n_genes, :].tocsc()
        ambiguous = stacked_csr[2 * n_genes:3 * n_genes, :].tocsc()

    # Resolve whitelist.
    resolved: ResolvedWhitelist = resolve_whitelist(
        sample_id=sample_id,
        barcode_whitelist=barcode_whitelist,
        tissue_positions=tissue_positions,
        use_internal_whitelist=use_internal_whitelist,
        internal_whitelist_path=paths.internal_whitelist,
        spatial_library_id=spatial_library_id,
    )

    if resolved.barcodes is not None:
        wl_set = resolved.barcodes
        # ``pd.Index(...).isin(set)`` is ~50× faster than a Python comprehension
        # on 6.7M barcodes and dtype-safe across str / numpy.str_ / arrow.
        keep = np.asarray(pd.Index(barcodes).isin(wl_set), dtype=bool)
        if not keep.any():
            warnings.warn(
                f"Velocyto sample {sample_id!r}: whitelist intersection is "
                "empty; this sample contributes 0 cells.",
                stacklevel=4,
            )
        spliced = spliced[:, keep]
        unspliced = unspliced[:, keep]
        ambiguous = ambiguous[:, keep]
        barcodes = barcodes[keep]

    # Transpose each to cells × genes.
    spliced = spliced.T.tocsc().astype(np.float64)
    unspliced = unspliced.T.tocsc().astype(np.float64)
    ambiguous = ambiguous.T.tocsc().astype(np.float64)

    # Align spatial.
    spatial_aligned = None
    if resolved.spatial is not None:
        spatial_aligned = resolved.spatial.reindex(barcodes)
        if spatial_aligned.isna().any().any():
            missing = int(spatial_aligned["in_tissue"].isna().sum())
            raise ValueError(
                f"Velocyto sample {sample_id!r}: {missing} kept barcodes are "
                "not present in tissue_positions; verify the tissue_positions "
                "file is from the same library as the MTX."
            )

    return _VelSampleArtifacts(
        features=features, barcodes=barcodes,
        spliced=spliced, unspliced=unspliced, ambiguous=ambiguous,
        spatial=spatial_aligned, library_id=resolved.library_id,
        whitelist_source=resolved.source, wire_format=wire_format,
    )


def _concat_velocyto_samples(
    artifacts: list[_VelSampleArtifacts],
    sample_ids: Sequence[str],
    *,
    var_names: Literal["gene_ids", "gene_symbols"],
) -> tuple[
    sp.csc_matrix, sp.csc_matrix, sp.csc_matrix,
    pd.DataFrame, pd.DataFrame, dict | None, np.ndarray | None,
]:
    """Union genes (zero-pad), stack cells. Mirrors the gene-reader concat.

    Returns the three layer matrices, ``var``, ``obs``, ``spatial_uns``,
    and ``spatial_arr`` (explicit; not smuggled via ``obs.attrs``).
    """
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

    s_blocks, u_blocks, a_blocks = [], [], []
    obs_rows: list[pd.DataFrame] = []
    spatial_obs_rows: list[pd.DataFrame] = []
    spatial_uns: dict[str, dict] = {}

    for art, sid in zip(artifacts, sample_ids, strict=True):
        n_cells = art.spliced.shape[0]

        def remap(mtx: sp.csc_matrix, art=art, n_cells=n_cells) -> sp.csc_matrix:
            if n_cells == 0:
                return sp.csc_matrix((0, n_global), dtype=np.float64)
            local_to_global = np.fromiter(
                (gid_to_global[str(g)] for g in art.features["gene_id"]),
                dtype=np.int64, count=len(art.features),
            )
            coo = mtx.tocoo()
            new_cols = local_to_global[coo.col]
            return sp.csr_matrix(
                (coo.data.astype(np.float64), (coo.row, new_cols)),
                shape=(n_cells, n_global),
            ).tocsc()

        s_blocks.append(remap(art.spliced))
        u_blocks.append(remap(art.unspliced))
        a_blocks.append(remap(art.ambiguous))

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

    # vstack in CSR (cheap) then convert to CSC at the end; CSC isn't the
    # natural concat format and vstacking it peaks at multi-GB on real data.
    def _stack(blocks: list[sp.csc_matrix]) -> sp.csc_matrix:
        if not blocks:
            return sp.csc_matrix((0, n_global), dtype=np.float64)
        return sp.vstack(blocks, format="csr").tocsc().astype(np.float64)
    spliced = _stack(s_blocks)
    unspliced = _stack(u_blocks)
    ambiguous = _stack(a_blocks)

    obs = pd.concat(obs_rows, axis=0, ignore_index=True) if obs_rows else \
        pd.DataFrame({"barcode": [], "sample_id": []})
    obs["sample_id"] = obs["sample_id"].astype("category")
    obs.index = pd.Index(
        [f"{bc}-{sid}" for bc, sid in zip(obs["barcode"], obs["sample_id"], strict=True)]
    )

    var = var_global.copy()
    if var_names == "gene_ids":
        var.index = pd.Index(var["gene_id"].astype(str))
    elif var_names == "gene_symbols":
        var.index = pd.Index(var["gene_name"].astype(str))
    else:
        raise ValueError(
            f"var_names must be 'gene_ids' or 'gene_symbols', got {var_names!r}"
        )

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
    else:
        spatial_arr = None

    return (
        spliced, unspliced, ambiguous,
        var, obs,
        (spatial_uns if spatial_uns else None),
        spatial_arr,
    )


def read_starsolo_velocyto(
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
    """Read STARsolo Velocyto output and assemble a cell × gene AnnData with three layers.

    Parameters
    ----------
    sample_dirs, sample_ids, barcode_whitelists, use_internal_whitelist,
    var_names, tissue_positions, spatial_library_ids, verbose
        Same shape as :func:`read_starsolo_gene`. ``sample_ids`` **must
        not contain ``'-'``**: the ``{barcode}-{sample_id}`` ``obs_names``
        convention becomes unparseable if a sample_id has an embedded
        hyphen.

    Returns
    -------
    AnnData
        ``X`` is sparse CSC float64 spliced counts (alias of
        ``layers["spliced"]``). Additional layers ``unspliced`` and
        ``ambiguous`` sit alongside on the same gene axis.

    Notes
    -----
    Both wire formats are auto-detected:

    - **Modern**: three separate ``spliced.mtx`` / ``unspliced.mtx`` /
      ``ambiguous.mtx`` files.
    - **Legacy**: a stacked ``matrix.mtx`` with rows
      ``[0..n_genes)`` spliced, ``[n_genes..2*n_genes)`` unspliced,
      ``[2*n_genes..3*n_genes)`` ambiguous.

    The Velocyto feature has only a ``raw/`` directory in stock STARsolo
    output, so the ``filtered/`` fallback is not applicable here. Internal
    whitelist (when enabled) reads from
    ``Solo.out/Gene/filtered/barcodes.tsv`` next door.

    The reader does not run velocity smoothing or moments — that's
    ``scvelo.pp.filter_and_normalize`` / ``scvelo.tl.velocity``.
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

    artifacts: list[_VelSampleArtifacts] = []
    for sd, sid, bw, tps, lib in zip(sample_dirs, sample_ids, bcw, tp, libs, strict=True):
        artifacts.append(
            _read_one_velocyto_sample(
                sd, sid,
                barcode_whitelist=bw, tissue_positions=tps,
                use_internal_whitelist=use_internal_whitelist,
                spatial_library_id=lib, verbose=verbose,
            )
        )

    (
        spliced, unspliced, ambiguous, var, obs, spatial_uns, spatial_arr
    ) = _concat_velocyto_samples(
        artifacts, sample_ids, var_names=var_names,
    )

    adata = ad.AnnData(
        X=spliced.copy(),
        layers={"spliced": spliced, "unspliced": unspliced, "ambiguous": ambiguous},
        obs=obs, var=var,
    )
    # Record per-sample wire-format detection.
    wire_formats = {sid: art.wire_format for sid, art in zip(sample_ids, artifacts, strict=True)}
    # Use the canonical scsplice namespace key. Legacy h5ad files written by
    # splikit-py 1.0.0 carry uns['splikit']; validators downstream transparently
    # migrate that via scsplice._core._validators.get_scsplice_ns.
    adata.uns["scsplice"] = {
        "version": 1,
        "source": "starsolo",
        "params": {
            "read_starsolo_velocyto": {
                "n_samples": n,
                "var_names": var_names,
                "use_internal_whitelist": bool(use_internal_whitelist),
                "any_explicit_whitelist": any(b is not None for b in bcw),
                "any_tissue_positions": any(t is not None for t in tp),
                "wire_formats": wire_formats,
            }
        },
    }
    if var_names == "gene_symbols":
        adata.var_names_make_unique()
    else:
        if not adata.var_names.is_unique:
            raise RuntimeError(
                "var_names not unique with var_names='gene_ids' — duplicate "
                "gene_id seen across samples."
            )

    if spatial_arr is not None:
        adata.obsm["spatial"] = spatial_arr.astype(np.float64, copy=False)
    if spatial_uns is not None:
        adata.uns["spatial"] = spatial_uns

    if verbose:
        print(
            f"[scsplice.io] Built Velocyto AnnData: {adata.n_obs} cells × "
            f"{adata.n_vars} genes (spliced nnz={adata.layers['spliced'].nnz}, "
            f"unspliced nnz={adata.layers['unspliced'].nnz}, "
            f"ambiguous nnz={adata.layers['ambiguous'].nnz})"
        )
    return adata
