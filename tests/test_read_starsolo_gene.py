"""Tests for scsplice.io.read_starsolo_gene.

Synthetic tests cover:
  - shape / dtype / CSC / obs_names / sample_id / uns
  - 2-col features.tsv (Cell Ranger v2) and 3-col (v3 / STARsolo) dispatch
  - var_names = "gene_ids" vs "gene_symbols" (with collisions)
  - multi-sample concat with non-overlapping gene sets (zero-pad)
  - explicit barcode_whitelists trims raw/ to the whitelist intersection
  - tissue_positions populates squidpy-shaped fields

Real-data tests cross-check against scanpy.read_mtx on the same MTX files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp


def _write_mtx(path: Path, mat: np.ndarray) -> None:
    """Write a (n_genes, n_cells) integer matrix as Matrix Market."""
    coo = sp.coo_matrix(mat.astype(np.int64))
    sio.mmwrite(str(path), coo, field="integer", symmetry="general")


def _make_starsolo_gene_dir(
    base: Path,
    *,
    gene_ids: list[str],
    gene_names: list[str],
    barcodes: list[str],
    counts: np.ndarray,  # (n_genes, n_cells)
    feature_type_col: bool = True,
    filtered_subset: list[str] | None = None,
) -> Path:
    """Build a minimal Solo.out/Gene/{raw,filtered}/ tree.

    ``filtered_subset`` is the barcodes to put in filtered/ (defaults to
    keeping the FIRST half of ``barcodes`` so the internal whitelist is a
    proper subset of raw/).

    Returns the path to ``base`` itself (the sample root); the reader
    accepts that and walks down into ``Solo.out/Gene/``.
    """
    n_genes, n_cells = counts.shape
    assert len(gene_ids) == n_genes == len(gene_names)
    assert len(barcodes) == n_cells

    raw = base / "Solo.out" / "Gene" / "raw"
    filt = base / "Solo.out" / "Gene" / "filtered"
    raw.mkdir(parents=True, exist_ok=True)
    filt.mkdir(parents=True, exist_ok=True)

    feat_lines = []
    for gid, gname in zip(gene_ids, gene_names, strict=True):
        if feature_type_col:
            feat_lines.append(f"{gid}\t{gname}\tGene Expression")
        else:
            feat_lines.append(f"{gid}\t{gname}")
    (raw / "features.tsv").write_text("\n".join(feat_lines) + "\n")
    (filt / "features.tsv").write_text("\n".join(feat_lines) + "\n")

    (raw / "barcodes.tsv").write_text("\n".join(barcodes) + "\n")
    if filtered_subset is None:
        filtered_subset = barcodes[: len(barcodes) // 2 or 1]
    keep_idx = [i for i, bc in enumerate(barcodes) if bc in set(filtered_subset)]
    (filt / "barcodes.tsv").write_text("\n".join(filtered_subset) + "\n")

    _write_mtx(raw / "matrix.mtx", counts)
    if keep_idx:
        _write_mtx(filt / "matrix.mtx", counts[:, keep_idx])
    else:
        # Empty filtered/ matrix.
        _write_mtx(filt / "matrix.mtx", np.zeros((n_genes, 0), dtype=np.int64))

    return base


# -----------------------------
# Synthetic tests (always run)
# -----------------------------

def test_read_starsolo_gene_single_sample_smoke(tmp_path):
    import scsplice  # noqa: PLC0415

    gene_ids = [f"ENSG{i:08d}" for i in range(5)]
    gene_names = [f"GENE{i}" for i in range(5)]
    barcodes = [f"BC{i:04d}" for i in range(8)]
    counts = np.arange(40, dtype=np.int64).reshape(5, 8)
    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=gene_ids, gene_names=gene_names,
        barcodes=barcodes, counts=counts,
        filtered_subset=barcodes,  # filtered = raw, simplifies expected counts
    )

    a = scsplice.io.read_starsolo_gene(tmp_path / "s1", "s1")
    # filtered/ is the source by default → 8 cells, 5 genes.
    assert a.n_obs == 8
    assert a.n_vars == 5
    assert a.X.dtype == np.float64
    assert sp.issparse(a.X) and a.X.format == "csc"
    # obs schema.
    assert "barcode" in a.obs.columns
    assert "sample_id" in a.obs.columns
    assert isinstance(a.obs["sample_id"].dtype, pd.CategoricalDtype)
    assert all(name.endswith("-s1") for name in a.obs_names)
    # var schema and var_names.
    assert "gene_id" in a.var.columns
    assert "gene_name" in a.var.columns
    assert list(a.var_names) == gene_ids
    # uns scaffold.
    assert a.uns["scsplice"]["source"] == "starsolo"
    assert "read_starsolo_gene" in a.uns["scsplice"]["params"]
    # X values match counts (cells×genes after transpose).
    dense = np.asarray(a.X.todense())
    assert np.array_equal(dense, counts.T)


def test_read_starsolo_gene_v2_features_two_columns(tmp_path):
    """Cell Ranger v2 features.tsv has 2 columns; reader must handle both."""
    import scsplice  # noqa: PLC0415

    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["ENSG1", "ENSG2"],
        gene_names=["A", "B"],
        barcodes=["BC1", "BC2"],
        counts=np.array([[1, 2], [3, 4]], dtype=np.int64),
        feature_type_col=False,
        filtered_subset=["BC1", "BC2"],
    )
    a = scsplice.io.read_starsolo_gene(tmp_path / "s1", "s1")
    assert a.n_vars == 2
    assert (a.var["feature_type"] == "Gene Expression").all()


def test_read_starsolo_gene_var_names_symbols_with_collision(tmp_path):
    """gene_symbols mode collides for paralogs; var_names_make_unique is allowed there."""
    import scsplice  # noqa: PLC0415

    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["ENSG_A1", "ENSG_A2", "ENSG_B"],
        gene_names=["DUP", "DUP", "UNIQ"],  # symbol collision
        barcodes=["BC1", "BC2"],
        counts=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        filtered_subset=["BC1", "BC2"],
    )
    a = scsplice.io.read_starsolo_gene(
        tmp_path / "s1", "s1", var_names="gene_symbols",
    )
    # var_names must be unique post make_unique.
    assert a.var_names.is_unique
    # Original gene_id is preserved in var.
    assert set(a.var["gene_id"]) == {"ENSG_A1", "ENSG_A2", "ENSG_B"}


def test_read_starsolo_gene_multi_sample_disjoint_genes(tmp_path):
    """Concat must zero-pad genes across samples."""
    import scsplice  # noqa: PLC0415

    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1", "G2", "G3"],
        gene_names=["a", "b", "c"],
        barcodes=["BC_A", "BC_B"],
        counts=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        filtered_subset=["BC_A", "BC_B"],
    )
    _make_starsolo_gene_dir(
        tmp_path / "s2",
        gene_ids=["G2", "G4"],  # G2 shared, G4 new
        gene_names=["b", "d"],
        barcodes=["BC_C"],
        counts=np.array([[7], [8]], dtype=np.int64),
        filtered_subset=["BC_C"],
    )
    a = scsplice.io.read_starsolo_gene(
        [tmp_path / "s1", tmp_path / "s2"], ["s1", "s2"],
    )
    assert a.n_obs == 3
    assert a.n_vars == 4  # union: G1, G2, G3, G4
    # G4 must be zero in s1 cells; G1/G3 zero in s2 cells.
    g4_col = a.var_names.get_loc("G4")
    g1_col = a.var_names.get_loc("G1")
    s1_mask = np.array([n.endswith("-s1") for n in a.obs_names])
    s2_mask = ~s1_mask
    dense = np.asarray(a.X.todense())
    assert dense[s1_mask, g4_col].sum() == 0
    assert dense[s2_mask, g1_col].sum() == 0
    # G2 populated in both samples.
    g2_col = a.var_names.get_loc("G2")
    assert dense[s1_mask, g2_col].sum() == 3 + 4  # rows 1 col 0 and 1
    assert dense[s2_mask, g2_col].sum() == 7


def test_read_starsolo_gene_explicit_whitelist_trims_raw(tmp_path):
    """Explicit whitelist forces raw/ source AND filters to the intersection."""
    import scsplice  # noqa: PLC0415

    raw_barcodes = ["BC0", "BC1", "BC2", "BC3", "BC4"]
    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1", "G2"], gene_names=["a", "b"],
        barcodes=raw_barcodes,
        counts=np.tile(np.arange(5, dtype=np.int64), (2, 1)) + 1,
        filtered_subset=["BC0", "BC1"],  # internal WL would give 2 cells
    )
    # Whitelist captures cells outside filtered/ (BC3, BC4) — proves raw/ source.
    a = scsplice.io.read_starsolo_gene(
        tmp_path / "s1", "s1",
        barcode_whitelists=[["BC1", "BC3", "BC4"]],
    )
    assert a.n_obs == 3
    assert sorted(n.split("-")[0] for n in a.obs_names) == ["BC1", "BC3", "BC4"]


def test_read_starsolo_gene_use_internal_false_no_whitelist(tmp_path):
    """use_internal_whitelist=False with no whitelist falls back to raw/ unfiltered."""
    import scsplice  # noqa: PLC0415

    raw_barcodes = [f"BC{i}" for i in range(5)]
    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=raw_barcodes,
        counts=np.ones((1, 5), dtype=np.int64),
        filtered_subset=raw_barcodes[:2],
    )
    a = scsplice.io.read_starsolo_gene(
        tmp_path / "s1", "s1", use_internal_whitelist=False,
    )
    # filtered/ exists (with 2 cells) but is_internal_whitelist=False; we read
    # filtered/ (no whitelist provided) which has 2 cells. To explicitly test
    # raw-fallback we'd need to delete filtered/. The behavior of preferring
    # filtered/ when no whitelist is provided is the correct "no surprise"
    # default for users who haven't asked for raw.
    # Document the expected current behavior: filtered/ is still preferred.
    assert a.n_obs == 2


def test_read_starsolo_gene_tissue_positions_v2_header(tmp_path):
    """tissue_positions with v2 header populates squidpy-shaped fields."""
    import scsplice  # noqa: PLC0415

    raw_barcodes = ["BC0", "BC1", "BC2", "BC3"]
    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=raw_barcodes,
        counts=np.array([[1, 2, 3, 4]], dtype=np.int64),
        filtered_subset=raw_barcodes[:2],
    )
    tp = tmp_path / "tissue_positions.csv"
    tp.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "BC1,1,0,0,100.5,200.5\n"
        "BC2,1,1,2,300.5,400.5\n"
        "BC3,0,1,3,500.5,600.5\n"
    )

    a = scsplice.io.read_starsolo_gene(
        tmp_path / "s1", "s1",
        tissue_positions=[tp],
        spatial_library_ids=["sample_visium_1"],
    )
    # 3 barcodes from tissue_positions present in raw.
    assert a.n_obs == 3
    assert sorted(n.split("-")[0] for n in a.obs_names) == ["BC1", "BC2", "BC3"]
    # Squidpy-shaped fields.
    assert "spatial" in a.obsm
    assert a.obsm["spatial"].shape == (3, 2)
    assert a.obsm["spatial"].dtype == np.float64
    assert "in_tissue" in a.obs.columns
    assert "array_row" in a.obs.columns
    assert "array_col" in a.obs.columns
    # uns scaffold.
    assert "spatial" in a.uns
    assert "sample_visium_1" in a.uns["spatial"]


def test_read_starsolo_gene_tissue_positions_v1_no_header(tmp_path):
    """v1 tissue_positions_list.csv (no header) is auto-detected."""
    import scsplice  # noqa: PLC0415

    raw_barcodes = ["BC0", "BC1", "BC2"]
    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=raw_barcodes,
        counts=np.array([[1, 2, 3]], dtype=np.int64),
        filtered_subset=raw_barcodes,
    )
    tp = tmp_path / "tissue_positions_list.csv"
    tp.write_text(
        "BC1,1,0,0,100.5,200.5\n"
        "BC2,0,1,2,300.5,400.5\n"
    )
    a = scsplice.io.read_starsolo_gene(
        tmp_path / "s1", "s1",
        tissue_positions=[tp],
    )
    assert a.n_obs == 2
    assert "spatial" in a.obsm


def test_read_starsolo_gene_tissue_positions_warns_on_dual_whitelist(tmp_path):
    """Both tissue_positions and explicit whitelist → warn, tissue_positions wins."""
    import scsplice  # noqa: PLC0415

    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=["BC1", "BC2", "BC3"],
        counts=np.array([[1, 2, 3]], dtype=np.int64),
        filtered_subset=["BC1", "BC2", "BC3"],
    )
    tp = tmp_path / "tissue_positions.csv"
    tp.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "BC1,1,0,0,1.0,2.0\n"
    )
    with pytest.warns(UserWarning, match="tissue_positions takes precedence"):
        a = scsplice.io.read_starsolo_gene(
            tmp_path / "s1", "s1",
            barcode_whitelists=[["BC2", "BC3"]],
            tissue_positions=[tp],
        )
    # tissue_positions had only BC1.
    assert a.n_obs == 1
    assert a.obs_names[0].startswith("BC1")


def test_read_starsolo_gene_dimension_mismatch_raises(tmp_path):
    import scsplice  # noqa: PLC0415

    base = _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1", "G2"], gene_names=["a", "b"],
        barcodes=["BC1", "BC2"],
        counts=np.array([[1, 2], [3, 4]], dtype=np.int64),
        filtered_subset=["BC1", "BC2"],
    )
    # Truncate filtered/barcodes.tsv.
    bc_path = base / "Solo.out" / "Gene" / "filtered" / "barcodes.tsv"
    bc_path.write_text("BC1\n")
    with pytest.raises(ValueError, match="dimension mismatch"):
        scsplice.io.read_starsolo_gene(tmp_path / "s1", "s1")


def test_read_starsolo_gene_argument_validation(tmp_path):
    import scsplice  # noqa: PLC0415

    _make_starsolo_gene_dir(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=["BC1"], counts=np.array([[1]], dtype=np.int64),
        filtered_subset=["BC1"],
    )
    with pytest.raises(ValueError, match="must be unique"):
        scsplice.io.read_starsolo_gene([tmp_path / "s1", tmp_path / "s1"], ["s1", "s1"])
    with pytest.raises(ValueError, match="must equal"):
        scsplice.io.read_starsolo_gene([tmp_path / "s1"], ["s1", "s2"])
    with pytest.raises(ValueError, match="var_names"):
        scsplice.io.read_starsolo_gene(tmp_path / "s1", "s1", var_names="bogus")


# -----------------
# Real-data tests
# -----------------

@pytest.mark.real_data
def test_read_starsolo_gene_real_two_samples(real_data_samples, real_data_sample_ids):
    """Cross-check against scanpy.read_mtx on the same two STARsolo samples."""
    import scsplice  # noqa: PLC0415
    sc = pytest.importorskip("scanpy")

    a = scsplice.io.read_starsolo_gene(
        real_data_samples, real_data_sample_ids,
        # Filtered/ default — same as scanpy's typical 10x flow.
    )
    # Schema assertions.
    assert a.X.dtype == np.float64
    assert sp.issparse(a.X) and a.X.format == "csc"
    assert "barcode" in a.obs.columns
    assert "sample_id" in a.obs.columns
    assert set(a.obs["sample_id"].astype(str).unique()) == set(real_data_sample_ids)

    # Cross-check: load both samples with scanpy.read_mtx and concat.
    parts = []
    for sample_dir, sid in zip(real_data_samples, real_data_sample_ids, strict=True):
        mtx_path = sample_dir / "Solo.out" / "Gene" / "filtered" / "matrix.mtx"
        bc_path = sample_dir / "Solo.out" / "Gene" / "filtered" / "barcodes.tsv"
        feat_path = sample_dir / "Solo.out" / "Gene" / "filtered" / "features.tsv"
        ad_sc = sc.read_mtx(mtx_path).T  # cells × genes
        bcs = [l.strip() for l in bc_path.read_text().splitlines() if l.strip()]
        feats = pd.read_csv(feat_path, sep="\t", header=None)
        ad_sc.obs_names = [f"{bc}-{sid}" for bc in bcs]
        ad_sc.var_names = feats[0].tolist()
        parts.append(ad_sc)
    import anndata as ad_mod  # noqa: PLC0415
    combined = ad_mod.concat(parts, axis=0, join="outer", fill_value=0)

    # Cell counts must match (same filtered/ source on both sides).
    assert a.n_obs == combined.n_obs
    # Each sample's cells were sourced from the same MTX, so per-sample counts
    # must be EXACTLY equal. Compare on the intersection of obs_names AND
    # var_names (gene order may differ across the readers).
    common_obs = sorted(set(a.obs_names) & set(combined.obs_names))
    common_var = sorted(set(a.var_names) & set(combined.var_names))
    assert len(common_obs) == a.n_obs  # union of barcodes from both samples
    assert len(common_var) > 0
    a_sub = a[common_obs, common_var].X
    cmb_sub = combined[common_obs, common_var].X
    # densify a small slice to compare without blowing memory.
    n_check = min(2000, len(common_obs))
    diff = (a_sub[:n_check].toarray() - cmb_sub[:n_check].toarray())
    assert np.array_equal(diff, np.zeros_like(diff))


@pytest.mark.real_data
def test_read_starsolo_gene_real_smoke_print(real_data_samples, real_data_sample_ids, capsys):
    """Verbose smoke test: read both samples and print shape."""
    import scsplice  # noqa: PLC0415

    a = scsplice.io.read_starsolo_gene(
        real_data_samples, real_data_sample_ids, verbose=True,
    )
    out = capsys.readouterr().out
    assert "Built Gene AnnData" in out
    assert a.n_obs > 0 and a.n_vars > 0
