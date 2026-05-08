"""Tests for splikit.io.read_starsolo_velocyto.

Both wire formats are exercised:
  - **split** (modern): three sibling spliced.mtx / unspliced.mtx / ambiguous.mtx
  - **stacked** (legacy): single matrix.mtx with rows partitioned 0..n / n..2n / 2n..3n

Real-data tests are gated by ``@pytest.mark.real_data`` and cross-check
against ``scanpy.read_mtx`` (and ``scvelo.read`` if available).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp


def _write_mtx(path: Path, mat: np.ndarray) -> None:
    coo = sp.coo_matrix(mat.astype(np.int64))
    sio.mmwrite(str(path), coo, field="integer", symmetry="general")


def _make_velocyto_split(
    base: Path,
    *,
    gene_ids: list[str],
    gene_names: list[str],
    barcodes: list[str],
    spliced: np.ndarray,
    unspliced: np.ndarray,
    ambiguous: np.ndarray,
) -> Path:
    raw = base / "Solo.out" / "Velocyto" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    feat_lines = [
        f"{gid}\t{gname}\tGene Expression"
        for gid, gname in zip(gene_ids, gene_names, strict=True)
    ]
    (raw / "features.tsv").write_text("\n".join(feat_lines) + "\n")
    (raw / "barcodes.tsv").write_text("\n".join(barcodes) + "\n")
    _write_mtx(raw / "spliced.mtx", spliced)
    _write_mtx(raw / "unspliced.mtx", unspliced)
    _write_mtx(raw / "ambiguous.mtx", ambiguous)
    return base


def _make_velocyto_stacked(
    base: Path,
    *,
    gene_ids: list[str],
    gene_names: list[str],
    barcodes: list[str],
    spliced: np.ndarray,
    unspliced: np.ndarray,
    ambiguous: np.ndarray,
) -> Path:
    raw = base / "Solo.out" / "Velocyto" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    feat_lines = [
        f"{gid}\t{gname}\tGene Expression"
        for gid, gname in zip(gene_ids, gene_names, strict=True)
    ]
    (raw / "features.tsv").write_text("\n".join(feat_lines) + "\n")
    (raw / "barcodes.tsv").write_text("\n".join(barcodes) + "\n")
    stacked = np.vstack([spliced, unspliced, ambiguous])
    _write_mtx(raw / "matrix.mtx", stacked)
    return base


# ---------------------
# Synthetic — split fmt
# ---------------------

def test_velocyto_split_single_sample_smoke(tmp_path):
    import splikit  # noqa: PLC0415

    n_genes, n_cells = 4, 6
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, (n_genes, n_cells)).astype(np.int64)
    u = rng.integers(0, 5, (n_genes, n_cells)).astype(np.int64)
    am = rng.integers(0, 2, (n_genes, n_cells)).astype(np.int64)
    _make_velocyto_split(
        tmp_path / "s1",
        gene_ids=[f"ENSG{i}" for i in range(n_genes)],
        gene_names=[f"G{i}" for i in range(n_genes)],
        barcodes=[f"BC{i}" for i in range(n_cells)],
        spliced=s, unspliced=u, ambiguous=am,
    )
    a = splikit.io.read_starsolo_velocyto(
        tmp_path / "s1", "s1", use_internal_whitelist=False,
    )
    assert a.n_obs == n_cells
    assert a.n_vars == n_genes
    for layer in ("spliced", "unspliced", "ambiguous"):
        assert layer in a.layers
        L = a.layers[layer]
        assert sp.issparse(L) and L.format == "csc"
        assert L.dtype == np.float64
        assert L.shape == (n_cells, n_genes)
    # X aliases / equals spliced.
    assert sp.issparse(a.X)
    assert np.array_equal(np.asarray(a.X.todense()), np.asarray(a.layers["spliced"].todense()))
    # Counts match the input matrices (after transpose).
    assert np.array_equal(np.asarray(a.layers["spliced"].todense()), s.T)
    assert np.array_equal(np.asarray(a.layers["unspliced"].todense()), u.T)
    assert np.array_equal(np.asarray(a.layers["ambiguous"].todense()), am.T)
    # uns / wire_format detection.
    assert a.uns["splikit"]["params"]["read_starsolo_velocyto"]["wire_formats"] == {"s1": "split"}


def test_velocyto_stacked_format_decoded(tmp_path):
    import splikit  # noqa: PLC0415

    n_genes, n_cells = 3, 4
    s = np.array([[1, 0, 0, 0], [2, 2, 0, 0], [0, 0, 3, 3]], dtype=np.int64)
    u = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [4, 0, 0, 0]], dtype=np.int64)
    am = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]], dtype=np.int64)
    _make_velocyto_stacked(
        tmp_path / "s1",
        gene_ids=[f"E{i}" for i in range(n_genes)],
        gene_names=[f"N{i}" for i in range(n_genes)],
        barcodes=[f"BC{i}" for i in range(n_cells)],
        spliced=s, unspliced=u, ambiguous=am,
    )
    a = splikit.io.read_starsolo_velocyto(
        tmp_path / "s1", "s1", use_internal_whitelist=False,
    )
    assert a.uns["splikit"]["params"]["read_starsolo_velocyto"]["wire_formats"] == {"s1": "stacked"}
    assert np.array_equal(np.asarray(a.layers["spliced"].todense()), s.T)
    assert np.array_equal(np.asarray(a.layers["unspliced"].todense()), u.T)
    assert np.array_equal(np.asarray(a.layers["ambiguous"].todense()), am.T)


def test_velocyto_split_multi_sample_concat_disjoint_genes(tmp_path):
    import splikit  # noqa: PLC0415

    _make_velocyto_split(
        tmp_path / "s1",
        gene_ids=["G1", "G2"], gene_names=["a", "b"],
        barcodes=["BC_A", "BC_B"],
        spliced=np.array([[1, 1], [2, 2]], dtype=np.int64),
        unspliced=np.array([[0, 0], [0, 0]], dtype=np.int64),
        ambiguous=np.array([[0, 0], [0, 0]], dtype=np.int64),
    )
    _make_velocyto_split(
        tmp_path / "s2",
        gene_ids=["G2", "G3"], gene_names=["b", "c"],
        barcodes=["BC_C"],
        spliced=np.array([[7], [8]], dtype=np.int64),
        unspliced=np.array([[0], [1]], dtype=np.int64),
        ambiguous=np.array([[0], [0]], dtype=np.int64),
    )
    a = splikit.io.read_starsolo_velocyto(
        [tmp_path / "s1", tmp_path / "s2"], ["s1", "s2"],
        use_internal_whitelist=False,
    )
    assert a.n_obs == 3
    assert a.n_vars == 3  # G1, G2, G3
    s1_mask = np.array([n.endswith("-s1") for n in a.obs_names])
    g3_idx = a.var_names.get_loc("G3")
    g1_idx = a.var_names.get_loc("G1")
    spliced = np.asarray(a.layers["spliced"].todense())
    assert spliced[s1_mask, g3_idx].sum() == 0  # zero-pad
    assert spliced[~s1_mask, g1_idx].sum() == 0
    # G2 spliced sum: s1 cells contribute 2+2=4, s2 contributes 7.
    g2_idx = a.var_names.get_loc("G2")
    assert spliced[s1_mask, g2_idx].sum() == 4
    assert spliced[~s1_mask, g2_idx].sum() == 7


def test_velocyto_explicit_whitelist(tmp_path):
    import splikit  # noqa: PLC0415

    _make_velocyto_split(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=["BC0", "BC1", "BC2", "BC3"],
        spliced=np.array([[1, 2, 3, 4]], dtype=np.int64),
        unspliced=np.array([[0, 0, 0, 0]], dtype=np.int64),
        ambiguous=np.array([[0, 0, 0, 0]], dtype=np.int64),
    )
    a = splikit.io.read_starsolo_velocyto(
        tmp_path / "s1", "s1",
        barcode_whitelists=[["BC1", "BC3"]],
        use_internal_whitelist=False,
    )
    assert a.n_obs == 2
    assert sorted(n.split("-")[0] for n in a.obs_names) == ["BC1", "BC3"]


def test_velocyto_tissue_positions_squidpy_fields(tmp_path):
    import splikit  # noqa: PLC0415

    _make_velocyto_split(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=["BC1", "BC2", "BC3", "BC4"],
        spliced=np.array([[1, 2, 3, 4]], dtype=np.int64),
        unspliced=np.array([[0, 0, 0, 0]], dtype=np.int64),
        ambiguous=np.array([[0, 0, 0, 0]], dtype=np.int64),
    )
    tp = tmp_path / "tissue_positions.csv"
    tp.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "BC2,1,2,3,10.0,20.0\n"
        "BC4,0,4,5,30.0,40.0\n"
    )
    a = splikit.io.read_starsolo_velocyto(
        tmp_path / "s1", "s1",
        tissue_positions=[tp],
        spatial_library_ids=["lib_X"],
        use_internal_whitelist=False,
    )
    assert a.n_obs == 2
    assert "spatial" in a.obsm
    assert a.obsm["spatial"].shape == (2, 2)
    assert "in_tissue" in a.obs.columns
    assert "lib_X" in a.uns["spatial"]


def test_velocyto_stacked_dimension_mismatch(tmp_path):
    """Stacked layout with wrong row count → ValueError."""
    import splikit  # noqa: PLC0415

    n_genes, n_cells = 2, 3
    raw = tmp_path / "s1" / "Solo.out" / "Velocyto" / "raw"
    raw.mkdir(parents=True)
    (raw / "features.tsv").write_text("G1\ta\tGene Expression\nG2\tb\tGene Expression\n")
    (raw / "barcodes.tsv").write_text("BC1\nBC2\nBC3\n")
    # Stacked must have 3*n_genes=6 rows; we write 5 to trigger the mismatch.
    bad = np.arange(5 * n_cells, dtype=np.int64).reshape(5, n_cells)
    _write_mtx(raw / "matrix.mtx", bad)
    with pytest.raises(ValueError, match="3 \\* n_genes"):
        splikit.io.read_starsolo_velocyto(
            tmp_path / "s1", "s1", use_internal_whitelist=False,
        )


def test_velocyto_argument_validation(tmp_path):
    import splikit  # noqa: PLC0415

    _make_velocyto_split(
        tmp_path / "s1",
        gene_ids=["G1"], gene_names=["a"],
        barcodes=["BC1"],
        spliced=np.array([[1]], dtype=np.int64),
        unspliced=np.array([[0]], dtype=np.int64),
        ambiguous=np.array([[0]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="must be unique"):
        splikit.io.read_starsolo_velocyto(
            [tmp_path / "s1", tmp_path / "s1"], ["s1", "s1"],
            use_internal_whitelist=False,
        )
    with pytest.raises(ValueError, match="must equal"):
        splikit.io.read_starsolo_velocyto(
            [tmp_path / "s1"], ["s1", "s2"],
            use_internal_whitelist=False,
        )


# -----------------
# Real-data tests
# -----------------

@pytest.mark.real_data
def test_velocyto_real_two_samples_split_format(real_data_samples, real_data_sample_ids):
    """Cross-check spliced.mtx exact-equal against scipy.io.mmread on real data.

    Uses the internal Gene/filtered/ whitelist throughout to keep memory
    bounded — the raw Velocyto MTX files in this dataset are ~1.2 GB each
    over ~6.7M barcodes, so loading without a whitelist OOM-kills the test.
    """
    import scipy.io as sio  # noqa: PLC0415
    import splikit  # noqa: PLC0415

    a = splikit.io.read_starsolo_velocyto(
        real_data_samples, real_data_sample_ids,
        use_internal_whitelist=True,
    )

    assert a.n_vars > 0 and a.n_obs > 0
    for layer in ("spliced", "unspliced", "ambiguous"):
        assert sp.issparse(a.layers[layer])
        assert a.layers[layer].dtype == np.float64
        assert a.layers[layer].shape == (a.n_obs, a.n_vars)
    # X mirrors spliced (alias-by-copy).
    assert a.X.shape == a.layers["spliced"].shape
    assert a.X.nnz == a.layers["spliced"].nnz

    wf = a.uns["splikit"]["params"]["read_starsolo_velocyto"]["wire_formats"]
    assert all(v == "split" for v in wf.values())

    # Cross-check sample 0's spliced.mtx against a direct scipy.io.mmread,
    # via column subsetting (cheap) instead of building a full transposed
    # AnnData (expensive). The on-disk layout is genes × cells.
    sample_dir = real_data_samples[0]
    sid = real_data_sample_ids[0]
    raw_dir = sample_dir / "Solo.out" / "Velocyto" / "raw"

    bcs_raw = pd.read_csv(
        raw_dir / "barcodes.tsv", sep="\t", header=None, names=["barcode"],
        dtype=str,
    )["barcode"].to_numpy()
    feats_raw = pd.read_csv(
        raw_dir / "features.tsv", sep="\t", header=None,
    )[0].to_numpy()

    # Build a barcode → raw-column-index map, then look up the columns we kept.
    raw_bc_to_idx = {bc: i for i, bc in enumerate(bcs_raw)}
    obs_names_for_sid = [n for n in a.obs_names if n.endswith(f"-{sid}")]
    kept_bcs = [n[: -(len(sid) + 1)] for n in obs_names_for_sid]
    raw_cols = np.array([raw_bc_to_idx[bc] for bc in kept_bcs], dtype=np.int64)

    raw_feat_to_idx = {f: i for i, f in enumerate(feats_raw)}
    raw_rows = np.array(
        [raw_feat_to_idx[v] for v in a.var_names if v in raw_feat_to_idx],
        dtype=np.int64,
    )
    var_keep_mask = np.array(
        [v in raw_feat_to_idx for v in a.var_names], dtype=bool,
    )

    spliced_full = sio.mmread(str(raw_dir / "spliced.mtx")).tocsc()  # genes × cells
    raw_block = spliced_full[np.ix_(raw_rows, raw_cols)].toarray()    # genes × kept_cells

    # Our reader stores cells × genes; transpose for comparison.
    our_block = (
        a[obs_names_for_sid, var_keep_mask].layers["spliced"].T.toarray()
    )
    assert raw_block.shape == our_block.shape
    assert np.array_equal(raw_block, our_block)


@pytest.mark.real_data
def test_velocyto_real_scvelo_compatible_if_available(real_data_samples, real_data_sample_ids):
    """Smoke check that the AnnData is shaped the way scvelo expects.

    Loads with the internal whitelist so we stay below the OOM threshold on
    the real Velocyto MTX (~1.2 GB / sample with ~6.7M raw barcodes).
    """
    import splikit  # noqa: PLC0415

    a = splikit.io.read_starsolo_velocyto(
        real_data_samples, real_data_sample_ids,
        use_internal_whitelist=True,
    )
    assert "spliced" in a.layers and "unspliced" in a.layers
    assert a.layers["spliced"].shape == a.X.shape

    scv = pytest.importorskip("scvelo", reason="scvelo not installed; skipping deeper check")
    b = a[:200, :500].copy()
    scv.utils.show_proportions(b)
