"""Tests for splikit.io.read_starsolo.

Synthetic STARsolo directories are built per-test with `_make_starsolo_dir`.
The fixtures cover the LJV-grouping edge cases that are easy to get wrong:
the dual-qualifying junction (qualifies for both _S and _E), the cross-sample
LJV (per-sample singleton becomes group member after global concat), and the
zero-padding union of disjoint junction sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp


def _make_starsolo_dir(
    base: Path,
    junctions: list[tuple[str, int, int, int, int, int, int]],
    barcodes: list[str],
    counts: np.ndarray,
) -> Path:
    """Write a minimal STARsolo SJ output to ``base / 'Solo.out' / 'SJ' / 'raw' / ...``.

    ``junctions`` rows: ``(chr, start, end, strand_int, intron_motif, is_annot,
    unique_mapped)`` matching SJ.out.tab columns.
    ``counts`` shape: ``(n_junctions, n_cells)`` integer count matrix.
    Returns the path to the SJ directory (``Solo.out/SJ/``) — the user-facing
    arg that read_starsolo accepts.
    """
    solo = base / "Solo.out"
    sj = solo / "SJ"
    raw = sj / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    n_jx, n_cells = counts.shape
    assert len(junctions) == n_jx
    assert len(barcodes) == n_cells

    # SJ.out.tab lives at the sample root (alongside Solo.out), matching
    # STAR's actual output convention. The resolver also accepts legacy
    # locations inside Solo.out / Solo.out/SJ as fallbacks.
    sj_tab = base / "SJ.out.tab"
    with sj_tab.open("w") as f:
        for jx in junctions:
            f.write("\t".join(str(x) for x in jx) + "\n")

    bc_path = raw / "barcodes.tsv"
    bc_path.write_text("\n".join(barcodes) + "\n")

    coo = sp.coo_matrix(counts.astype(np.int64))
    sio.mmwrite(str(raw / "matrix.mtx"), coo, field="integer", symmetry="general")

    return sj


def test_read_starsolo_single_sample_smoke(tmp_path):
    import splikit  # noqa: PLC0415

    junctions = [
        # Two junctions sharing (chr, start) → start group of size 2 (both _S).
        ("chr1", 100, 200, 1, 1, 1, 5),
        ("chr1", 100, 300, 1, 1, 0, 4),
        # Two junctions sharing (chr, end=400) → end group of size 2 (both _E).
        ("chr1", 500, 400, 1, 1, 1, 6),
        ("chr1", 600, 400, 1, 1, 0, 7),
        # Singleton (no shared start, no shared end) — DROPPED.
        ("chr2", 1000, 2000, 1, 1, 1, 3),
    ]
    barcodes = ["AAAACCCC", "AAAAGGGG", "TTTTCCCC", "TTTTGGGG"]
    counts = np.array(
        [[5, 0, 1, 2],
         [3, 1, 0, 0],
         [0, 4, 5, 1],
         [1, 0, 2, 3],
         [9, 9, 9, 9]],
        dtype=np.int64,
    )
    sj_dir = _make_starsolo_dir(tmp_path / "s1", junctions, barcodes, counts)

    a = splikit.io.read_starsolo(sj_dir, "s1", use_internal_whitelist=False)
    assert a.n_obs == 4
    # 2 _S rows + 2 _E rows; the chr2 singleton is dropped.
    assert a.n_vars == 4
    kinds = a.var["group_kind"].astype(str).tolist()
    assert kinds.count("S") == 2
    assert kinds.count("E") == 2
    assert all(name.endswith("_S") or name.endswith("_E") for name in a.var_names)
    assert a.var_names.is_unique
    # var_names_make_unique would have appended -1 / -2 here if any duplicates leaked.
    # group_id must be dense 0..G-1.
    gid = a.var["group_id"].to_numpy()
    assert gid.dtype == np.int32
    assert set(gid.tolist()) == {0, 1}  # 1 S-group + 1 E-group
    # M1 is cells x events float64 CSC.
    M1 = a.layers["M1"]
    assert M1.shape == (4, 4)
    assert M1.dtype == np.float64
    assert sp.issparse(M1) and M1.format == "csc"
    # uns provenance.
    assert a.uns["splikit"]["m2_valid"] is False
    assert a.uns["splikit"]["source"] == "starsolo"
    assert a.uns["splikit"]["ljv_kind"] == "start_end"
    # obs schema.
    assert "sample_id" in a.obs.columns
    assert "barcode" in a.obs.columns
    assert all(name.endswith("-s1") for name in a.obs_names)


def test_read_starsolo_dual_qualifying_junction(tmp_path):
    """A junction in BOTH a start-group and an end-group must produce TWO event rows."""
    import splikit  # noqa: PLC0415

    junctions = [
        # Junction A and B share start=100 (start-group of 2)
        ("chr1", 100, 200, 1, 1, 1, 5),  # A
        ("chr1", 100, 300, 1, 1, 0, 4),  # B
        # Junction C shares end=200 with A (end-group of 2 with A)
        ("chr1", 50, 200, 1, 1, 1, 6),   # C
    ]
    barcodes = ["BC1", "BC2", "BC3"]
    counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    sj_dir = _make_starsolo_dir(tmp_path / "s1", junctions, barcodes, counts)

    a = splikit.io.read_starsolo(sj_dir, "s1", use_internal_whitelist=False)
    # A is in both S-group (with B) and E-group (with C) → A appears twice (_S and _E).
    # B is in S-group only → one _S row.
    # C is in E-group only → one _E row.
    a_rows = a.var.loc[a.var["row_names_mtx"] == "chr1:100-200"]
    assert len(a_rows) == 2
    assert set(a_rows["group_kind"].astype(str)) == {"S", "E"}
    # A's two var rows must have identical M1 columns (it's the same junction).
    a_idx = a.var.index[a.var["row_names_mtx"] == "chr1:100-200"].tolist()
    assert len(a_idx) == 2
    col0 = np.asarray(a.layers["M1"][:, a.var_names.get_loc(a_idx[0])].todense()).ravel()
    col1 = np.asarray(a.layers["M1"][:, a.var_names.get_loc(a_idx[1])].todense()).ravel()
    assert np.array_equal(col0, col1)


def test_read_starsolo_two_samples_cross_sample_ljv(tmp_path):
    """A junction is a singleton WITHIN each sample but qualifies for an LJV
    group AFTER global concat. The reader must compute groups globally, not
    per-sample.
    """
    import splikit  # noqa: PLC0415

    # Sample 1 has junction X = (chr1, 100, 200) — locally singleton.
    s1 = _make_starsolo_dir(
        tmp_path / "s1",
        [("chr1", 100, 200, 1, 1, 1, 5)],
        ["BC_A", "BC_B"],
        np.array([[3, 4]], dtype=np.int64),
    )
    # Sample 2 has junction Y = (chr1, 100, 999) — also locally singleton, but
    # shares (chr, start=100, strand=+) with X. After concat they form a start-group.
    s2 = _make_starsolo_dir(
        tmp_path / "s2",
        [("chr1", 100, 999, 1, 1, 1, 6)],
        ["BC_C", "BC_D"],
        np.array([[7, 8]], dtype=np.int64),
    )

    a = splikit.io.read_starsolo(
        [s1, s2], ["s1", "s2"], use_internal_whitelist=False
    )
    # X and Y form one S-group of size 2.
    s_rows = a.var.loc[a.var["group_kind"].astype(str) == "S"]
    assert len(s_rows) == 2
    assert set(s_rows["row_names_mtx"]) == {"chr1:100-200", "chr1:100-999"}
    # No E-group (different ends, no shared end coordinate).
    assert (a.var["group_kind"].astype(str) == "E").sum() == 0
    # group_count is 2 for both.
    assert (s_rows["group_count"] == 2).all()
    # group_id is dense 0..G-1, single group → all 0.
    assert set(s_rows["group_id"]) == {0}
    # Cells from both samples, suffixed correctly.
    assert a.n_obs == 4
    assert sum(name.endswith("-s1") for name in a.obs_names) == 2
    assert sum(name.endswith("-s2") for name in a.obs_names) == 2
    # The X event has cells from s1 (positive) and s2 (zero, padded).
    x_idx = a.var_names.get_loc("chr1:100-200_S")
    M1 = np.asarray(a.layers["M1"][:, x_idx].todense()).ravel()
    s1_mask = np.array([name.endswith("-s1") for name in a.obs_names])
    s2_mask = ~s1_mask
    assert M1[s1_mask].sum() == 7  # 3 + 4
    assert M1[s2_mask].sum() == 0  # padded with zeros


def test_read_starsolo_disjoint_junctions_zero_padded(tmp_path):
    import splikit  # noqa: PLC0415

    s1 = _make_starsolo_dir(
        tmp_path / "s1",
        [("chr1", 100, 200, 1, 1, 1, 5),
         ("chr1", 100, 300, 1, 1, 0, 4)],
        ["BC_A"],
        np.array([[1], [2]], dtype=np.int64),
    )
    s2 = _make_starsolo_dir(
        tmp_path / "s2",
        [("chr2", 5000, 6000, 1, 1, 1, 6),
         ("chr2", 5000, 7000, 1, 1, 0, 7)],
        ["BC_B"],
        np.array([[3], [4]], dtype=np.int64),
    )
    a = splikit.io.read_starsolo(
        [s1, s2], ["s1", "s2"], use_internal_whitelist=False
    )
    # 2 _S events from s1 (chr1) + 2 _S events from s2 (chr2) = 4 total events.
    assert a.n_vars == 4
    # Each event's M1 column must be zero in the cell from the OTHER sample.
    chr1_event = a.var_names[a.var["chr"].astype(str) == "chr1"][0]
    M1_chr1 = np.asarray(
        a.layers["M1"][:, a.var_names.get_loc(chr1_event)].todense()
    ).ravel()
    s1_idx = [i for i, n in enumerate(a.obs_names) if n.endswith("-s1")]
    s2_idx = [i for i, n in enumerate(a.obs_names) if n.endswith("-s2")]
    assert M1_chr1[s2_idx].sum() == 0  # padded zero in s2
    assert M1_chr1[s1_idx].sum() > 0  # populated in s1


def test_read_starsolo_keep_multi_mapped_filter(tmp_path):
    import splikit  # noqa: PLC0415

    junctions = [
        ("chr1", 100, 200, 1, 1, 1, 5),  # unique-mapped > 0
        ("chr1", 100, 300, 1, 1, 0, 0),  # unique-mapped = 0 → dropped by default
        ("chr1", 100, 400, 1, 1, 0, 3),  # unique-mapped > 0
    ]
    barcodes = ["BC1", "BC2"]
    counts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    sj_dir = _make_starsolo_dir(tmp_path / "s1", junctions, barcodes, counts)

    a_default = splikit.io.read_starsolo(sj_dir, "s1", use_internal_whitelist=False)
    # Default keep_multi_mapped=False filters out the second junction; the
    # remaining two share start=100 and form one S-group of size 2.
    assert a_default.n_vars == 2
    assert (a_default.var["group_kind"].astype(str) == "S").sum() == 2

    a_kept = splikit.io.read_starsolo(
        sj_dir, "s1", keep_multi_mapped=True, use_internal_whitelist=False
    )
    # All three junctions share start → S-group of size 3, three _S events.
    assert a_kept.n_vars == 3


def test_read_starsolo_min_counts_filters(tmp_path):
    import splikit  # noqa: PLC0415

    junctions = [
        ("chr1", 100, 200, 1, 1, 1, 5),
        ("chr1", 100, 300, 1, 1, 0, 4),
    ]
    barcodes = ["BC1", "BC2"]
    counts = np.array([[3, 4], [0, 0]], dtype=np.int64)  # second junction all zero
    sj_dir = _make_starsolo_dir(tmp_path / "s1", junctions, barcodes, counts)

    a = splikit.io.read_starsolo(
        sj_dir, "s1", min_counts=1, use_internal_whitelist=False
    )
    # The all-zero junction gets row-sum 0 < 1 → dropped. Only 1 _S row survives.
    assert a.n_vars == 1


def test_read_starsolo_dimension_mismatch_raises(tmp_path):
    import splikit  # noqa: PLC0415

    sj_dir = _make_starsolo_dir(
        tmp_path / "s1",
        [("chr1", 100, 200, 1, 1, 1, 5),
         ("chr1", 100, 300, 1, 1, 0, 4)],
        ["BC1", "BC2"],
        np.array([[1, 2], [3, 4]], dtype=np.int64),
    )
    # Truncate barcodes.tsv by one line to break the cell-axis dim.
    bc_path = sj_dir / "raw" / "barcodes.tsv"
    bc_path.write_text("BC1\n")
    with pytest.raises(ValueError, match="dimension mismatch"):
        splikit.io.read_starsolo(sj_dir, "s1", use_internal_whitelist=False)


def test_read_starsolo_explicit_whitelist(tmp_path):
    import splikit  # noqa: PLC0415

    junctions = [
        ("chr1", 100, 200, 1, 1, 1, 5),
        ("chr1", 100, 300, 1, 1, 0, 4),
    ]
    sj_dir = _make_starsolo_dir(
        tmp_path / "s1", junctions,
        ["KEEP_A", "DROP_B", "KEEP_C"],
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    )
    a = splikit.io.read_starsolo(
        sj_dir, "s1",
        barcode_whitelists=[["KEEP_A", "KEEP_C"]],
        use_internal_whitelist=False,
    )
    assert a.n_obs == 2
    assert sorted(name.split("-")[0] for name in a.obs_names) == ["KEEP_A", "KEEP_C"]


def test_read_starsolo_argument_validation(tmp_path):
    import splikit  # noqa: PLC0415

    sj_dir = _make_starsolo_dir(
        tmp_path / "s1",
        [("chr1", 100, 200, 1, 1, 1, 5),
         ("chr1", 100, 300, 1, 1, 0, 4)],
        ["BC1"], np.array([[1], [2]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="must be unique"):
        splikit.io.read_starsolo([sj_dir, sj_dir], ["s1", "s1"],
                                  use_internal_whitelist=False)
    with pytest.raises(ValueError, match="must equal"):
        splikit.io.read_starsolo([sj_dir], ["s1", "s2"],
                                  use_internal_whitelist=False)
    with pytest.raises(ValueError, match="ljv_kind"):
        splikit.io.read_starsolo(sj_dir, "s1", ljv_kind="bogus",
                                  use_internal_whitelist=False)
    with pytest.raises(ValueError, match="non-negative"):
        splikit.io.read_starsolo(sj_dir, "s1", min_counts=-1,
                                  use_internal_whitelist=False)


def test_read_starsolo_ljv_kind_modes(tmp_path):
    import splikit  # noqa: PLC0415

    junctions = [
        ("chr1", 100, 200, 1, 1, 1, 5),
        ("chr1", 100, 300, 1, 1, 0, 4),  # shares start with #0 → S-group of 2
        ("chr1", 50, 400, 1, 1, 1, 6),
        ("chr1", 70, 400, 1, 1, 0, 7),   # shares end with #2 → E-group of 2
    ]
    sj_dir = _make_starsolo_dir(
        tmp_path / "s1", junctions, ["BC1", "BC2"],
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64),
    )
    a_both = splikit.io.read_starsolo(
        sj_dir, "s1", ljv_kind="start_end", use_internal_whitelist=False
    )
    a_s = splikit.io.read_starsolo(
        sj_dir, "s1", ljv_kind="start", use_internal_whitelist=False
    )
    a_e = splikit.io.read_starsolo(
        sj_dir, "s1", ljv_kind="end", use_internal_whitelist=False
    )
    assert a_both.n_vars == 4
    assert a_s.n_vars == 2 and (a_s.var["group_kind"].astype(str) == "S").all()
    assert a_e.n_vars == 2 and (a_e.var["group_kind"].astype(str) == "E").all()


def test_read_starsolo_end_to_end_pipeline(tmp_path):
    """Full v1.0 pipeline: read_starsolo → make_m2 → highly_variable_events."""
    import splikit  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n_jx = 10
    n_cells_per_sample = 50
    junctions = []
    # Build junctions where most share either a start or end with another so
    # plenty pass LJV filtering.
    for i in range(n_jx):
        start = 1000 + (i // 2) * 100
        end = 2000 + (i % 2) * 100
        junctions.append(("chr1", start, end, 1, 1, 1, int(rng.integers(1, 10))))

    samples = []
    for sid in ["sA", "sB"]:
        bcs = [f"BC{sid}_{i}" for i in range(n_cells_per_sample)]
        # Counts: dense-ish, integer.
        cnts = (rng.random((n_jx, n_cells_per_sample)) * 20).astype(np.int64)
        sj = _make_starsolo_dir(tmp_path / sid, junctions, bcs, cnts)
        samples.append((sj, sid))

    sj_dirs = [s[0] for s in samples]
    sample_ids = [s[1] for s in samples]
    a = splikit.io.read_starsolo(sj_dirs, sample_ids, use_internal_whitelist=False)
    assert a.n_obs == 2 * n_cells_per_sample
    assert a.n_vars > 0

    splikit.tl.make_m2(a, n_threads=1)
    assert "M2" in a.layers
    assert a.uns["splikit"]["m2_valid"] is True

    splikit.pp.highly_variable_events(a, min_row_sum=1, n_threads=1)
    assert "sum_deviance" in a.var.columns
    finite = np.isfinite(a.var["sum_deviance"]).sum()
    assert finite > 0
