"""Compare R vs Python E2E pipeline outputs on samples A01 + B01.

Loads ``data/r_pipeline.h5`` and ``data/py_pipeline.h5ad``, aligns events on
``var["row_names_mtx"]`` and obs on ``obs_names``, and asserts:

* M1 bit-exact (``np.array_equal`` on CSC ``indptr`` / ``indices`` / ``data``).
* M2 bit-exact (kernel is documented bit-exact vs R splikit::make_m2).
* eventdata fields equal (``chr/start/end/strand/group_id/group_kind/group_count``).

On any mismatch, prints a diagnostic (first N differing rows, max abs diff,
fraction of disagreement) and exits non-zero.

Tolerance choice: bit-exact across the board. M1 is a sparse re-encoding of
STARsolo SJ.out raw counts (integer values stored as float64); the LJV
grouping is deterministic on (chr, start, strand) / (chr, end, strand)
tuples; the make_m2 kernel is column-disjoint with thread-local workspaces
and explicit per-column std::sort. There is no source of floating-point
drift in this pipeline.

Run:
    python validation/e2e/compare_outputs.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent
R_PATH = HERE / "data" / "r_pipeline.h5"
PY_PATH = HERE / "data" / "py_pipeline.h5ad"


def _decode(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "S" or arr.dtype == object:
        return np.array(
            [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
        )
    return arr.astype(str)


def _read_csc(group: h5py.Group) -> sp.csc_matrix:
    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    indices = np.asarray(group["indices"][:], dtype=np.int64)
    data = np.asarray(group["data"][:], dtype=np.float64)
    shape = tuple(int(x) for x in group["shape"][:])
    return sp.csc_matrix((data, indices, indptr), shape=shape)


def _diagnostic(name: str, py: np.ndarray, r: np.ndarray, k: int = 5) -> str:
    if py.dtype.kind in "fc" and r.dtype.kind in "fc":
        diff = np.abs(py - r)
        idx = np.argsort(-diff)[:k]
        return (f"{name}: first {k} largest abs diffs at "
                f"{idx.tolist()}; max={float(diff.max()):.6e}; "
                f"py[idx]={py[idx]}, r[idx]={r[idx]}")
    diff_idx = np.where(py != r)[0][:k]
    n_diff = int((py != r).sum())
    return (f"{name}: {n_diff}/{len(py)} entries differ; first {k} at "
            f"{diff_idx.tolist()}; py={py[diff_idx]}, r={r[diff_idx]}")


def main() -> int:
    if not R_PATH.exists():
        print(f"[ERROR] R reference not found at {R_PATH}\n"
              "  Run: ssh c170 'module load r/4.4.0 && Rscript "
              "validation/e2e/run_r_pipeline.R' first.", file=sys.stderr)
        return 2
    if not PY_PATH.exists():
        print(f"[ERROR] Python output not found at {PY_PATH}\n"
              "  Run: python validation/e2e/run_py_pipeline.py first.",
              file=sys.stderr)
        return 2

    print(f"[compare] loading R reference: {R_PATH}", flush=True)
    with h5py.File(R_PATH, "r") as f:
        r_m1 = _read_csc(f["m1"])
        r_m2 = _read_csc(f["m2"])
        r_event_id = _decode(f["eventdata"]["row_names_mtx"][:])
        r_group_id = np.asarray(f["eventdata"]["group_id"][:], dtype=np.int32)
        r_group_kind = _decode(f["eventdata"]["group_kind"][:])
        r_group_count = np.asarray(f["eventdata"]["group_count"][:], dtype=np.int32)
        r_obs_names = _decode(f["obs"]["obs_names"][:])
        r_sample_id = _decode(f["obs"]["sample_id"][:])

    print(f"[compare]   r_m1: {r_m1.shape} nnz={r_m1.nnz:,}", flush=True)
    print(f"[compare]   r_m2: {r_m2.shape} nnz={r_m2.nnz:,}", flush=True)

    print(f"[compare] loading Py output: {PY_PATH}", flush=True)
    py_adata = ad.read_h5ad(PY_PATH)
    print(f"[compare]   py adata: {py_adata.n_obs} x {py_adata.n_vars}", flush=True)

    # Align var on (row_names_mtx_unsuffixed, group_kind). Schema note: R splikit
    # stores row_names_mtx WITH the _S/_E suffix (legacy from R make_m1 ~line 378
    # which appends "_S"/"_E" to the un-suffixed junction id). splikit-py stores
    # the un-suffixed form per the documented schema. Strip the suffix on the R
    # side for the comparison key; same junction, same coordinates.
    def _strip_suffix(name: str) -> str:
        return name[:-2] if name.endswith(("_S", "_E")) else name

    py_bare = np.asarray(py_adata.var["row_names_mtx"]).astype(str)
    py_kind = np.asarray(py_adata.var["group_kind"]).astype(str)
    r_bare = np.array([_strip_suffix(n) for n in r_event_id])
    r_kind = np.asarray(r_group_kind).astype(str)
    py_keys = np.array([f"{rn}|{kk}" for rn, kk in zip(py_bare, py_kind)])
    r_keys = np.array([f"{rn}|{kk}" for rn, kk in zip(r_bare, r_kind)])

    py_idx = pd.Index(py_keys)
    if not py_idx.is_unique:
        print(f"[compare] FAIL: py var keys are not unique "
              f"(duplicates: {(py_idx.duplicated()).sum()})", file=sys.stderr)
        return 3
    r_idx = pd.Index(r_keys)
    if not r_idx.is_unique:
        print(f"[compare] FAIL: r var keys are not unique "
              f"(duplicates: {(r_idx.duplicated()).sum()})", file=sys.stderr)
        return 3

    # Set membership.
    only_py = set(py_keys) - set(r_keys)
    only_r = set(r_keys) - set(py_keys)
    common = set(py_keys) & set(r_keys)
    print(f"[compare] events: {len(common)} common, "
          f"{len(only_py)} py-only, {len(only_r)} r-only", flush=True)
    if only_py or only_r:
        sample_py = list(only_py)[:5]
        sample_r = list(only_r)[:5]
        print(f"[compare]   py-only sample: {sample_py}", flush=True)
        print(f"[compare]   r-only sample:  {sample_r}", flush=True)

    if not common:
        print("[compare] FAIL: no common events.", file=sys.stderr)
        return 4

    # Reindex both M1 and M2 onto the common ordering (lex sort of common).
    common_keys = sorted(common)
    py_to_common = py_idx.get_indexer(common_keys)
    r_to_common = r_idx.get_indexer(common_keys)

    # obs alignment.
    py_obs_names = np.asarray(py_adata.obs_names)
    py_obs_idx = pd.Index(py_obs_names)
    r_obs_idx = pd.Index(r_obs_names)
    common_obs = sorted(set(py_obs_names) & set(r_obs_names))
    only_py_obs = set(py_obs_names) - set(r_obs_names)
    only_r_obs = set(r_obs_names) - set(py_obs_names)
    print(f"[compare] cells: {len(common_obs)} common, "
          f"{len(only_py_obs)} py-only, {len(only_r_obs)} r-only", flush=True)
    if not common_obs:
        print("[compare] FAIL: no common cells.", file=sys.stderr)
        return 4

    py_obs_take = py_obs_idx.get_indexer(common_obs)
    r_obs_take = r_obs_idx.get_indexer(common_obs)

    # Slice both M1 and M2 on common (events x cells) ordering.
    # py is cells x events (AnnData layout); r is events x cells.
    py_M1 = sp.csc_matrix(py_adata.layers["M1"][py_obs_take, :][:, py_to_common]).T.tocsc()
    r_M1 = sp.csc_matrix(r_m1[:, r_obs_take][r_to_common, :]).tocsc()
    py_M2 = sp.csc_matrix(py_adata.layers["M2"][py_obs_take, :][:, py_to_common]).T.tocsc()
    r_M2 = sp.csc_matrix(r_m2[:, r_obs_take][r_to_common, :]).tocsc()

    # Sort indices for canonical CSC compare.
    py_M1.sort_indices()
    r_M1.sort_indices()
    py_M2.sort_indices()
    r_M2.sort_indices()

    failures: list[str] = []
    successes: list[str] = []

    def cmp_csc(name: str, a: sp.csc_matrix, b: sp.csc_matrix) -> None:
        if not (a.shape == b.shape):
            failures.append(f"{name}: shape mismatch py={a.shape} r={b.shape}")
            return
        ok_indptr = np.array_equal(a.indptr, b.indptr)
        ok_indices = np.array_equal(a.indices, b.indices)
        ok_data = np.array_equal(a.data, b.data)
        if ok_indptr and ok_indices and ok_data:
            successes.append(f"{name}: bit-exact (nnz={a.nnz:,})")
        else:
            parts = []
            if not ok_indptr:
                parts.append("indptr differs")
            if not ok_indices:
                parts.append("indices differs")
            if not ok_data:
                diff = np.abs(a.data - b.data)
                parts.append(
                    f"data differs (max abs {float(diff.max()):.6e})"
                )
            failures.append(f"{name}: " + "; ".join(parts))

    cmp_csc("M1", py_M1, r_M1)
    cmp_csc("M2", py_M2, r_M2)

    # Eventdata equality.
    py_var_aligned = py_adata.var.iloc[py_to_common]
    r_event_aligned = r_bare[r_to_common]  # un-suffixed; matches schema
    r_kind_aligned = r_group_kind[r_to_common]
    r_count_aligned = r_group_count[r_to_common]
    r_gid_aligned = r_group_id[r_to_common]

    if not np.array_equal(
        np.asarray(py_var_aligned["row_names_mtx"]).astype(str), r_event_aligned
    ):
        failures.append(_diagnostic(
            "row_names_mtx",
            np.asarray(py_var_aligned["row_names_mtx"]).astype(str),
            r_event_aligned,
        ))
    else:
        successes.append("row_names_mtx: equal")

    if not np.array_equal(
        np.asarray(py_var_aligned["group_kind"]).astype(str), r_kind_aligned
    ):
        failures.append(_diagnostic(
            "group_kind",
            np.asarray(py_var_aligned["group_kind"]).astype(str),
            r_kind_aligned,
        ))
    else:
        successes.append("group_kind: equal")

    if not np.array_equal(
        np.asarray(py_var_aligned["group_count"]).astype(np.int32), r_count_aligned
    ):
        failures.append(_diagnostic(
            "group_count",
            np.asarray(py_var_aligned["group_count"]).astype(np.int32),
            r_count_aligned,
        ))
    else:
        successes.append("group_count: equal")

    # group_id: both sides remap to dense 0..G-1, but the *labels* are
    # arbitrary (factorize order depends on iteration order); compare the
    # induced equivalence classes instead.
    py_gid = np.asarray(py_var_aligned["group_id"]).astype(np.int32)
    py_partition = pd.Series(py_gid).groupby(py_gid).indices
    r_partition = pd.Series(r_gid_aligned).groupby(r_gid_aligned).indices
    py_blocks = sorted(tuple(sorted(v.tolist())) for v in py_partition.values())
    r_blocks = sorted(tuple(sorted(v.tolist())) for v in r_partition.values())
    if py_blocks == r_blocks:
        successes.append(f"group_id: equivalence classes match ({len(py_blocks)} groups)")
    else:
        failures.append(
            f"group_id: equivalence classes differ "
            f"({len(py_blocks)} py groups vs {len(r_blocks)} r groups)"
        )

    # obs sample_id consistency on common cells.
    py_sid = np.asarray(py_adata.obs["sample_id"].astype(str))[py_obs_take]
    r_sid_aligned = r_sample_id[r_obs_take]
    if not np.array_equal(py_sid, r_sid_aligned):
        failures.append(_diagnostic("sample_id", py_sid, r_sid_aligned))
    else:
        successes.append("sample_id: equal")

    # Final report.
    print()
    print("=" * 72)
    print("E2E COMPARISON RESULT")
    print("=" * 72)
    for s in successes:
        print(f"  ✓ {s}")
    for f in failures:
        print(f"  ✗ {f}")
    print("=" * 72)

    if only_py or only_r or only_py_obs or only_r_obs:
        print()
        print(f"NOTE: structural set differences exist "
              f"(py-only events: {len(only_py)}, r-only events: {len(only_r)}, "
              f"py-only cells: {len(only_py_obs)}, r-only cells: {len(only_r_obs)}). "
              "These need investigation before the bit-exact gate is meaningful.")

    if failures:
        return 1
    if only_py or only_r or only_py_obs or only_r_obs:
        return 1
    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
