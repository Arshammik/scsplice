"""Loader for the R-exported HDF5 reference fixture.

Mirrors the pattern in multigedipy_pkg/tests/load_r_ref.py: a single function
returning a typed dataclass so individual test files don't re-read the file or
hand-roll the CSC reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class RReference:
    """A frozen snapshot of R splikit's reference outputs on the toy dataset."""

    m1: sp.csc_matrix
    m2: sp.csc_matrix
    group_id: np.ndarray
    row_names_mtx: np.ndarray
    group_kind: np.ndarray
    group_count: np.ndarray
    chr_arr: np.ndarray | None
    start: np.ndarray | None
    end: np.ndarray | None
    strand: np.ndarray | None
    splikit_version: str
    r_version: str
    generated_at: str
    blas_vendor: str


def _decode_strings(arr: np.ndarray) -> np.ndarray:
    """Convert HDF5 byte strings to Python str."""
    if arr.dtype.kind == "S" or arr.dtype == object:
        return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x)
                         for x in arr])
    return arr.astype(str)


def _read_csc(group: h5py.Group) -> sp.csc_matrix:
    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    indices = np.asarray(group["indices"][:], dtype=np.int64)
    data = np.asarray(group["data"][:], dtype=np.float64)
    shape = tuple(int(x) for x in group["shape"][:])
    return sp.csc_matrix((data, indices, indptr), shape=shape)


def _read_attr(f: h5py.File, name: str) -> str:
    if name not in f.attrs:
        return ""
    val = f.attrs[name]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray) and val.size == 1:
        v = val.item()
        return v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return str(val)


def _maybe_array(group: h5py.Group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    arr = group[name][:]
    if arr.dtype.kind == "S" or arr.dtype == object:
        return _decode_strings(arr)
    return np.asarray(arr)


def load_reference(path: str | Path) -> RReference:
    """Load ``tests/data/r_reference.h5`` (or any compatible file)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"R reference fixture not found: {path}\n"
            "Regenerate via: Rscript tests/r_export/export_reference.R"
        )

    with h5py.File(path, "r") as f:
        m1 = _read_csc(f["m1"])
        m2 = _read_csc(f["m2"])
        ed = f["eventdata"]
        group_id = np.asarray(ed["group_id"][:], dtype=np.int32)
        row_names_mtx = _decode_strings(ed["row_names_mtx"][:])
        group_kind = _decode_strings(ed["group_kind"][:])
        group_count = np.asarray(ed["group_count"][:], dtype=np.int32)
        chr_arr = _maybe_array(ed, "chr")
        start = _maybe_array(ed, "start")
        end = _maybe_array(ed, "end")
        strand = _maybe_array(ed, "strand")
        return RReference(
            m1=m1,
            m2=m2,
            group_id=group_id,
            row_names_mtx=row_names_mtx,
            group_kind=group_kind,
            group_count=group_count,
            chr_arr=chr_arr,
            start=start,
            end=end,
            strand=strand,
            splikit_version=_read_attr(f, "splikit_version"),
            r_version=_read_attr(f, "r_version"),
            generated_at=_read_attr(f, "generated_at"),
            blas_vendor=_read_attr(f, "blas_vendor"),
        )
