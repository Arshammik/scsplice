"""Loader for the R-exported HDF5 reference fixture.

Mirrors the pattern in multigedipy_pkg/tests/load_r_ref.py: a single function
returning a typed dict so individual test files don't re-read the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py


def load_reference(path: str | Path) -> dict[str, Any]:
    """Load tests/data/r_reference.h5 into a nested dict.

    Returns
    -------
    A dict with the schema documented in tests/r_export/export_reference.R.
    Sparse matrices come back as four-tuples (indptr, indices, data, shape)
    so callers can rebuild scipy.sparse.csc_matrix without forcing this loader
    to depend on scipy.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference fixture not found: {path}\n"
            "Regenerate via: Rscript tests/r_export/export_reference.R"
        )

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for attr in ("splikit_version", "r_version", "generated_at"):
            if attr in f.attrs:
                out[attr] = f.attrs[attr]
        # Concrete schema fills in once export_reference.R is implemented.
    return out
