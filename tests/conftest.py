"""Shared pytest fixtures for scsplice tests."""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# Real-data sandbox path; tests gated by @pytest.mark.real_data are
# auto-skipped when the path does not exist. Override with the env var
# ``SPLIKIT_REAL_DATA_DIR`` to point to a real STARsolo output directory.
_REAL_DATA_DIR_STR = os.environ.get("SPLIKIT_REAL_DATA_DIR")
_REAL_DATA_DIR = Path(_REAL_DATA_DIR_STR) if _REAL_DATA_DIR_STR else None
_REAL_DATA_SAMPLES = (
    "L8TX_181211_01_A01_S01_L003",
    "L8TX_181211_01_B01_S01_L003",
)


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    if _REAL_DATA_DIR is None or not _REAL_DATA_DIR.is_dir():
        pytest.skip(
            f"Real-data sandbox not found; set SPLIKIT_REAL_DATA_DIR env var "
            "to enable @pytest.mark.real_data tests."
        )
    return _REAL_DATA_DIR


@pytest.fixture(scope="session")
def real_data_samples(real_data_dir: Path) -> list[Path]:
    paths = [real_data_dir / s for s in _REAL_DATA_SAMPLES]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        pytest.skip(f"Real-data samples missing: {missing}")
    return paths


@pytest.fixture(scope="session")
def real_data_sample_ids() -> list[str]:
    return list(_REAL_DATA_SAMPLES)


@pytest.fixture
def synthetic_splicing_adata():
    """Factory fixture: build a tiny well-formed splicing AnnData on demand.

    Usage:
        adata = synthetic_splicing_adata(n_events=20, n_cells=50, n_groups=4)
    """
    def _make(
        n_events: int = 20,
        n_cells: int = 50,
        n_groups: int = 4,
        *,
        density: float = 0.3,
        seed: int = 0,
        kind: str = "S",
    ) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        M1 = sp.random(
            n_cells, n_events, density=density, format="csc",
            dtype=np.float64, random_state=rng,
            data_rvs=lambda n: rng.integers(1, 20, size=n).astype(np.float64),
        ).astype(np.float64)

        # Round-robin assign events to groups so every group has >= 1 member.
        group_id = np.arange(n_events, dtype=np.int32) % int(n_groups)
        group_count = np.bincount(group_id, minlength=int(n_groups))[group_id].astype(
            np.int32
        )

        var = pd.DataFrame(
            {
                "chr": pd.Categorical(["chr1"] * n_events),
                "start": np.arange(n_events, dtype=np.int64) * 100,
                "end": np.arange(n_events, dtype=np.int64) * 100 + 50,
                "strand": pd.Categorical(["+"] * n_events),
                "row_names_mtx": [f"chr1:{i*100}-{i*100+50}" for i in range(n_events)],
                "group_id": group_id,
                "group_kind": pd.Categorical(
                    [kind] * n_events, categories=["S", "E"]
                ),
                "group_count": group_count,
            },
            index=[f"chr1:{i*100}-{i*100+50}_{kind}" for i in range(n_events)],
        )
        obs = pd.DataFrame(
            {"barcode": [f"bc{i}" for i in range(n_cells)],
             "sample_id": ["s1"] * n_cells},
            index=[f"bc{i}-s1" for i in range(n_cells)],
        )
        a = ad.AnnData(layers={"M1": M1}, obs=obs, var=var)
        a.uns["scsplice"] = {"m2_valid": False}
        return a

    return _make
