"""Per-sample splitting helper."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scsplice._core._split import split_paired_by_sample


def _make_adata(samples: list[str], n_var: int = 3) -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_obs = len(samples)
    M1 = sp.random(n_obs, n_var, density=0.6, format="csc", dtype=np.float64,
                   random_state=rng).astype(np.float64)
    M2 = M1.copy()
    obs = pd.DataFrame(
        {"sample_id": samples},
        index=[f"bc{i}-{s}" for i, s in enumerate(samples)],
    )
    var = pd.DataFrame({"id": np.arange(n_var)}, index=[f"v{i}_S" for i in range(n_var)])
    return ad.AnnData(layers={"M1": M1, "M2": M2}, obs=obs, var=var)


def test_split_returns_per_sample_lists():
    a = _make_adata(["s1", "s2", "s1", "s2", "s2"])
    M1i, M2i, sids = split_paired_by_sample(a)
    assert sids == ["s1", "s2"]
    assert len(M1i) == 2 == len(M2i)
    assert M1i[0].shape == (2, 3)
    assert M1i[1].shape == (3, 3)


def test_split_preserves_categorical_order():
    a = _make_adata(["s2", "s1", "s2", "s1"])
    a.obs["sample_id"] = pd.Categorical(a.obs["sample_id"], categories=["s1", "s2"])
    _, _, sids = split_paired_by_sample(a)
    assert sids == ["s1", "s2"]


def test_split_first_seen_order_when_not_categorical():
    a = _make_adata(["s2", "s1", "s2"])
    _, _, sids = split_paired_by_sample(a)
    assert sids == ["s2", "s1"]


def test_split_missing_sample_key():
    a = _make_adata(["s1"])
    a.obs = a.obs.drop(columns=["sample_id"])
    with pytest.raises(KeyError, match="sample_id"):
        split_paired_by_sample(a)


def test_split_round_trip_dtype_and_format():
    a = _make_adata(["s1", "s1"])
    M1i, M2i, _ = split_paired_by_sample(a)
    for M in [*M1i, *M2i]:
        assert M.dtype == np.float64
        assert sp.issparse(M)
        assert M.format == "csc"


def test_split_single_sample():
    a = _make_adata(["s1", "s1", "s1"])
    M1i, M2i, sids = split_paired_by_sample(a)
    assert sids == ["s1"]
    assert len(M1i) == 1
    assert M1i[0].shape == (3, 3)
