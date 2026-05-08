"""Schema invariants for the splicing AnnData."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from splikit._core._validators import (
    invalidate_m2,
    mark_m2_valid,
    validate_m1_layer,
    validate_paired_layers,
    validate_var_schema,
)


def _make_adata(n_obs: int = 10, n_var: int = 4, *, with_m2: bool = True,
                kind: str = "S") -> ad.AnnData:
    rng = np.random.default_rng(0)
    M1 = sp.random(n_obs, n_var, density=0.4, format="csc", dtype=np.float64,
                   random_state=rng).astype(np.float64)
    var = pd.DataFrame(
        {
            "chr": pd.Categorical(["chr1"] * n_var),
            "start": np.arange(n_var, dtype=np.int64) * 100,
            "end": np.arange(n_var, dtype=np.int64) * 100 + 50,
            "strand": pd.Categorical(["+"] * n_var),
            "row_names_mtx": [f"chr1:{i*100}-{i*100+50}" for i in range(n_var)],
            "group_id": np.zeros(n_var, dtype=np.int32),
            "group_kind": pd.Categorical([kind] * n_var, categories=["S", "E"]),
            "group_count": np.full(n_var, 4, dtype=np.int32),
        },
        index=[f"chr1:{i*100}-{i*100+50}_{kind}" for i in range(n_var)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"bc{i}" for i in range(n_obs)], "sample_id": ["s1"] * n_obs},
        index=[f"bc{i}-s1" for i in range(n_obs)],
    )
    layers = {"M1": M1}
    if with_m2:
        layers["M2"] = M1.copy()
    a = ad.AnnData(layers=layers, obs=obs, var=var)
    if with_m2:
        a.uns["splikit"] = {"m2_valid": True}
    return a


def test_validate_m1_layer_ok():
    validate_m1_layer(_make_adata())


def test_validate_m1_layer_missing():
    a = _make_adata()
    del a.layers["M1"]
    with pytest.raises(KeyError, match="M1"):
        validate_m1_layer(a)


def test_validate_m1_layer_dense_rejected():
    a = _make_adata()
    a.layers["M1"] = a.layers["M1"].toarray()
    with pytest.raises(TypeError, match="scipy.sparse"):
        validate_m1_layer(a)


def test_validate_m1_layer_dtype():
    a = _make_adata()
    a.layers["M1"] = a.layers["M1"].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        validate_m1_layer(a)


def test_validate_paired_layers_ok():
    validate_paired_layers(_make_adata())


def test_validate_paired_layers_missing_m2():
    a = _make_adata(with_m2=False)
    with pytest.raises(KeyError, match="M2"):
        validate_paired_layers(a)


def test_validate_paired_layers_stale_m2():
    a = _make_adata()
    invalidate_m2(a)
    with pytest.raises(RuntimeError, match="m2_valid"):
        validate_paired_layers(a)


def test_validate_paired_layers_skip_m2_valid():
    a = _make_adata()
    invalidate_m2(a)
    validate_paired_layers(a, require_m2_valid=False)


def test_validate_var_schema_ok():
    validate_var_schema(_make_adata())


def test_validate_var_schema_missing_col():
    a = _make_adata()
    a.var = a.var.drop(columns=["group_id"])
    with pytest.raises(KeyError, match="group_id"):
        validate_var_schema(a)


def test_validate_var_schema_duplicate_var_names_rejected():
    a = _make_adata(n_var=3)
    a.var.index = pd.Index(["x_S", "x_S", "y_E"])
    with pytest.raises(ValueError, match="unique"):
        validate_var_schema(a)


def test_validate_var_schema_bad_group_kind():
    a = _make_adata()
    a.var["group_kind"] = pd.Categorical(["X"] * len(a.var), categories=["S", "E", "X"])
    with pytest.raises(ValueError, match="group_kind"):
        validate_var_schema(a)


def test_mark_invalidate_roundtrip():
    a = _make_adata()
    invalidate_m2(a)
    assert a.uns["splikit"]["m2_valid"] is False
    mark_m2_valid(a)
    assert a.uns["splikit"]["m2_valid"] is True
