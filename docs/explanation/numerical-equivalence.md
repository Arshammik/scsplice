# Numerical equivalence with R splikit

`splikit-py` reproduces the results of the R package `splikit` to documented tolerance bands on a fixed reference dataset. This page explains what "equivalent" means for each function, why some tolerance is necessary, and how to regenerate the reference fixtures.

## Reference fixtures

Cross-language tests use two HDF5 fixtures. The protocol is documented in [`tests/r_export/README.md`](https://github.com/Arshammik/splikitpy/blob/main/tests/r_export/README.md).

| Fixture | Size | Committed | Coverage |
|---|---|---|---|
| `tests/data/r_reference.h5` | ~28 MB (toy) | Yes | `test_make_m2`, `test_highly_variable_events`, `test_pseudo_correlation` |
| `tests/data/r_reference_merfish.h5` | ~100 MB (real) | No (.gitignore) | `tests/test_r_equivalence_real.py` |

The toy fixture uses `splikit::load_toy_M1_M2_object()` (2000 events × 2000 cells). The MERFISH fixture uses a real single-sample slice of a trained MultiGEDI model (12,774 events × 835 cells).

## Tolerance by function

### `make_m2` — bit-exact

M2 is **bit-identical** to R's `make_m2(use_cpp=TRUE)`. Both implementations use the same two-pass CSC algorithm (sum columns by group, subtract self), carried out in float64 with no rounding in between. The test assertion is:

```python
np.array_equal(py_m2, r_m2)  # exact equality, no tolerance
```

This guarantee holds across all thread counts (`n_threads=1` through `n_threads=16`); the kernel produces the same output regardless.

### `highly_variable_events` — near-exact

Per-library binomial deviance agrees to:

```python
np.allclose(py_dev, r_dev, rtol=1e-10, atol=1e-12)
```

plus an exact NaN-mask match (events filtered out by `min_row_sum` receive `NaN` in both R and Python, on the same set of events).

The small residual arises from different BLAS implementations for the intermediate matrix multiplications (OpenBLAS on Linux CI vs. Apple Accelerate on macOS, vs. R's compiled LAPACK). The per-library deviance formula involves only additions and logarithms on scalars so the difference is well within IEEE 754 rounding.

### `pseudo_correlation` — IRLS tolerance band

Pseudo-correlation agrees to:

```python
np.allclose(py_rho, r_rho, rtol=1e-7, atol=1e-9)
```

plus an exact NaN-mask match.

The looser tolerance reflects accumulated floating-point divergence in the iteratively reweighted least squares (IRLS) loop. R and Python use the same algorithm (logistic GLM, 25-iteration IRLS, same convergence criterion), but different matrix routines for the Hessian solve (`qr.coef` in R vs. Eigen's `fullPivHouseholderQr` in C++). Events where the two implementations disagree by more than `1e-7` are flagged as test failures; in practice all events on the reference dataset agree to better than `1e-8`.

**Permutation null distributions are not cross-language equivalent.** R uses Mersenne Twister for its RNG; Python uses PCG64. Column permutations produced by `set.seed(k)` in R differ from those produced by `np.random.default_rng(k)` in Python. Within-language reproducibility is guaranteed; cross-language bit-parity of null distributions is not.

## Regenerating the toy fixture

The toy fixture is committed and should regenerate cleanly from a machine with R 4.1+ and the R `splikit` package installed:

```bash
module load r/4.4.0   # or your local R installation
Rscript tests/r_export/export_reference.R tests/data/r_reference.h5
```

## Regenerating the MERFISH fixture

The MERFISH fixture requires access to the cluster RDS files and is not committed. See [`tests/r_export/README.md`](https://github.com/Arshammik/splikitpy/blob/main/tests/r_export/README.md) for the full protocol.

## Running the equivalence tests

```bash
# Toy fixture (fast, committed, runs in CI)
pytest tests/ -m r_required -v

# MERFISH fixture (requires regeneration; skipped when fixture is absent)
pytest tests/test_r_equivalence_real.py -v
```
