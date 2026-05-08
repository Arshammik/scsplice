# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package overview

`splikit-py` is a Python port of the R/Rcpp package [splikit](https://github.com/csglab/splikit) for alternative-splicing analysis in single-cell RNA-seq. It is **scverse-compatible** (AnnData-native, scanpy `pp/tl/pl/io` namespacing, free functions only) and aims for **numerical equivalence with R splikit** on a fixed reference dataset.

Greenfield rewrite — not a faithful Cython port. Numerical equivalence is verified by golden-file regression, not by byte-for-byte algorithmic mimicry.

## v1.0 scope (intentionally narrow)

Three function families only:

- `splk.io.read_starsolo` — STARsolo `Solo.out/SJ/` ingestion for one or more samples (replaces R `make_junction_ab` + `make_m1`).
- `splk.tl.make_m2` — exclusion-matrix builder from M1 + LJV `group_id` (replaces R `make_m2`; ports `splikit/src/make_m2_cpp.cpp`).
- `splk.pp.highly_variable_events` — per-library binomial-deviance HVE selection (replaces R `find_variable_events`; ports `splikit/src/calcDeviances.cpp`).
- `splk.tl.pseudo_correlation` — beta-binomial Cox-Snell / Nagelkerke pseudo-R² via IRLS (replaces R `get_pseudo_correlation`; ports `splikit/src/cpp_pseudoR2.cpp`, **collapses the four R dispatch wrappers into one templated kernel**).

**Out of v1.0**: HVG (`find_variable_genes`), silhouette, row variance, GTF plotting, gene/velo counts. Use `scanpy`, `sklearn`, `pyranges` — they cover this ground. The R6 `SplikitObject` class is dropped entirely.

## Architecture

### AnnData schema (load-bearing)

- `X = None`. **Both M1 and M2 live in `layers`** (`layers["M1"]`, `layers["M2"]`). Mirrors scvelo's spliced/unspliced precedent. Putting M1 in `X` invites `sc.pp.normalize_total` to mutate inclusion counts and silently break `m2_valid`.
- `var` required columns: `chr`, `start`, `end`, `strand`, `row_names_mtx` (un-suffixed `chr:start-end`), `group_id` (int32, dense `0..G-1`), `group_kind` (categorical `"S"`/`"E"`), `group_count` (int32). Optional: `gene_id`, `gene_name`.
- `var_names` invariant: `chr:start-end_S` or `chr:start-end_E`. **`var_names_make_unique()` is forbidden** — it would silently mangle the suffix scheme. Two var rows can share `var["row_names_mtx"]` but never `adata.var_names`. Asserted at ingestion.
- `obs` required columns: `barcode`, `sample_id`. The R-side regex `sub("^.{16}-(.*$)", "\\1", brc)` for splitting libraries is replaced by an explicit `obs["sample_id"]` populated at ingestion time.
- `uns["splikit"]`: `version`, `m2_valid: bool`, `ljv_kind`, `params` (last-call settings, scanpy idiom).
- Per-event scores live in `var["sum_deviance"]`, `var["highly_variable"]`, `var["pseudo_correlation"]`. Permutation null distributions go in `varm["pseudo_R2_null_dist"]` (n_var × n_perm) — `varm`, not `uns`, because it must subset with `adata[:, mask]`.
- **Structural-alignment contract**: `layers["M1"]` and `layers["M2"]` must both be `scipy.sparse.csc_matrix`, `float64`, identical shape `(n_obs, n_var)`. Enforced by `_validate_paired_layers` at the top of every `tl`/`pp` function. After any var-axis subset, `m2_valid` flips to `False` (LJV co-membership changes invalidate M2).

### Public API mapping

| R | Python |
|---|---|
| `make_junction_ab` + `make_m1` | `splk.io.read_starsolo(sj_dirs, sample_ids, ...)` |
| `make_m2` / `obj$makeM2()` | `splk.tl.make_m2(adata, *, n_threads=1, copy=False)` |
| `find_variable_events` | `splk.pp.highly_variable_events(adata, *, min_row_sum=50, sample_key="sample_id", ...)` |
| `get_pseudo_correlation` | `splk.tl.pseudo_correlation(adata, zdb, *, metric="CoxSnell", n_permutations=0, ...)` |

All functions take `adata` first-positional, mutate in place by default, and accept `copy=False` per scanpy convention. The R6 chain `obj$makeM2()$findVariableEvents()` becomes the imperative sequence `splk.tl.make_m2(adata); splk.pp.highly_variable_events(adata)`.

### C++ / pybind11 boundary

We mirror the pattern in `/project/6007998/arsham79/multigedipy_pkg/` and `/project/6007998/arsham79/gedi2py/` — both reference packages already solve the dual-matrix problem with the same stack.

- Build: `scikit-build-core>=0.5` + `pybind11>=2.11` + `Eigen3` + optional OpenMP. R-matching compile flags `-march=nocona -mtune=haswell -ftree-vectorize` so floating-point reduction order matches R's CRAN baseline.
- Per-sample splitting in Python (`splikit/_core/_split.py`, near-copy of `multigedipy_pkg/src/multigedipy/_core/_multi_model.py:96-110`). C++ receives `std::vector<Eigen::SparseMatrix<double>>` for M1i and M2i.
- One C++ class `SplikitKernel` in `src/_splikit_cpp/splikit_kernel.{hpp,cpp}` with `set_M1i_M2i(M1i, M2i)`, `find_variable_events(min_row_sum)`, `pseudo_correlation(zdb, metric)` plus a static `make_m2(M1, group_ids, n_threads)`. All long methods released via `py::call_guard<py::gil_scoped_release>()` (see `multigedipy_pkg/src/_multigedipy_cpp/bindings.cpp:91-107`).
- The four R wrappers `cppBetabinPseudoR2{,_sparse,_mixed1,_mixed2}` (in `splikit/src/cpp_pseudoR2.cpp:248-280`) collapse to one templated Eigen kernel; Python promotes dense to sparse before crossing the boundary because layers are always sparse.
- Determinism: outer-loop-only OpenMP, **thread-local workspaces with disjoint output writes** (no `#pragma omp critical`, no `reduction(+:)`). Already the pattern in `splikit/src/make_m2_cpp.cpp:70-103,191-195` and `splikit/src/calcDeviances.cpp:14-53` — port verbatim. This is what makes the kernel bit-identical regardless of `n_threads`.

### Numerical-equivalence regression strategy

- Reference dataset: the R splikit toy at `/project/6007998/arsham79/splikit/inst/extdata/toy_m1_m2_obj.rds` (~2000 events × 2000 cells).
- Export script: `tests/r_export/export_reference.R` runs `make_m2` + `find_variable_events` + `get_pseudo_correlation` and writes everything (M1, M2, deviance vector, pseudo-R² vector, **the Z draw**) to `tests/data/r_reference.h5` via `rhdf5`. **Z is exported, never RNG-regenerated on the Python side** (R Mersenne-Twister and numpy PCG64 are not bit-comparable).
- Python loader at `tests/load_r_ref.py` (mirror of `multigedipy_pkg/tests/load_r_ref.py`).
- Tolerance bands:
  - `make_m2` `indptr`/`indices`/`data` — exact (`np.array_equal`, atol=0). Justified by deterministic thread-local writes.
  - `find_variable_events` deviance — `np.allclose(rtol=1e-10, atol=1e-12)`. Per-row independent reductions; no cross-thread accumulation.
  - `pseudo_correlation` (fixed Z) — `np.allclose(rtol=1e-9, atol=1e-12)`. Widened because IRLS path can drift by 1 iter near `tol=1e-6`.
  - Permutation null distribution — KS-test `p > 0.001` + mean within 3σ.
- CI matrix: pure-Python jobs on Py 3.10/3.11/3.12 × {ubuntu, macos} for the bulk of the suite; one separate `r_required` job (gated by `pytest.mark.r_required`) regenerates the H5 fixture from the latest CRAN/GitHub R splikit.

### Numerical-equivalence traps to replicate verbatim

These constants and patterns from the R/C++ source are load-bearing and must port unchanged:

- `EPS=1e-8` probability clamp (`splikit/src/cpp_pseudoR2.cpp:7`). Never replace with `np.finfo(float).eps`.
- IRLS `max_iter=100`, `tol=1e-6` (`splikit/src/cpp_pseudoR2.cpp:48-49`).
- `Eigen::ColPivHouseholderQR` to replace Armadillo's `solve_opts::no_approx`. On singular matrix → return `NaN` (matches R `NA_REAL`).
- Sparse CSC must be sorted within columns: call `M.makeCompressed()` before passing to kernels; assert sorted indices in debug builds.
- `float64` enforced at every API boundary. `gedi2py`'s silent `float32` path was a bug — don't repeat it.
- For null-distribution tests: load Z from R via the H5 fixture (or via `rpy2` in CI), never generate it Python-side.

## Repo layout

```
splikit-py/
├── pyproject.toml                  # scikit-build-core + cibuildwheel
├── CMakeLists.txt                  # Python+pybind11+Eigen3+OpenMP; delegates to src/_splikit_cpp/
├── src/
│   ├── splikit/                    # pure-Python package
│   │   ├── __init__.py
│   │   ├── io/_starsolo.py         # read_starsolo (lands later)
│   │   ├── tl/{_make_m2.py, _pseudo_correlation.py}
│   │   ├── pp/_hve.py              # highly_variable_events
│   │   ├── pl/                     # placeholder for v1.1
│   │   └── _core/                  # _validators.py, _split.py
│   └── _splikit_cpp/
│       ├── CMakeLists.txt
│       ├── bindings.cpp            # pybind11 module entry
│       ├── splikit_kernel.{hpp,cpp}  # SplikitKernel class
│       ├── make_m2.cpp             # ports splikit/src/make_m2_cpp.cpp
│       ├── deviance.cpp            # ports splikit/src/calcDeviances.cpp
│       └── pseudo_r2.cpp           # ports splikit/src/cpp_pseudoR2.cpp (templated)
├── tests/
│   ├── conftest.py
│   ├── data/r_reference.h5         # gitignored if >5 MB; CI-regenerated
│   ├── r_export/export_reference.R
│   ├── load_r_ref.py
│   └── test_*.py
└── .github/workflows/{test.yml, wheels.yml}
```

## Reference repos to mirror

When making non-trivial decisions about the build, kernel boundary, or test layout, look at these first — they solve the same dual-matrix problem in the same ecosystem:

- `/project/6007998/arsham79/multigedipy_pkg/` — most complete reference. Has full pytest suite with `tests/test_cpu_vs_r.py` (golden-file regression with Pearson-cor tolerance bands) and `tests/load_r_ref.py`.
- `/project/6007998/arsham79/gedi2py/` — earlier, lighter take. Same build stack but **no `tests/` directory** — that's the explicit anti-pattern to avoid here.
- `/project/6007998/arsham79/splikit/` — the R source. CLAUDE.md at its root documents the LJV grouping logic and the four-variant pseudo-R² dispatch.

## Specialist agents available

Three project-scoped agents in `.claude/agents/` — invoke when the work matches:

- `pybind11-cmake-engineer` — touches `src/_splikit_cpp/`, `CMakeLists.txt`, `bindings.cpp`, cibuildwheel matrix, OpenMP detection.
- `cross-language-numerical-equivalence-engineer` — touches `tests/test_*_vs_r.py`, `tests/load_r_ref.py`, the R export script, tolerance-band design, OpenMP-determinism debugging.
- `single-cell-ingestion-engineer` — touches `splikit/io/`, STARsolo / 10x format quirks, var-name uniqueness, multi-sample concat conventions.

The user may also invoke the user-scope `scverse-python-architect` for AnnData schema decisions and `R-rcpp-compuational-biologiest` for source-side R/Rcpp questions.

## Common commands

Once the C++ extension lands:

```bash
# Editable install + build the extension
pip install -e ".[test]"
# Or (for development with CMake reconfigure on every C++ edit)
pip install --no-build-isolation -e ".[test]" -Cbuild-dir=build/dev

# Run tests
pytest -v
pytest -v -m "not r_required"     # skip R-equivalence regen
pytest -v tests/test_make_m2.py    # one file
pytest -v -k "pseudo_correlation"  # by name

# Build wheel (matches CI)
pip install cibuildwheel
cibuildwheel --platform linux

# Regenerate R reference fixture
Rscript tests/r_export/export_reference.R tests/data/r_reference.h5
```

## What NOT to do

- Do not call `var_names_make_unique()` anywhere — it silently breaks the `_S`/`_E` suffix scheme.
- Do not put M1 in `X`. Both matrices are `layers`; `X` stays `None`.
- Do not RNG-generate Z for cross-language equivalence tests. Always load Z from the H5 fixture.
- Do not use `#pragma omp reduction(+:...)` or `#pragma omp critical` in hot kernels — they make output non-deterministic across thread counts. Use thread-local workspaces with disjoint writes.
- Do not port `find_variable_genes` / VST. Users compose with `sc.pp.highly_variable_genes` on the gene-expression modality.
- Do not port byte-equality with R as the testing strategy. Use tolerance bands tied to operations.
- Do not mix dtypes silently. Cast to `float64` at every API boundary; assert it.
