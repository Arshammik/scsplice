# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-09-04

### Added

- `scs.pp.highly_variable_genes` — Seurat/R-splikit VST (variance-stabilizing
  transformation) highly-variable-gene selection, porting R splikit's
  `find_variable_genes(method = "vst")`. Requires the optional `scikit-misc`
  dependency (`pip install "scsplice[hvg]"`); its `loess` binding wraps the
  same netlib/Cleveland-Grosse Fortran/C code R's `stats::loess()` calls, so
  results are numerically comparable to R, not just approximately similar —
  verified against real R splikit 2.3.3 output to a max absolute difference
  of 4.4e-15 across 500 genes (`tests/test_highly_variable_genes_vs_r.py`).
  R splikit's other method, `sum_deviance`, is not ported (see the function
  docstring for rationale). New C++ kernels `hvg_row_mean_var` and
  `hvg_standardize_variance` in `_scsplice_cpp` back the non-loess passes.

[2.1.0]: https://github.com/Arshammik/scsplice/compare/v2.0.1...v2.1.0

## [2.0.1] - 2026-07-30

### Added

- `scs.io.read_starsolo_gene` now accepts independent `matrix_source` and
  `matrix_file` controls, including STARsolo `UniqueAndMult-EM.mtx` and gzip
  variants. Resolved per-sample inputs are recorded in AnnData metadata.
- `scs.tl.pseudo_correlation` now computes per-event null means, null standard
  deviations, valid-draw counts, two-sided empirical p-values, and
  Benjamini-Hochberg adjusted p-values.
- `scs.tl.get_pseudo_correlation_result` materializes export-ready per-event
  statistics and long event/permutation null tables from AnnData.

### Changed

- The gene reader now defaults to `matrix_source="raw"` and prefers
  `UniqueAndMult-EM.mtx` before falling back to `matrix.mtx`. Internal filtered
  barcodes are applied independently to the selected count matrix.
- Pseudo-correlation now runs 100 permutations by default, matching R splikit
  2.3.2. Pass `n_permutations=0` for observed-only computation.

[2.0.1]: https://github.com/Arshammik/scsplice/compare/v2.0.0...v2.0.1

## [2.0.0] - 2026-05-11

### Breaking Changes

- PyPI distribution renamed `splikit-py` → `scsplice`. Install with `pip install scsplice`.
- Python import alias changes: `import scsplice as scs` (was `import splikit as splk`).
- `uns["splikit"]` schema key renamed to `uns["scsplice"]`. A one-release compat shim reads the legacy key with a `FutureWarning` and migrates it automatically; will be removed in 3.0.
- `SPLIKIT_REAL_DATA_DIR` env var renamed to `SCSPLICE_REAL_DATA_DIR`. Old name accepted with `FutureWarning` for one release.
- C++ extension module renamed `_splikit_cpp` → `_scsplice_cpp`.

### Added

- Compat shim in `scsplice._core._validators` (`get_scsplice_ns` / `setdefault_scsplice_ns`) for graceful migration of v1.0 AnnData objects carrying `uns["splikit"]`.
- `SCSPLICE_REAL_DATA_DIR` env var support in `tests/conftest.py` with fallback to `SPLIKIT_REAL_DATA_DIR` + `FutureWarning`.

[2.0.0]: https://github.com/Arshammik/scsplice/compare/v1.0.0...v2.0.0

## [1.0.0] - 2026-05-11

### Added

- `splk.io.read_starsolo` — Ingest STARsolo `Solo.out/SJ/` for one or more samples into a single AnnData with M1 (inclusion counts) in `layers["M1"]` and LJV grouping in `var["group_id"]`. Supports spatial data via optional `tissue_positions=` parameter.
- `splk.io.read_starsolo_gene` — Ingest `Solo.out/Gene/` gene-expression counts into a standard cell × gene AnnData with counts in `X`. Drop-in for `scanpy.pp.normalize_total` and `scvi-tools`. Supports `tissue_positions=` and populates `obsm["spatial"]` for Visium samples.
- `splk.io.read_starsolo_velocyto` — Ingest `Solo.out/Velocyto/` spliced/unspliced/ambiguous layers into an AnnData compatible with `scvelo`. Handles both modern (split-file) and legacy (stacked `matrix.mtx`) STARsolo wire formats.
- `scs.tl.make_m2` — Build the exclusion matrix M2 from M1 and LJV grouping via C++ kernel with optional OpenMP parallelism. Output is bit-exact with R splikit.
- `splk.tl.pseudo_correlation` — Per-event signed pseudo-R² (Cox-Snell or Nagelkerke) against an external predictor matrix via iteratively reweighted least squares (IRLS).
- `splk.pp.highly_variable_events` — Select highly variable splicing events per library using binomial-deviance scoring.
- `scsplice.settings` — Global settings object for configurable behavior (verbosity, I/O defaults).
- MkDocs Material documentation site with tutorials, how-to guides, API reference, and conceptual explanations.

### Known Limitations

- `splk.pl` (plotting) is a v1.1 placeholder. Use `scanpy.pl`, `scvelo.pl`, and `squidpy` for downstream visualization.
- HVG (highly variable genes) and silhouette metrics are intentionally not ported — compose with `scanpy.pp.highly_variable_genes` and `sklearn.metrics.silhouette_score` instead.
- GTF annotation and gene-level plotting are out of scope; use `pyranges` for GTF operations.
- R-equivalence validation suite (cross-language regression tests, R fixtures, tolerance bands) lives on the [`validation` branch](https://github.com/Arshammik/scsplice/tree/validation), not main.

### Internal Notes

This is the inaugural v1.0 PyPI release. The package has been tested for numerical equivalence with R splikit on a fixed reference dataset (M2 bit-exact; HVE deviance rtol=1e-10; pseudo-correlation rtol=1e-7). Production usage at scale is encouraged; please report bugs and feature requests on GitHub.

[1.0.0]: https://github.com/Arshammik/scsplice/releases/tag/v1.0.0
