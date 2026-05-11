# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-11

### Added

- `splk.io.read_starsolo` — Ingest STARsolo `Solo.out/SJ/` for one or more samples into a single AnnData with M1 (inclusion counts) in `layers["M1"]` and LJV grouping in `var["group_id"]`. Supports spatial data via optional `tissue_positions=` parameter.
- `splk.io.read_starsolo_gene` — Ingest `Solo.out/Gene/` gene-expression counts into a standard cell × gene AnnData with counts in `X`. Drop-in for `scanpy.pp.normalize_total` and `scvi-tools`. Supports `tissue_positions=` and populates `obsm["spatial"]` for Visium samples.
- `splk.io.read_starsolo_velocyto` — Ingest `Solo.out/Velocyto/` spliced/unspliced/ambiguous layers into an AnnData compatible with `scvelo`. Handles both modern (split-file) and legacy (stacked `matrix.mtx`) STARsolo wire formats.
- `splk.tl.make_m2` — Build the exclusion matrix M2 from M1 and LJV grouping via C++ kernel with optional OpenMP parallelism. Output is bit-exact with R splikit.
- `splk.tl.pseudo_correlation` — Per-event signed pseudo-R² (Cox-Snell or Nagelkerke) against an external predictor matrix via iteratively reweighted least squares (IRLS).
- `splk.pp.highly_variable_events` — Select highly variable splicing events per library using binomial-deviance scoring.
- `splikit.settings` — Global settings object for configurable behavior (verbosity, I/O defaults).
- MkDocs Material documentation site with tutorials, how-to guides, API reference, and conceptual explanations.

### Known Limitations

- `splk.pl` (plotting) is a v1.1 placeholder. Use `scanpy.pl`, `scvelo.pl`, and `squidpy` for downstream visualization.
- HVG (highly variable genes) and silhouette metrics are intentionally not ported — compose with `scanpy.pp.highly_variable_genes` and `sklearn.metrics.silhouette_score` instead.
- GTF annotation and gene-level plotting are out of scope; use `pyranges` for GTF operations.
- R-equivalence validation suite (cross-language regression tests, R fixtures, tolerance bands) lives on the [`validation` branch](https://github.com/Arshammik/splikitpy/tree/validation), not main.

### Internal Notes

This is the inaugural v1.0 PyPI release. The package has been tested for numerical equivalence with R splikit on a fixed reference dataset (M2 bit-exact; HVE deviance rtol=1e-10; pseudo-correlation rtol=1e-7). Production usage at scale is encouraged; please report bugs and feature requests on GitHub.

[1.0.0]: https://github.com/Arshammik/splikitpy/releases/tag/v1.0.0
