# Release notes

## v2.0.1 (2026-07-30) — R splikit 2.3.2–2.3.3 parity

### Gene-expression matrix selection

`scs.io.read_starsolo_gene` now separates matrix selection from barcode
filtering. The default reads `Gene/raw/UniqueAndMult-EM.mtx` when present,
falls back to `Gene/raw/matrix.mtx`, and then applies the internal
`Gene/filtered/barcodes.tsv` whitelist. Both uncompressed and `.gz` files are
supported.

Use `matrix_source="filtered"` to read an already-filtered matrix,
`matrix_file="matrix.mtx"` to force standard unique-count input, or
`matrix_source="auto"` to retain scsplice 2.0.0 source selection.

### Repeated pseudo-correlation nulls and export

`scs.tl.pseudo_correlation` now runs 100 permutations by default and stores the
raw event-by-permutation matrix in `adata.varm["pseudo_correlation_null"]`.
Per-event null means, sample standard deviations, valid counts, two-sided
empirical p-values, and BH-adjusted p-values are stored as aligned `adata.var`
columns. Pass `n_permutations=0` for observed-only computation.

`scs.tl.get_pseudo_correlation_result(adata)` returns an export-friendly object
with `statistics`, long `null_distribution`, and `metadata` components. The
pooled long table is descriptive; empirical p-values use each event's own null
draws.

---

## v2.0.0 (2026-05-11) — package renamed scsplice

**Breaking change:** The PyPI distribution name changes from `splikit-py` to `scsplice`.
The canonical import alias is now `scs` (parallels `sc` / `scv` / `sq`).

```bash
# Before (splikit-py)
pip install splikit-py
import splikit as splk

# After (scsplice)
pip install scsplice
import scsplice as scs
```

### Migration guide

| v1.0 (splikit-py) | v2.0 (scsplice) |
|---|---|
| `import splikit as splk` | `import scsplice as scs` |
| `splk.io.read_starsolo(...)` | `scs.io.read_starsolo(...)` |
| `splk.tl.make_m2(adata)` | `scs.tl.make_m2(adata)` |
| `splk.pp.highly_variable_events(adata)` | `scs.pp.highly_variable_events(adata)` |
| `splk.tl.pseudo_correlation(adata, Z)` | `scs.tl.pseudo_correlation(adata, Z)` |
| `adata.uns["splikit"]` | `adata.uns["scsplice"]` |
| `pip install splikit-py` | `pip install scsplice` |
| `SPLIKIT_REAL_DATA_DIR` env var | `SCSPLICE_REAL_DATA_DIR` env var |

### `uns["splikit"]` compatibility shim

AnnData objects created with v1.0 carry `uns["splikit"]`. In v2.0, the canonical
key is `uns["scsplice"]`. The shim in `scsplice._core._validators.get_scsplice_ns()`
and `setdefault_scsplice_ns()` reads from `uns["splikit"]` with a `FutureWarning`
and migrates the value to `uns["scsplice"]` automatically. The legacy key will be
removed in v3.0.

To migrate an existing AnnData object explicitly:

```python
import scsplice as scs
import warnings

# Load an old AnnData (will emit FutureWarning on first scsplice call)
with warnings.catch_warnings(record=True):
    scs.tl.make_m2(adata)  # shim fires, migrates uns key

# adata.uns["scsplice"] is now populated; adata.uns["splikit"] is removed
```

### `SCSPLICE_REAL_DATA_DIR` env var

The `SPLIKIT_REAL_DATA_DIR` environment variable used by the test suite is
renamed to `SCSPLICE_REAL_DATA_DIR`. The old name is still accepted with a
`FutureWarning` for one release cycle; update your shell config.

---

## v1.0.0 (2026-05-11)

Initial PyPI release as `splikit-py`.

**Scope:** six public functions across `io`, `tl`, and `pp`.

**Cross-language parity:** M2 bit-exact vs. R splikit; HVE deviance `rtol=1e-10`; pseudo-correlation `rtol=1e-7`. The full regression suite lives on the [`validation` branch](https://github.com/Arshammik/scsplice/tree/validation).

### New in v1.0.0

**`scs.io.read_starsolo`** — Ingest STARsolo `Solo.out/SJ/` for one or more samples into a single AnnData with M1 (inclusion counts) in `layers["M1"]` and LJV grouping in `var["group_id"]`. Supports spatial data via optional `tissue_positions=` parameter.

**`scs.io.read_starsolo_gene`** — Ingest `Solo.out/Gene/` gene-expression counts into a standard cell × gene AnnData with raw UMI counts in `X`. Drop-in for `scanpy.pp.normalize_total`, `scanpy.pp.highly_variable_genes`, and `scvi-tools`. Supports `tissue_positions=` for spatial samples with full squidpy `obsm["spatial"]` and `uns["spatial"]` population.

**`scs.io.read_starsolo_velocyto`** — Ingest `Solo.out/Velocyto/` spliced/unspliced/ambiguous layers into an AnnData compatible with `scvelo`. Handles both modern (split-file) and legacy (stacked `matrix.mtx`) STARsolo wire formats.

**`scs.tl.make_m2`** — Build the exclusion matrix M2 from M1 and LJV grouping via C++ kernel with optional OpenMP parallelism. Output is bit-exact with R splikit.

**`scs.tl.pseudo_correlation`** — Per-event signed pseudo-R² (Cox-Snell or Nagelkerke) against an external predictor matrix via iteratively reweighted least squares (IRLS).

**`scs.pp.highly_variable_events`** — Select highly variable splicing events per library using binomial-deviance scoring.

**`scsplice.settings`** — Global settings object for configurable behavior (verbosity, I/O defaults).

MkDocs Material documentation site with tutorials, how-to guides, API reference, and conceptual explanations.

### Known limitations

- `scs.pl` (plotting) is a v2.1 placeholder. Use `scanpy.pl`, `scvelo.pl`, and `squidpy` for downstream visualization.
- HVG (highly variable genes) and silhouette metrics are intentionally not ported — compose with `scanpy.pp.highly_variable_genes` and `sklearn.metrics.silhouette_score` instead.
- GTF annotation and gene-level plotting are out of scope; use `pyranges` for GTF operations.
- R-equivalence validation suite (cross-language regression tests, R fixtures, tolerance bands) lives on the [`validation` branch](https://github.com/Arshammik/scsplice/tree/validation), not main.
