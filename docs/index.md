---
hide:
  - toc
---

# scsplice

Single-cell alternative-splicing analysis for the [scverse](https://scverse.org) ecosystem.

`scsplice` is the Python port of the R package [splikit](https://github.com/csglab/splikit). It analyses splice-junction count data in single-cell RNA-seq, treating each event as a pair of inclusion (`M1`) and exclusion (`M2`) counts derived from **local junction variants** (LJVs). The package is AnnData-native — junctions live on the `var` axis, M1 and M2 sit in `layers`, and downstream analysis composes naturally with `scanpy`.

---

## Public API

| Function | Module | Purpose |
|---|---|---|
| `read_starsolo` | `scsplice.io` | Ingest STARsolo `Solo.out/SJ/` for one or more samples into a splicing AnnData. Supports `tissue_positions=` for Visium / spatial samples. |
| `read_starsolo_gene` | `scsplice.io` | Ingest `Solo.out/Gene/` into a cell × gene AnnData with raw counts in `X`. Drop-in for `scanpy.pp.normalize_total` and `scvi-tools`. Supports `tissue_positions=` and squidpy `obsm["spatial"]`. |
| `read_starsolo_velocyto` | `scsplice.io` | Ingest `Solo.out/Velocyto/` into an AnnData with `layers["spliced"]`, `layers["unspliced"]`, `layers["ambiguous"]`. Handles both modern (split-file) and legacy (stacked `matrix.mtx`) STARsolo wire formats. Drop-in for `scvelo`. |
| `make_m2` | `scsplice.tl` | Build the exclusion matrix M2 from M1 and LJV `group_id` (C++ kernel, OpenMP-parallel) |
| `highly_variable_events` | `scsplice.pp` | Select highly variable splicing events via per-library binomial deviance |
| `pseudo_correlation` | `scsplice.tl` | Per-event signed pseudo-R² (Cox-Snell / Nagelkerke) against an external predictor matrix |

All three readers share the same `(sample_dirs, sample_ids)` call shape and produce AnnDatas with identical `obs_names`, making multi-modal pipelines a clean set-intersection.

---

## Quick start

```python
import scsplice as scs

adata = scs.io.read_starsolo(
    sj_dirs=["sample1/Solo.out/SJ", "sample2/Solo.out/SJ"],
    sample_ids=["s1", "s2"],
)
scs.tl.make_m2(adata, n_threads=8)
scs.pp.highly_variable_events(adata, min_row_sum=50, n_threads=8)
```

See [Getting started](getting-started.md) for the full install and walkthrough.

---

## Acknowledgements

`scsplice` is a Python port of the R package [splikit](https://github.com/csglab/splikit), developed by the Computational and Statistical Genomics (CSG) Laboratory at McGill University. We gratefully acknowledge the original R splikit authors and the CSG Lab for the foundational algorithm and reference implementation.

---

## Design principles

- **AnnData-native.** Junctions are `var`, cells are `obs`, M1 and M2 are `layers`. No custom objects — `scanpy`, `scvi-tools`, and the rest of the ecosystem work on the output without adapters.
- **Bit-exact parity with R splikit.** M2 is bit-identical; HVE deviance agrees to `rtol=1e-10`; pseudo-correlation agrees to `rtol=1e-7`. The cross-language regression suite lives on the [`validation` branch](https://github.com/Arshammik/scsplice/tree/validation).
- **Two C++ kernels, one thin Python layer.** `make_m2` and `highly_variable_events` delegate to pybind11-wrapped Eigen kernels; the Python layer handles validation, AnnData conventions, and dispatch only.
- **Intentionally narrow scope.** HVG, plotting, and silhouette utilities are not included — `scanpy`, `pyranges`, and `sklearn` already cover those.

---

## Status

v1.0. The six functions in the table above are the complete v1.0 API.

**R package:** [splikit (CRAN)](https://github.com/csglab/splikit) — same algorithm, R / AnnData-agnostic design.

---

*Brand note: the hexagonal seagull logo is the original R splikit artwork, shared between both packages to signal that the Python and R implementations are the same algorithm with a unified brand. In dark mode the logo is rendered on a white pill background so the black-on-white illustration remains legible against the dark site background.*
