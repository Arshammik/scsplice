---
hide:
  - toc
---

# splikit-py

Single-cell alternative-splicing analysis for the [scverse](https://scverse.org) ecosystem.

`splikit-py` is the Python port of the R package [splikit](https://github.com/csglab/splikit). It analyses splice-junction count data in single-cell RNA-seq, treating each event as a pair of inclusion (`M1`) and exclusion (`M2`) counts derived from **local junction variants** (LJVs). The package is AnnData-native — junctions live on the `var` axis, M1 and M2 sit in `layers`, and downstream analysis composes naturally with `scanpy`.

---

## Public API

| Function | Module | Purpose |
|---|---|---|
| `read_starsolo` | `splikit.io` | Ingest STARsolo `Solo.out/SJ/` for one or more samples into a splicing AnnData |
| `make_m2` | `splikit.tl` | Build the exclusion matrix M2 from M1 and LJV `group_id` (C++ kernel, OpenMP-parallel) |
| `highly_variable_events` | `splikit.pp` | Select highly variable splicing events via per-library binomial deviance |
| `pseudo_correlation` | `splikit.tl` | Per-event signed pseudo-R² (Cox-Snell / Nagelkerke) against an external predictor matrix |

---

## Quick start

```python
import splikit as splk

adata = splk.io.read_starsolo(
    sj_dirs=["sample1/Solo.out/SJ", "sample2/Solo.out/SJ"],
    sample_ids=["s1", "s2"],
)
splk.tl.make_m2(adata, n_threads=8)
splk.pp.highly_variable_events(adata, min_row_sum=50, n_threads=8)
```

See [Getting started](getting-started.md) for the full install and walkthrough.

---

## Design principles

- **AnnData-native.** Junctions are `var`, cells are `obs`, M1 and M2 are `layers`. No custom objects — `scanpy`, `scvi-tools`, and the rest of the ecosystem work on the output without adapters.
- **Bit-exact parity with R splikit.** M2 is bit-identical; HVE deviance agrees to `rtol=1e-10`; pseudo-correlation agrees to `rtol=1e-7`. See [Numerical equivalence](explanation/numerical-equivalence.md).
- **Two C++ kernels, one thin Python layer.** `make_m2` and `highly_variable_events` delegate to pybind11-wrapped Eigen kernels; the Python layer handles validation, AnnData conventions, and dispatch only.
- **Intentionally narrow scope.** HVG, plotting, and silhouette utilities are not included — `scanpy`, `pyranges`, and `sklearn` already cover those.

---

## Status

Pre-alpha (v0.1.0.dev0). v1.0 scope is the four functions in the table above. API may change before 1.0.

**R package:** [splikit (CRAN)](https://github.com/csglab/splikit) — same algorithm, R / AnnData-agnostic design.
