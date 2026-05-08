# splikit-py

Single-cell alternative-splicing analysis for the [scverse](https://scverse.org) ecosystem.

`splikit-py` is the Python port of the R package [splikit](https://github.com/csglab/splikit). It analyses splice-junction count data in single-cell RNA-seq, treating each event as a pair of inclusion (`M1`) and exclusion (`M2`) counts derived from local junction variants (LJVs). The package is AnnData-native — junctions live on the `var` axis, M1 and M2 sit in `layers`, and downstream analysis composes naturally with `scanpy`.

## Status

Pre-alpha. v1.0 scope is intentionally narrow:

- `splk.io.read_starsolo` — ingest STARsolo `Solo.out/SJ/` for one or more samples.
- `splk.tl.make_m2` — build the exclusion matrix from M1 + LJV grouping.
- `splk.pp.highly_variable_events` — per-library binomial-deviance HVE selection.
- `splk.tl.pseudo_correlation` — beta-binomial Cox-Snell / Nagelkerke pseudo-R² against an external matrix.

HVG, plotting, and silhouette utilities from the R package are intentionally omitted — `scanpy`, `pyranges`, and `sklearn` already cover those.

## Install (once published)

```bash
pip install splikit-py
```

## Quick start

```python
import splikit as splk
import scanpy as sc

adata = splk.io.read_starsolo(
    sj_dirs=["sample1/Solo.out/SJ", "sample2/Solo.out/SJ"],
    sample_ids=["s1", "s2"],
)
splk.tl.make_m2(adata, n_threads=8)
splk.pp.highly_variable_events(adata, min_row_sum=50, n_threads=8)

# Optional: compose with scanpy on the splicing embedding
# (PCA / neighbors / leiden over logit(M1 / (M1 + M2)))
```

## Numerical equivalence

`splikit-py` reproduces R `splikit` results to a documented tolerance on a fixed reference dataset; see `tests/test_*_vs_r.py` and `tests/r_export/export_reference.R` for the regression protocol.

## License

MIT.
