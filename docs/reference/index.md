# API Reference

Auto-generated from source docstrings. All public functions carry NumPy-style docstrings with parameter types, return types, and at least one runnable example.

The v2.1.0 public API spans three modules:

| Module | Function | Description |
|---|---|---|
| `scsplice.io` | [`read_starsolo`](io.md) | Ingest STARsolo SJ output |
| `scsplice.io` | [`read_starsolo_gene`](io.md) | Ingest selectable STARsolo gene-count matrices |
| `scsplice.io` | [`read_starsolo_velocyto`](io.md) | Ingest STARsolo velocity matrices |
| `scsplice.tl` | [`make_m2`](tl.md) | Build the M2 exclusion layer |
| `scsplice.tl` | [`pseudo_correlation`](tl.md) | Per-event pseudo-R² and permutation inference |
| `scsplice.tl` | [`get_pseudo_correlation_result`](tl.md) | Materialize export-ready result tables |
| `scsplice.pp` | [`highly_variable_events`](pp.md) | HVE selection via binomial deviance |
| `scsplice.pp` | [`highly_variable_genes`](pp.md) | HVG selection via Seurat/R-splikit VST (requires the `[hvg]` extra) |

`scsplice.pl` remains reserved for a future plotting module.
