# Tutorials

Tutorials walk through complete analyses from raw STARsolo output to biological results. They are learning-oriented: every step is shown, every intermediate object is described.

---

## Planned for v1.1

**Quickstart: end-to-end splicing analysis (Jupyter notebook)**

A self-contained notebook that:

1. Downloads a small public STARsolo SJ output (PBMC 3k, ~200 cells after whitelist filtering).
2. Runs `read_starsolo` → `make_m2` → `highly_variable_events`.
3. Computes logit-PSI, runs PCA via `scanpy`, and clusters cells.
4. Demonstrates `pseudo_correlation` against a gene-expression PC1 derived from the same cells.

The notebook will be rendered and embedded here once the v1.1 release ships. Until then, see [Getting started](../getting-started.md) for a condensed walkthrough using synthetic data.

---

!!! note "Notebook execution"
    All tutorial notebooks are executed at documentation build time using `mkdocs-jupyter`. Committed notebooks with stale outputs are not accepted; the CI workflow re-runs them from scratch on each merge to `main`.
