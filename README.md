<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/scsplice-logotype-lightmood-1.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/scsplice-logotype-lightmood.svg">
    <img alt="MultiGEDI" src="docs/_static/multigedi-logo-light.svg" width="350">
  </picture>
</div>

<div align="center">



[![PyPI](https://img.shields.io/pypi/v/scsplice.svg?logo=pypi&logoColor=white)](https://pypi.org/project/scsplice/)
[![Python versions](https://img.shields.io/pypi/pyversions/scsplice.svg?logo=python&logoColor=white)](https://pypi.org/project/scsplice/)
[![License: MIT](https://img.shields.io/pypi/l/scsplice.svg)](https://github.com/Arshammik/scsplice/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/Arshammik/scsplice/test.yml?branch=main&label=tests&logo=github)](https://github.com/Arshammik/scsplice/actions/workflows/test.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/Arshammik/scsplice/docs.yml?branch=main&label=docs&logo=materialformkdocs&logoColor=white)](https://arshammik.github.io/scsplice/)


Single-cell alternative-splicing analysis for the [scverse](https://scverse.org) ecosystem.

`scsplice` is the Python port of the R package [splikit](https://github.com/csglab/splikit). It analyses splice-junction count data in single-cell RNA-seq, treating each event as a pair of inclusion (`M1`) and exclusion (`M2`) counts derived from local junction variants (LJVs). The package is AnnData-native — junctions live on the `var` axis, M1 and M2 sit in `layers`, and downstream analysis composes naturally with `scanpy`.

## Status

v2.0.1 keeps a deliberately focused API:

- `scs.io.read_starsolo` — ingest STARsolo `Solo.out/SJ/` for one or more samples.
- `scs.tl.make_m2` — build the exclusion matrix from M1 + LJV grouping.
- `scs.pp.highly_variable_events` — per-library binomial-deviance HVE selection.
- `scs.io.read_starsolo_gene` — read raw or filtered gene counts, including STARsolo EM matrices.
- `scs.tl.pseudo_correlation` — beta-binomial Cox-Snell / Nagelkerke pseudo-R² with event-wise permutation inference.
- `scs.tl.get_pseudo_correlation_result` — materialize export-ready statistics and long null tables.

HVG, plotting, and silhouette utilities from the R package are intentionally omitted — `scanpy`, `pyranges`, and `sklearn` already cover those.

## Installation

`scsplice` ships a C++ extension built via `scikit-build-core` + `pybind11`.
Eigen3 (header-only) is required at install time; OpenMP is optional but
strongly recommended for multi-threaded kernels.

### From PyPI

```bash
pip install scsplice
```

### From source

```bash
git clone https://github.com/Arshammik/scsplice
cd scsplice
pip install .
```

System dependencies before running `pip install`:

**Ubuntu / Debian**

```bash
sudo apt install libeigen3-dev libomp-dev
```

**macOS** (Homebrew)

```bash
brew install eigen libomp
# tell CMake where Apple Clang's OpenMP lives
export OpenMP_ROOT="$(brew --prefix libomp)"
export LDFLAGS="-L${OpenMP_ROOT}/lib"
export CPPFLAGS="-I${OpenMP_ROOT}/include"
```

**HPC cluster (Compute Canada / Sharcnet pattern)**

```bash
module load eigen/3.4.0
# any modern GCC with OpenMP (gcc/12+) on the system module path
```

### Editable install (development)

```bash
pip install -e ".[dev]"
```

This installs the package, all test dependencies, the docs toolchain
(`mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`), and `ruff` /
`pre-commit`. C++ edits require re-running `pip install -e .`; pure-Python
edits take effect immediately.

## Quick start

```python
import numpy as np
import scsplice as scs
import scanpy as sc

# (1) Ingest STARsolo splice-junction counts (M1) and LJV grouping
adata = scs.io.read_starsolo(
    sj_dirs=["sample1/Solo.out/SJ", "sample2/Solo.out/SJ"],
    sample_ids=["s1", "s2"],
)

# (2) Build exclusion matrix (M2) from inclusion counts + junction grouping
scs.tl.make_m2(adata, n_threads=8)

# (3) Identify highly variable events per library using binomial deviance
scs.pp.highly_variable_events(adata, min_row_sum=50, n_threads=8)

# (4) Compute pseudo-correlation with the 100-permutation default
# zdb must be events × cells and aligned to adata.var_names / adata.obs_names
zdb = np.random.default_rng(42).normal(size=(adata.n_vars, adata.n_obs))
scs.tl.pseudo_correlation(adata, zdb, seed=42, n_threads=8)
result = scs.tl.get_pseudo_correlation_result(adata)
result.statistics.to_csv("pseudo_correlation_statistics.csv", index=False)
result.null_distribution.to_csv("pseudo_correlation_null.csv", index=False)

# Optional: compose with scanpy on the splicing embedding
# (PCA / neighbors / leiden over logit(M1 / (M1 + M2)))
```

Gene-expression ingestion defaults to raw STARsolo counts, preferring
`UniqueAndMult-EM.mtx` and falling back to `matrix.mtx`, then independently
applies the internal filtered-barcode whitelist. Use
`matrix_source="filtered"` to read the filtered matrix directly or
`matrix_source="auto"` for the scsplice 2.0.0 source-selection behavior.

## Numerical equivalence

`scsplice` reproduces R `splikit` results to a documented tolerance on a
fixed reference dataset (M2 bit-exact; HVE deviance `rtol=1e-10`;
pseudo-correlation `rtol=1e-7`). The cross-language regression suite,
R reference fixtures, and end-to-end M1/M2 validation pipeline live on the
[`validation` branch](https://github.com/Arshammik/scsplice/tree/validation).

## Documentation

Full documentation is available at https://arshammik.github.io/scsplice/.

Topics include:
- [Getting Started](https://arshammik.github.io/scsplice/getting-started/) — installation and first workflow
- [Tutorials](https://arshammik.github.io/scsplice/tutorials/) — step-by-step notebooks
- [How-to Guides](https://arshammik.github.io/scsplice/how-to-guides/) — recipes for common tasks
- [Reference](https://arshammik.github.io/scsplice/reference/) — complete API documentation
- [Explanation](https://arshammik.github.io/scsplice/explanation/) — conceptual background and design

## License

MIT.
