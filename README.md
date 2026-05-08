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

## Installation

`splikit-py` ships a C++ extension built via `scikit-build-core` + `pybind11`.
Eigen3 (header-only) is required at install time; OpenMP is optional but
strongly recommended for multi-threaded kernels.

### From PyPI (once v1.0 is published)

```bash
pip install splikit-py
```

### From source

```bash
git clone https://github.com/Arshammik/splikitpy
cd splikitpy
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

`splikit-py` reproduces R `splikit` results to a documented tolerance on a
fixed reference dataset (M2 bit-exact; HVE deviance `rtol=1e-10`;
pseudo-correlation `rtol=1e-7`). The cross-language regression suite,
R reference fixtures, and end-to-end M1/M2 validation pipeline live on the
[`validation` branch](https://github.com/Arshammik/splikitpy/tree/validation).

## License

MIT.
