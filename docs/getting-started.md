# Getting started

## Requirements

- Python 3.10 or later
- [Eigen3](https://eigen.tuxfamily.org/) header library (required to build the C++ extension)
- A C++17-capable compiler (GCC 10+, Clang 13+, MSVC 2019+)

Install Eigen3 with your system package manager before running `pip install`:

```bash
# Debian / Ubuntu
sudo apt-get install libeigen3-dev

# macOS (Homebrew)
brew install eigen

# Conda
conda install -c conda-forge eigen
```

## Install

Once published to PyPI:

```bash
pip install splikit-py
```

For development (editable install from source):

```bash
git clone https://github.com/Arshammik/splikitpy
cd splikitpy
pip install -e ".[test,docs]"
```

The `scikit-build-core` backend calls CMake to compile the Eigen-based C++ extension at install time. The build takes 15–60 seconds depending on hardware.

## Verify the install

```python
import splikit as splk

print(splk.__version__)
print(splk._splikit_cpp.__openmp__)  # True if built with OpenMP
```

---

## Hello-world walkthrough

This walkthrough uses a tiny synthetic STARsolo directory that ships with the test suite. It exercises the full pipeline in under a second.

### 1. Generate synthetic input

```python
import tempfile, pathlib, numpy as np, scipy.sparse as sp, scipy.io as sio, pandas as pd

# Build a minimal Solo.out/SJ/raw/ directory with 4 junctions × 10 cells.
# Junction 1 and 2 share start coord (chr1:100) → form an _S LJV group.
# Junction 1 and 3 share end coord (chr1:200) → form an _E LJV group.

tmp = pathlib.Path(tempfile.mkdtemp())
raw_dir = tmp / "Solo.out" / "SJ" / "raw"
raw_dir.mkdir(parents=True)

# barcodes.tsv
barcodes = [f"ACGT{'A'*12}-{i}" for i in range(1, 11)]  # 10 cells
(raw_dir / "barcodes.tsv").write_text("\n".join(barcodes) + "\n")

# matrix.mtx: 4 junctions × 10 cells, sparse
rng = np.random.default_rng(0)
data = rng.integers(0, 10, size=(4, 10)).astype(np.float64)
mat = sp.csr_matrix(data)
sio.mmwrite(str(raw_dir / "matrix.mtx"), mat)

# SJ.out.tab: 4 junctions
sj_rows = [
    # chr    start  end    strand  motif  annot  uniq_mapped
    ["chr1", 100,   200,   1,      2,     1,     5],
    ["chr1", 100,   300,   1,      2,     1,     3],
    ["chr1", 50,    200,   1,      2,     1,     4],
    ["chr1", 400,   500,   1,      2,     1,     2],
]
sj_df = pd.DataFrame(sj_rows, columns=[
    "chr", "start", "end", "strand", "intron_motif", "is_annot", "unique_mapped"
])
sj_df.to_csv(tmp / "Solo.out" / "SJ.out.tab", sep="\t", header=False, index=False)
```

### 2. Read with `read_starsolo`

```python
import splikit as splk

adata = splk.io.read_starsolo(
    sj_dirs=[tmp / "Solo.out" / "SJ"],
    sample_ids=["demo"],
    verbose=True,
)
print(adata)
# AnnData object with n_obs × n_vars = 10 × N
# layers: 'M1'
# obs: 'barcode', 'sample_id'
# var: 'chr', 'start', 'end', 'strand', 'intron_motif', 'is_annot',
#      'unique_mapped', 'row_names_mtx', 'group_id', 'group_kind', 'group_count'
# uns: 'splikit'
```

`adata.var_names` look like `chr1:100-200_S`, `chr1:100-300_S`, etc. — globally unique by construction. Never call `var_names_make_unique` on this object.

### 3. Build M2 with `make_m2`

```python
splk.tl.make_m2(adata, n_threads=1)

print("M2 stored:", "M2" in adata.layers)  # True
print("m2_valid:", adata.uns["splikit"]["m2_valid"])  # True
```

M2 has the same shape, sparsity format (CSC), and dtype (float64) as M1. For every event `i` and cell `j`:

```
M2[j, i] = sum(M1[j, k] for k in same LJV group as i) - M1[j, i]
```

### 4. Select highly variable events

```python
splk.pp.highly_variable_events(adata, min_row_sum=1, n_threads=1)

hve = adata.var["highly_variable"]
print(f"{hve.sum()} events selected out of {len(hve)}")
print(adata.var[["sum_deviance", "highly_variable"]].head())
```

### 5. Compose with scanpy

Once you have M1 and M2, you can compute the logit-PSI representation and pass it to `scanpy` for dimensionality reduction and clustering:

```python
import scanpy as sc
import scipy.sparse as sp
import numpy as np

# Logit-PSI: logit(M1 / (M1 + M2)) for cells with coverage
M1 = adata.layers["M1"].toarray()
M2 = adata.layers["M2"].toarray()
total = M1 + M2
psi = np.where(total > 0, M1 / total, np.nan)

# Subset to highly variable events for downstream analysis
adata_hve = adata[:, adata.var["highly_variable"]].copy()
# ... sc.pp.pca, sc.pp.neighbors, sc.tl.leiden, etc.
```

---

## Next steps

- [Tutorials](tutorials/index.md) — end-to-end walkthrough on real STARsolo output
- [Multi-sample ingestion](how-to/multi-sample-ingestion.md) — passing multiple directories
- [Recompute M2 after subsetting](how-to/recompute-m2-after-subsetting.md) — why you must call `make_m2` again after `adata = adata[:, mask]`
- [Data model](explanation/data-model.md) — the LJV concept, `_S` / `_E` suffixes, why `group_id` is dense
