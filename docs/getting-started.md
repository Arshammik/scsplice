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
pip install scsplice
```

For development (editable install from source):

```bash
git clone https://github.com/Arshammik/scsplice
cd scsplice
pip install -e ".[test,docs]"
```

The `scikit-build-core` backend calls CMake to compile the Eigen-based C++ extension at install time. The build takes 15–60 seconds depending on hardware.

## Verify the install

```python
import scsplice as scs

print(scs.__version__)
print(scs._scsplice_cpp.__openmp__)  # True if built with OpenMP
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
import scsplice as scs

adata = scs.io.read_starsolo(
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
# uns: 'scsplice'
```

`adata.var_names` look like `chr1:100-200_S`, `chr1:100-300_S`, etc. — globally unique by construction. Never call `var_names_make_unique` on this object.

### 3. Build M2 with `make_m2`

```python
scs.tl.make_m2(adata, n_threads=1)

print("M2 stored:", "M2" in adata.layers)  # True
print("m2_valid:", adata.uns["scsplice"]["m2_valid"])  # True
```

M2 has the same shape, sparsity format (CSC), and dtype (float64) as M1. For every event `i` and cell `j`:

```
M2[j, i] = sum(M1[j, k] for k in same LJV group as i) - M1[j, i]
```

### 4. Select highly variable events

```python
scs.pp.highly_variable_events(adata, min_row_sum=1, n_threads=1)

hve = adata.var["highly_variable"]
print(f"{hve.sum()} events selected out of {len(hve)}")
print(adata.var[["sum_deviance", "highly_variable"]].head())
```

### 5. Compute pseudo-correlation and export the null result

`pseudo_correlation` expects one predictor per event and cell. It stores the
observed values and event-wise inference directly in AnnData and runs 100
permutations by default:

```python
zdb = np.random.default_rng(42).normal(size=(adata.n_vars, adata.n_obs))
scs.tl.pseudo_correlation(adata, zdb, seed=42, n_threads=1)

print(adata.var[[
    "pseudo_correlation",
    "pseudo_correlation_null_mean",
    "pseudo_correlation_emp_pvalue",
    "pseudo_correlation_emp_padj",
]].head())

result = scs.tl.get_pseudo_correlation_result(adata)
result.statistics.to_csv("pseudo_correlation_statistics.csv", index=False)
result.null_distribution.to_csv("pseudo_correlation_null.csv", index=False)
```

The long null table is generated on demand instead of duplicated in the
AnnData file. Its pooled values are descriptive; each empirical p-value is
calculated from that event's own permutation draws. Use `n_permutations=0`
when only the observed pseudo-correlation is required.

### 6. Compose with scanpy

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

## Gene expression and velocity readers

If you need gene counts or RNA velocity in the same pipeline, two additional readers produce AnnData objects with aligned `obs_names` so you can intersect cells across modalities:

```python
# Gene-expression AnnData (counts in X, drop-in for scanpy)
gex = scs.io.read_starsolo_gene(
    sample_dirs=["sample1", "sample2"],
    sample_ids=["s1", "s2"],
    var_names="gene_ids",       # default; Ensembl IDs as var_names
    matrix_source="raw",        # default; independent from barcode filtering
    matrix_file="auto",         # EM counts first, then matrix.mtx
    verbose=True,
)
# Ready for: sc.pp.normalize_total(gex), sc.pp.highly_variable_genes(gex)

# Velocyto AnnData (spliced / unspliced / ambiguous in layers, drop-in for scvelo)
vel = scs.io.read_starsolo_velocyto(
    sample_dirs=["sample1", "sample2"],
    sample_ids=["s1", "s2"],
)
# Ready for: scv.pp.filter_and_normalize(vel), scv.tl.velocity(vel)
```

By default the gene reader selects `raw/UniqueAndMult-EM.mtx` when available,
falls back to `raw/matrix.mtx`, and then retains the barcodes listed in
`filtered/barcodes.tsv`. Set `matrix_source="filtered"` to read the filtered
matrix directly, or set `use_internal_whitelist=False` to retain every barcode
from the selected matrix. Raw matrices can be much larger in memory; choose
`matrix_source="filtered"` when EM counts are unnecessary.

Because all three readers use the same `(sample_dirs, sample_ids)` inputs and produce `<barcode>-<sample_id>` `obs_names`, aligning modalities is a simple intersection:

```python
common = sorted(set(adata.obs_names) & set(gex.obs_names) & set(vel.obs_names))
adata, gex, vel = adata[common].copy(), gex[common].copy(), vel[common].copy()
```

### Spatial / Visium samples

Pass a `tissue_positions.csv` from Space Ranger to any reader to populate squidpy-compatible spatial metadata:

```python
vis = scs.io.read_starsolo_gene(
    sample_dirs=["visium_sample"],
    sample_ids=["vis1"],
    tissue_positions=["visium_sample/outs/tissue_positions.csv"],
    spatial_library_ids=["vis1"],
)
# Populated: obs["in_tissue"], obs["array_row"], obs["array_col"]
#            obsm["spatial"]  — (n_obs, 2) float64 pixel coordinates
#            uns["spatial"]["vis1"]  — squidpy-shaped scaffold

import squidpy as sq
sq.pl.spatial_scatter(vis, color="in_tissue")
```

See [Read spatial data with tissue_positions](how-to/read-spatial-data-with-tissue-positions.md) for the full how-to, including multi-sample mixed spatial / non-spatial concat and Space Ranger v1 vs v2 CSV detection.

---

## Next steps

- [Tutorials](tutorials/index.md) — end-to-end walkthrough on real STARsolo output
- [Multi-sample ingestion](how-to/multi-sample-ingestion.md) — passing multiple directories
- [Recompute M2 after subsetting](how-to/recompute-m2-after-subsetting.md) — why you must call `make_m2` again after `adata = adata[:, mask]`
- [Read spatial data with tissue_positions](how-to/read-spatial-data-with-tissue-positions.md) — Visium, squidpy, mixed-sample concat
- [Multi-modal pipeline](how-to/multi-modal-pipeline.md) — feeding all three readers into a joint model
- [Data model](explanation/data-model.md) — the LJV concept, `_S` / `_E` suffixes, why `group_id` is dense
