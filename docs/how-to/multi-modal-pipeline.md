# Multi-modal pipeline

This how-to shows how to feed the same `(sample_dirs, sample_ids)` inputs into
all three STARsolo readers, align cells across modalities by `obs_names`, and
pass the result to a joint model (`multigedipy.MultiGEDIModel`).

For the reader design rationale and AnnData schemas see
[STARsolo readers and AnnData data layouts](../explanation/io-readers-and-data-layouts.md).

---

## 1. Read all three modalities

The three readers share an identical call shape. Running them on the same
`(sample_dirs, sample_ids)` produces AnnDatas with a common `obs_names`
convention (`<barcode>-<sample_id>`):

```python
import splikit as splk

sample_dirs = ["sample1", "sample2", "sample3"]
sample_ids  = ["s1",      "s2",      "s3"]

# Splicing (M1 + M2 in layers; X = None)
spl = splk.io.read_starsolo(
    sj_dirs=[f"{d}/Solo.out/SJ" for d in sample_dirs],
    sample_ids=sample_ids,
    verbose=True,
)

# Gene expression (raw counts in X)
gex = splk.io.read_starsolo_gene(
    sample_dirs=sample_dirs,
    sample_ids=sample_ids,
    var_names="gene_ids",
)

# RNA velocity (spliced / unspliced / ambiguous in layers; X = spliced)
vel = splk.io.read_starsolo_velocyto(
    sample_dirs=sample_dirs,
    sample_ids=sample_ids,
)
```

!!! note "Whitelist alignment"
    Each reader applies its own whitelist by default (`use_internal_whitelist=True`),
    which means the three AnnDatas may have slightly different cell sets (the SJ
    reader keeps cells with at least one junction count; the gene and velocyto
    readers use `Gene/filtered/barcodes.tsv`). The intersection step below
    handles this.

---

## 2. Prepare the splicing modality

Run the splicing pipeline before aligning cells — HVE selection may further
reduce the cell set if you filter on `min_row_sum`:

```python
splk.tl.make_m2(spl, n_threads=8)
splk.pp.highly_variable_events(spl, min_row_sum=50, n_top=10_000,
                                sample_key="sample_id")
spl_hve = spl[:, spl.var["highly_variable"]].copy()
# M2 is invalidated by var-axis subsetting; rebuild.
splk.tl.make_m2(spl_hve, n_threads=8)
```

---

## 3. Prepare the gene-expression modality

```python
import scanpy as sc

sc.pp.filter_cells(gex, min_genes=200)
sc.pp.filter_genes(gex, min_cells=3)
sc.pp.normalize_total(gex, target_sum=1e4)
sc.pp.log1p(gex)
sc.pp.highly_variable_genes(gex, n_top_genes=5_000, batch_key="sample_id")
gex_hvg = gex[:, gex.var["highly_variable"]].copy()
```

---

## 4. Align cells across modalities

Use set-intersection on `obs_names`. All three readers guarantee
`<barcode>-<sample_id>` names, so the intersection is barcode-level exact:

```python
common = sorted(
    set(spl_hve.obs_names) & set(gex_hvg.obs_names) & set(vel.obs_names)
)
spl_hve  = spl_hve[common].copy()
gex_hvg  = gex_hvg[common].copy()
vel_c    = vel[common].copy()

print(f"{len(common)} cells in common across all three modalities")
```

---

## 5. Pass to a joint model

### Option A — `multigedipy.MultiGEDIModel`

This is the pattern used in `validation/e2e/run_multimodal_pipeline.py`.

```python
import numpy as np
import scipy.sparse as sp
from multigedipy import MultiGEDIModel

sample_vec = np.asarray(spl_hve.obs["sample_id"].astype(str))

# Transpose to features × cells (multigedipy convention).
M1 = sp.csc_matrix(spl_hve.layers["M1"].T, dtype=np.float64)
M2 = sp.csc_matrix(spl_hve.layers["M2"].T, dtype=np.float64)
G  = sp.csc_matrix(gex_hvg.X.T, dtype=np.float64)

model = MultiGEDIModel(K=20, mode="Bsphere", seed=42, num_threads=8)
model.add_modality(
    "gene_expression", data=G, sample_vec=sample_vec,
    obs_type="M", orthoZ=True,
)
model.add_modality(
    "splicing", data=(M1, M2), sample_vec=sample_vec,
    obs_type="M_list", orthoZ=False, is_si_fixed=True,
)
model.train(iterations=30)
Z = model.get_Z()   # (n_cells, K) latent factors
```

### Option B — scvelo velocity on aligned cells

```python
import scvelo as scv

scv.pp.filter_and_normalize(vel_c)
scv.tl.moments(vel_c)
scv.tl.velocity(vel_c)
scv.tl.velocity_graph(vel_c)
scv.pl.velocity_embedding_stream(vel_c, basis="umap")
```

---

## Notes on modality alignment

**`obs_names` are globally unique by construction.** The `<barcode>-<sample_id>`
scheme means two cells from different samples that happen to share the same
16-mer barcode become distinct entries (`ACGT…-s1` vs `ACGT…-s2`). Set
intersection is therefore unambiguous — no secondary key is needed.

**`obs["sample_id"]` is preserved across subsetting.** After
`adata = adata[common].copy()`, `adata.obs["sample_id"]` still holds the
correct per-cell sample label, suitable for use as a batch key in
downstream tools (`batch_key="sample_id"` in `scanpy.pp.highly_variable_genes`,
`sc.pp.combat`, `scvi.model.SCVI`, etc.).

**M2 validity after var-axis subsetting.** If you subset the splicing AnnData
on the var axis (e.g. `spl[:, mask]`), `adata.uns["splikit"]["m2_valid"]`
flips to `False`. Always call `splk.tl.make_m2` again on the subsetted object
before passing M2 downstream. See
[Recompute M2 after subsetting](recompute-m2-after-subsetting.md) for details.
