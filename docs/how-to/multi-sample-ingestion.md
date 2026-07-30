# Multi-sample ingestion

`read_starsolo` accepts one or more STARsolo SJ directories in a single call. All samples are processed together: junctions are unioned across samples (zero-padded where absent), barcodes are namespaced by `sample_id`, and LJV grouping is applied to the pooled junction table.

## Basic usage

```python
import scsplice as scs

adata = scs.io.read_starsolo(
    sj_dirs=[
        "run1/Solo.out/SJ",
        "run2/Solo.out/SJ",
        "run3/Solo.out/SJ",
    ],
    sample_ids=["donor_A", "donor_B", "donor_C"],
)

print(adata.obs["sample_id"].value_counts())
# donor_A    4821
# donor_B    5103
# donor_C    3977
```

`adata.obs_names` are `f"{barcode}-{sample_id}"`, so barcodes that collide across samples remain distinct.

## Directory layout accepted

`read_starsolo` accepts two path forms for each directory in `sj_dirs`:

- `<run>/Solo.out/SJ/` — with a `raw/` subdirectory containing `matrix.mtx` and `barcodes.tsv`.
- `<run>/Solo.out/SJ/raw/` — pointing directly to the `raw/` subdirectory.

It locates `SJ.out.tab` two levels above the `raw/` directory (i.e. at `Solo.out/SJ.out.tab`).

## Per-sample barcode whitelists

By default (`use_internal_whitelist=True`), each sample's barcode list is filtered using STARsolo's own `Gene/filtered/barcodes.tsv`. Pass explicit per-sample whitelists to override:

```python
adata = scs.io.read_starsolo(
    sj_dirs=["run1/Solo.out/SJ", "run2/Solo.out/SJ"],
    sample_ids=["s1", "s2"],
    barcode_whitelists=[
        "whitelists/s1_barcodes.txt",  # path to a one-barcode-per-line file
        None,                           # fall back to internal whitelist for s2
    ],
)
```

Each entry in `barcode_whitelists` can be:

- A path (`str` or `pathlib.Path`) to a one-barcode-per-line file (optionally gzip-compressed).
- A `list[str]` of raw 16-mer barcodes.
- `None` to defer to `use_internal_whitelist` for that sample.

## Handling samples with no overlapping junctions

Samples that share zero junctions still work correctly. Zero-padding fills in the missing rows so the global junction table is always the union. Events present in only one sample will have zero M1 counts in all other samples — they survive `min_counts=1` only if at least one cell in the contributing sample has a non-zero count.

To drop events that appear in fewer than K samples, filter after ingestion:

```python
import numpy as np

# Keep events present (non-zero row sum per sample) in at least 2 of 3 samples
sample_ids = adata.obs["sample_id"].cat.categories.tolist()
per_sample_nonzero = np.column_stack([
    np.asarray((adata[adata.obs["sample_id"] == sid, :].layers["M1"] > 0).sum(axis=0)).ravel() > 0
    for sid in sample_ids
])
n_samples_present = per_sample_nonzero.sum(axis=1)
adata = adata[:, n_samples_present >= 2].copy()
```

Then recompute M2 — see [Recompute M2 after subsetting](recompute-m2-after-subsetting.md).

## Large cohorts: memory considerations

Each sample's junction matrix is fully loaded into memory before merging. For cohorts with many samples, memory scales with the total number of unique junctions times total cells. As a rough guide:

| Samples | Cells / sample | Unique junctions | Peak RAM |
|---|---|---|---|
| 2 | 5 000 | 50 000 | ~500 MB |
| 10 | 5 000 | 80 000 | ~4 GB |
| 50 | 5 000 | 120 000 | ~20 GB |

If memory is a constraint, load samples individually, subset to high-quality junctions, then merge the subsets. The function does not currently support on-disk streaming.
