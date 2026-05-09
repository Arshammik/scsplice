# Read spatial data with tissue_positions

This how-to shows how to load a Visium sample using a Space Ranger
`tissue_positions.csv`, produce a squidpy-ready AnnData, and handle mixed
spatial / non-spatial sample concatenation.

For the design rationale behind the whitelist precedence rule see
[STARsolo readers and AnnData data layouts](../explanation/io-readers-and-data-layouts.md#spatial-tissue_positions-and-squidpy-compatibility).

---

## 1. Read a single Visium sample

```python
import splikit as splk

gex = splk.io.read_starsolo_gene(
    sample_dirs=["path/to/visium_sample"],      # (1)
    sample_ids=["vis1"],
    tissue_positions=[
        "path/to/visium_sample/outs/tissue_positions.csv"  # Space Ranger 2.x
    ],
    spatial_library_ids=["vis1"],               # (2)
    verbose=True,
)
```

1. The reader accepts the sample root (parent of `Solo.out/`), `Solo.out/`,
   `Solo.out/Gene/`, or `Solo.out/Gene/raw/` directly.
2. `spatial_library_ids` sets the key under `uns["spatial"]`. Defaults to
   `sample_ids[i]` when omitted.

!!! note "Space Ranger v1 vs v2"
    Both `tissue_positions.csv` (v2, with header) and
    `tissue_positions_list.csv` (v1, no header) are detected automatically
    from the file's first row. Pass whichever path you have.

After the call, the AnnData carries:

```python
gex.obs[["in_tissue", "array_row", "array_col"]].head()
gex.obsm["spatial"]        # (n_obs, 2) float64 — pixel coordinates
gex.uns["spatial"]["vis1"] # squidpy-shaped scaffold
```

Only cells present in `tissue_positions.csv` are kept. Counts are read from
`Gene/raw/` and intersected — never from `Gene/filtered/` — because the Visium
spot whitelist and STARsolo's knee-point filter are derived independently.

---

## 2. Verify with squidpy

```python
import squidpy as sq

# squidpy expects exactly this layout; no adapter needed.
sq.pl.spatial_scatter(gex, color="in_tissue", library_id="vis1")
sq.gr.spatial_neighbors(gex, library_id="vis1", coord_type="visium")
```

---

## 3. Subset to in-tissue spots

The reader keeps all barcodes from the tissue-positions file, including spots
where `in_tissue == 0` (fiducials, background). Filter before downstream analysis:

```python
gex_tissue = gex[gex.obs["in_tissue"] == 1].copy()
print(f"{gex_tissue.n_obs} in-tissue spots retained")
```

---

## 4. Multi-sample: mixed spatial and non-spatial

When some samples have `tissue_positions` and others do not, pass `None` for
the non-spatial slots:

```python
gex = splk.io.read_starsolo_gene(
    sample_dirs=["visium_sample", "dropseq_sample"],
    sample_ids=["vis1", "ds1"],
    tissue_positions=[
        "visium_sample/outs/tissue_positions.csv",
        None,           # non-spatial sample; uses internal filtered/ whitelist
    ],
    spatial_library_ids=["vis1", None],
)
```

The merged AnnData assigns sentinel values for non-spatial cells:

| Column | Spatial | Non-spatial |
|---|---|---|
| `obs["in_tissue"]` | `0` or `1` | `-1` |
| `obs["array_row"]` | grid row | `-1` |
| `obs["array_col"]` | grid column | `-1` |
| `obsm["spatial"][i]` | pixel coords | `(-1.0, -1.0)` |

Filter to spatial cells only with `adata[adata.obs["in_tissue"] >= 0]`.

---

## 5. Same pattern for the splicing and velocyto readers

The same `tissue_positions=` and `spatial_library_ids=` kwargs work identically
on `read_starsolo` (splicing) and `read_starsolo_velocyto`:

```python
spl = splk.io.read_starsolo(
    sj_dirs=["visium_sample/Solo.out/SJ"],
    sample_ids=["vis1"],
    tissue_positions=["visium_sample/outs/tissue_positions.csv"],
    spatial_library_ids=["vis1"],
)
vel = splk.io.read_starsolo_velocyto(
    sample_dirs=["visium_sample"],
    sample_ids=["vis1"],
    tissue_positions=["visium_sample/outs/tissue_positions.csv"],
    spatial_library_ids=["vis1"],
)

# All three share obs_names → direct intersection
common = sorted(set(spl.obs_names) & set(gex.obs_names) & set(vel.obs_names))
spl, gex, vel = spl[common].copy(), gex[common].copy(), vel[common].copy()
```

---

## Why not read from `filtered/` for spatial?

STARsolo's `Gene/filtered/barcodes.tsv` is determined by an unsupervised
knee-point algorithm on per-barcode UMI counts. Visium spots in low-RNA
tissue regions (e.g. white matter in brain sections) can have UMI counts
below the knee and will be dropped by `filtered/` — even though they are
valid tissue spots. The tissue-positions whitelist is derived from the
physical spot grid, which is the correct authority for Visium data.

When `tissue_positions=` is provided, every reader sources from `raw/`
(the full barcode set) and intersects with the spot whitelist. This
guarantees zero false negatives from the STARsolo filtering step.

See [STARsolo readers and AnnData data layouts — whitelist precedence](../explanation/io-readers-and-data-layouts.md#whitelist-precedence-per-sample)
for the four-level precedence rule.
