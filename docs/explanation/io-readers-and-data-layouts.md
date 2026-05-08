# STARsolo readers and AnnData data layouts

This page documents the three STARsolo readers in `splikit.io` — what each
one produces, how they compose into multi-modal pipelines, and why we
deliberately stay on AnnData (with `layers` for paired measurements) rather
than reaching for MuData. It is a design / explanation document; the
auto-generated reference docs are under [API Reference](../reference/io.md).

---

## Core design rule

**One container, one feature axis.** AnnData is the unit of `splikit.io`
output. When two measurements share the same feature axis (M1 + M2 over the
same junction-events; spliced + unspliced over the same gene), they go into
`adata.layers`, not into separate AnnData objects.

```
Same feature axis, paired measurements      →  AnnData with multiple layers
Different feature axes, shared cells          →  build two AnnDatas; user combines
```

This rule is not splikit-py-specific. It's the scverse convention:

| Pattern                              | Container         | Examples                                   |
|---|---|---|
| Paired measurements, same `var`      | `AnnData.layers`  | **M1 / M2** (splikit-py); **spliced / unspliced** (scvelo); raw / normalized / scaled |
| Distinct feature axes, shared cells  | (caller's choice) | gene + splicing; RNA + ATAC; RNA + protein |

For the second pattern, the caller can either keep two separate AnnData
objects and pass them into multi-modal tools (`multigedipy.MultiGEDIModel`,
custom code), or wrap them in `mudata.MuData` if their workflow benefits
from a single container. **`splikit.io` itself never produces MuData**;
that's a downstream composition decision, not an ingestion concern.

### Why not produce MuData

1. **scverse precedent.** `scvelo` produces AnnData with `layers["spliced"] /
   layers["unspliced"]`, not MuData. Splicing M1/M2 is the same shape of
   problem (paired counts on a common feature axis); the same answer applies.
2. **Single-modality users shouldn't pay a MuData tax.** Most splicing-only
   analyses don't need a multi-modal container. Forcing one would add
   indirection (`mdata.mod["splicing"].layers["M1"]` vs `adata.layers["M1"]`)
   for no gain.
3. **Cross-modality composition is a one-liner anyway.** If you have a gene
   AnnData and a splicing AnnData, `mudata.MuData({"rna": gex, "splicing":
   spl})` wraps them in a single line. We don't need to bake that into the
   reader.

---

## The three STARsolo readers

`STARsolo` writes three cell-by-feature matrices per sample, one per feature
type. `splikit.io` ships one reader for each, with a consistent API shape.

| Function                     | STARsolo source                                | Output shape                                                                      | Status |
|---|---|---|---|
| `splk.io.read_starsolo`      | `Solo.out/SJ/raw/` + `<sample_root>/SJ.out.tab` + `Solo.out/Gene/filtered/barcodes.tsv` (whitelist) | AnnData with `layers["M1"]`, `layers["M2"]`; events on `var`                       | **Implemented** |
| `splk.io.read_starsolo_gene` | `Solo.out/Gene/filtered/`                       | AnnData with raw counts in `X` (single matrix); genes on `var`                    | **Planned** |
| `splk.io.read_starsolo_velocyto` | `Solo.out/Velocyto/raw/`                    | AnnData with `layers["spliced"]`, `layers["unspliced"]`, `layers["ambiguous"]`; genes on `var` | **Planned** |

All three share:

- **Multi-sample concat.** Each reader accepts a sequence of sample dirs +
  matching `sample_ids`, reads each, then concatenates on the cell axis.
- **`<barcode>-<sample_id>` `obs_names` convention.** Each cell is uniquely
  identifiable across samples without modifying the raw barcode.
- **`obs["sample_id"]` categorical column.** This is the canonical per-cell
  sample label that splikit-py kernels and downstream tools (gedi2py /
  multigedipy `batch_key`, scvelo's normalization batch key) consume.
- **Whitelist semantics.** Each reader can apply an external `barcode_whitelist`
  per sample, or auto-discover STARsolo's internal `Gene/filtered/barcodes.tsv`
  whitelist when `use_internal_whitelist=True`. Cells outside the whitelist are
  dropped at ingestion.
- **Path-resolver flexibility.** Inputs can be the **sample root** (the
  directory containing `Solo.out/`), `Solo.out/`, or the matrix-feature-bearing
  subdirectory directly. The resolver walks the expected tree.

This consistency means the three readers compose naturally: feeding the
*same* `(sample_dirs, sample_ids)` tuple into all three gives you three
AnnDatas with **identical `obs_names`**, ready for multi-modal analysis.

---

## Per-reader AnnData layouts

### `splk.io.read_starsolo` (splicing)

```
adata
├── X = None
├── layers
│   ├── "M1"  (n_obs, n_vars) sparse CSC float64   inclusion counts
│   └── "M2"  (n_obs, n_vars) sparse CSC float64   exclusion counts (LJV-derived)
├── obs (n_obs)
│   ├── "barcode"     str                          raw 16-mer cell barcode
│   └── "sample_id"   pd.Categorical              "<sample_id>" per cell
├── obs_names                                      "<barcode>-<sample_id>"
├── var (n_vars)
│   ├── "chr"             pd.Categorical          chromosome
│   ├── "start"           int64                   junction start (1-based, STAR-native)
│   ├── "end"             int64                   junction end
│   ├── "strand"          pd.Categorical          "+", "-", "." (STAR strand)
│   ├── "intron_motif"    int8                    STAR motif code
│   ├── "is_annot"        int8                    annotated junction flag
│   ├── "unique_mapped"   int32                   unique-mapped read count
│   ├── "row_names_mtx"   str                     un-suffixed "chr:start-end"
│   ├── "group_id"        int32                   dense 0..G-1, LJV grouping key
│   ├── "group_kind"      pd.Categorical          "S" or "E" (start- or end-coord LJV)
│   └── "group_count"     int32                   members in this LJV (post-filter)
├── var_names                                      "<row_names_mtx>_S" or "<row_names_mtx>_E"
└── uns["splikit"]
    ├── "version"     int                          schema version
    ├── "m2_valid"    bool                         True after make_m2; flips False on var subset
    ├── "ljv_kind"    str                          "start_end" / "start" / "end"
    ├── "source"      str                          "starsolo"
    └── "params"      dict                         last-call settings per kernel
```

**Why two layers and not `X = M1, layers["M2"] = M2`?** Because M1 and M2 are
co-equal partners — neither is "the matrix"; both are required by every
downstream kernel (`make_m2` builds M2 from M1 + group_id; `find_variable_events`
needs both; `pseudo_correlation` needs both). Putting M1 in `X` would invite
`scanpy.pp.normalize_total(adata)` to silently mutate inclusion counts and
break the M2 invariant. With `X = None`, the structure is unambiguous: you
must address layers explicitly.

**Key invariant: `var_names` must remain unique with the `_S` / `_E`
suffix.** The same junction `chr:start-end` can appear twice as a var row
when it qualifies for both a start-coordinate LJV and an end-coordinate LJV
(different M2 values). The suffix is what disambiguates. **Never call
`var_names_make_unique()`** — it would mangle the suffix scheme by appending
`-1` / `-2`. The kernels check this invariant on entry.

**Why `row_names_mtx` is un-suffixed.** Atomicity: the suffix is an
*orthogonal* axis of the event identity and should live in its own column
(`group_kind`). The fully-suffixed name is recoverable from `var_names`;
the un-suffixed `row_names_mtx` is what you join on if you want to find
"all events for junction `chr:start-end` regardless of LJV side." This is
a deliberate divergence from R splikit (which keeps the suffix in
`row_names_mtx`); both representations are reversible, but atomic columns
are the scverse-idiomatic shape.

### `splk.io.read_starsolo_gene` (gene expression)

```
adata
├── X         (n_obs, n_vars) sparse CSC float64    raw UMI counts
├── obs (n_obs)
│   ├── "barcode"     str
│   └── "sample_id"   pd.Categorical
├── obs_names                                       "<barcode>-<sample_id>"
├── var (n_vars)
│   ├── "gene_id"        str       Ensembl / NCBI gene id (used as var_names)
│   └── "gene_name"      str       gene symbol
├── var_names                                       gene_id (set by `var_names="gene_ids"`)
└── uns["splikit"]
    ├── "source"   str             "starsolo"
    └── "params"   dict
```

Single-layer, single-matrix. Counts go in `X` directly because there's no
paired companion measurement on the same axis. Drop-in for
`scanpy.pp.normalize_total`, `scanpy.pp.highly_variable_genes`,
`scvi-tools.SCVI`, etc.

**Choice of `var_names`:** default to `gene_id` (Ensembl IDs are stable and
unique). `gene_name` (symbols) goes in a sibling column; symbols are not
guaranteed unique across the genome (e.g. `IGH@`) so they're poor `var_names`.
The reader will accept `var_names="gene_symbols"` for users who need
symbol-keyed AnnDatas, with `var_names_make_unique()` allowed in that mode
because symbols don't carry the load-bearing structure that splicing
suffixes do.

### `splk.io.read_starsolo_velocyto`

```
adata
├── X         (n_obs, n_vars) sparse CSC float64    spliced counts (alias of layers["spliced"])
├── layers
│   ├── "spliced"     (n_obs, n_vars) sparse CSC float64
│   ├── "unspliced"   (n_obs, n_vars) sparse CSC float64
│   └── "ambiguous"   (n_obs, n_vars) sparse CSC float64
├── obs (n_obs)
│   ├── "barcode"     str
│   └── "sample_id"   pd.Categorical
├── obs_names                                       "<barcode>-<sample_id>"
├── var (n_vars)
│   ├── "gene_id"        str
│   └── "gene_name"      str
├── var_names                                       gene_id
└── uns["splikit"]
    ├── "source"   str             "starsolo"
    └── "params"   dict
```

Three paired layers on the gene axis. `X` is set to a copy of `layers["spliced"]`
so callers that only know about `X` (some scvelo helpers) work transparently;
callers that need the unspliced and ambiguous slabs reach into `layers`.

**STARsolo's MTX wire format is unusual here.** `Solo.out/Velocyto/raw/matrix.mtx`
contains all three matrices stacked: rows `[0, n_genes)` are spliced, rows
`[n_genes, 2 * n_genes)` are unspliced, rows `[2 * n_genes, 3 * n_genes)` are
ambiguous. The reader parses this into the three layers above. This is the same
unpacking scvelo's loaders do when they read STARsolo Velocyto output.

**Drop-in for scvelo:**

```python
adata = splk.io.read_starsolo_velocyto(...)
import scvelo as scv
scv.pp.filter_and_normalize(adata)
scv.tl.velocity(adata)        # works as-is; scvelo expects exactly this layout
scv.tl.velocity_graph(adata)
```

---

## Common conventions across all three readers

### `obs_names` → `<barcode>-<sample_id>`

Set at ingestion in every reader. Two consequences:

1. **Cross-modality cell intersection is set-based.** Given a splicing AnnData
   and a gene AnnData built from the same `(sample_dirs, sample_ids)`,
   `set(spl.obs_names) & set(gex.obs_names)` is guaranteed to be the
   intersection of the two whitelists. Most often, identical.
2. **Single-cell barcode collisions across samples become impossible.** Two
   different cells from samples A01 and B01 that happen to share the same
   16-mer barcode become distinct entries: `ACGT…-A01` vs `ACGT…-B01`.

### Strand encoding (`read_starsolo` only)

STARsolo writes strand as `0` (undefined), `1` (+), `2` (-) in `SJ.out.tab`.
The reader decodes to `"."`, `"+"`, `"-"` strings and stores `var["strand"]`
as a `pd.Categorical`. Downstream tools see canonical-strand strings, never
the integer codes.

### Whitelist semantics

The `barcode_whitelists` argument is per-sample. Each entry is one of:

- `None` — no explicit whitelist for this sample; defer to
  `use_internal_whitelist`.
- A `Path` to a one-barcode-per-line file.
- A `Sequence[str]` of raw barcodes.

When `use_internal_whitelist=True` (default), missing slots fall back to
`<sample_root>/Solo.out/Gene/filtered/barcodes.tsv` per sample. If neither an
explicit whitelist nor the internal one is available, the reader falls back to
the full barcode set in `Solo.out/<feature>/raw/barcodes.tsv`.

### `min_counts` filter (where applicable)

- `read_starsolo`: filters events whose `M1.sum(axis=0) >= min_counts` in
  any cell (default `min_counts=1` drops zero-count events, which only
  exist as zero-padded entries from the multi-sample union).
- `read_starsolo_gene`, `read_starsolo_velocyto`: do not apply a
  `min_counts` filter at ingestion. Gene and Velocyto users have their own
  HVG / normalization workflows downstream (`scanpy.pp.highly_variable_genes`,
  `scvelo.pp.filter_and_normalize`); the reader stays inert.

### `uns["splikit"]["params"]`

Each reader records the call's params in `uns["splikit"]["params"][reader_name]`
so the AnnData carries enough metadata to reproduce the ingestion. This
mirrors the scanpy convention (`uns["pca"]`, `uns["neighbors"]`, etc.).

---

## Composition into multi-modal pipelines

splikit-py produces three independent AnnData objects; the user assembles
them into a multi-modal pipeline. There are two common patterns:

### Pattern 1 — explicit list, into `multigedipy.MultiGEDIModel`

This is what the validation/e2e walkthrough uses today. The two AnnData
objects stay separate; both are aligned to the same cells; their per-modality
data is handed to `add_modality()`.

```python
from multigedipy import MultiGEDIModel
import scipy.sparse as sp
import numpy as np

spl = splk.io.read_starsolo(sample_dirs, sample_ids)
splk.tl.make_m2(spl, n_threads=8)
splk.pp.highly_variable_events(spl, n_top=10_000, sample_key="sample_id", inplace=True)
spl = spl[:, spl.var["highly_variable"]].copy()
splk.tl.make_m2(spl, n_threads=8)

gex = splk.io.read_starsolo_gene(sample_dirs, sample_ids)   # planned
gex = gex[:, gex.var["highly_variable_top_5000"]].copy()    # via scanpy.pp.highly_variable_genes

# Cell-paired alignment
common = sorted(set(spl.obs_names) & set(gex.obs_names))
spl, gex = spl[common, :].copy(), gex[common, :].copy()

sample_vec = np.asarray(spl.obs["sample_id"].astype(str))
M1 = sp.csc_matrix(spl.layers["M1"].T, dtype=np.float64)   # events × cells
M2 = sp.csc_matrix(spl.layers["M2"].T, dtype=np.float64)
G  = sp.csc_matrix(gex.X.T, dtype=np.float64)               # genes × cells

model = MultiGEDIModel(K=20, mode="Bsphere", seed=42, num_threads=8)
model.add_modality("gene_expression", data=G, sample_vec=sample_vec, obs_type="M",
                   orthoZ=True)
model.add_modality("splicing", data=(M1, M2), sample_vec=sample_vec,
                   obs_type="M_list", orthoZ=False, is_si_fixed=True)
model.train(iterations=30)
```

This is the pattern used in `validation/e2e/run_multimodal_pipeline.py`.

### Pattern 2 — one-modality-at-a-time scanpy / scvelo

When you only care about one modality, you don't need any composition:

```python
spl = splk.io.read_starsolo(...)
splk.tl.make_m2(spl); splk.pp.highly_variable_events(spl); ...
```

Or for velocyto:

```python
adata = splk.io.read_starsolo_velocyto(...)
import scvelo as scv
scv.pp.filter_and_normalize(adata); scv.tl.velocity(adata)
```

No multi-modal infrastructure required.

---

## Why this shape works for splikit-py

The kernels in `splk.tl` and `splk.pp` are tuned for the **paired layers**
shape:

- `splk.tl.make_m2` — reads `layers["M1"]`, writes `layers["M2"]`. Treats
  them as a paired pair on the same `var` axis. ✓
- `splk.pp.highly_variable_events` — reads both layers, splits per-library
  via `obs["sample_id"]`, computes per-event sum-deviance. ✓
- `splk.tl.pseudo_correlation` — reads both layers, fits per-event IRLS
  against an external Z. ✓

If M1 and M2 lived in separate AnnData objects (or a MuData), every kernel
would have to reach into two containers and re-align by `var_names`. That's
the pattern MuData was designed for *across* modalities (RNA + ATAC),
*not within* one modality's paired measurements. Layers exist for this
exact reason; we use them as intended.

---

## Implementation status

| Reader                       | Status         | Tests | R-equivalence |
|---|---|---|---|
| `splk.io.read_starsolo`      | Implemented (`src/splikit/io/_starsolo.py`) | 11 synthetic + bit-exact e2e on real samples on `validation` branch | Bit-exact M1, M2, eventdata vs `splikit::make_junction_ab + make_m1 + make_m2` |
| `splk.io.read_starsolo_gene` | Planned        | TBD                                              | Mechanical I/O; R parity by construction |
| `splk.io.read_starsolo_velocyto` | Planned    | TBD                                              | Mechanical I/O; R parity by construction |

The two planned readers follow the same internal layout as `_starsolo.py`:

```
src/splikit/io/_<reader>.py
├── _resolve_<reader>_paths(sample_dir)            walks the STARsolo tree
├── _read_one_sample(paths, sample_id, ...)        reads MTX + barcodes + features for one sample
├── _concat_samples([artifacts])                    multi-sample union, zero-padded
└── read_starsolo_<reader>(sample_dirs, sample_ids, ...)   public entry point
```

This symmetry means a user who has internalized one reader's API has
internalized all three; only the per-reader output schema differs (one
layer for gene, three for velocyto, two-plus-LJV-grouping for SJ).

### What's deliberately *not* in this scope

- **Normalization.** Counts go in raw; users normalize via `scanpy` / `scvelo`.
- **HVG / HVE selection inside the reader.** `read_starsolo` doesn't run
  `splk.pp.highly_variable_events` automatically — that's a separate step
  callers compose. Same will be true of the gene and velocyto readers.
- **Cross-modality alignment inside the reader.** When you call all three
  readers on the same `(sample_dirs, sample_ids)`, the resulting AnnDatas
  are *already* aligned on `obs_names`. We don't bake a "give me all three
  in a MuData" convenience because that's a one-line composition the user
  controls.
- **Loom / h5ad / 10x .h5 inputs.** These readers target STARsolo MTX output
  specifically. Users with other formats use `scanpy.read_*` and assemble
  the splikit-py-shaped AnnData manually (the schema is documented above
  in full).

---

## Related modules

- [`src/splikit/io/_starsolo.py`](https://github.com/Arshammik/splikitpy/blob/main/src/splikit/io/_starsolo.py)
  — current splicing reader, the layout reference for the planned readers.
- [`src/splikit/_core/_validators.py`](https://github.com/Arshammik/splikitpy/blob/main/src/splikit/_core/_validators.py)
  — `validate_var_schema`, `validate_paired_layers`, `mark_m2_valid`,
  `invalidate_m2`. Enforces the schema invariants documented above.
- [`docs/explanation/data-model.md`](data-model.md) — the LJV concept
  (the splicing-specific data model that motivates the M1 / M2 paired layers).

The cross-language regression suite (R splikit ↔ splikit-py) lives on the
[`validation` branch](https://github.com/Arshammik/splikitpy/tree/validation).
