# End-to-end validation: samples A01 + B01

Live pipeline test: starts from raw STARsolo `Solo.out/SJ/` directories,
runs M1/M2 creation in **both R splikit and Python splikit-py**, and asserts
bit-exact agreement on the resulting matrices + eventdata.

## What this validates

| Layer | Tested |
|---|---|
| `splk.io.read_starsolo` | full ingest of two real human-brain samples (A01, B01) including STARsolo SJ MTX, barcodes, SJ.out.tab, internal Gene/filtered whitelist |
| LJV grouping | start-coord `_S` + end-coord `_E` events, dense `group_id` int32 0..G-1, `group_count` invariants, var-name suffix scheme |
| `splk.tl.make_m2` | bit-exact agreement with R `splikit::make_m2(use_cpp=TRUE)` on real data, on a CSC the size of an actual scRNA-seq sample |
| Cross-language obs alignment | `<barcode>-<sample_id>` convention round-trips between the R wrapper's regex and the Python reader's explicit `sample_id` column |
| gedi2py integration | (`test_gedi2py_integration.py`, separate file) the splikit-py AnnData feeds straight into `gd.tl.gedi(layer="M1", layer2="M2", batch_key="sample_id")` with no adapter |

## Inputs (shared FS, c170)

- `/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_A01_S01_L003/`
- `/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_B01_S01_L003/`

Each has `Solo.out/SJ/raw/{matrix.mtx, barcodes.tsv}`, `Solo.out/SJ.out.tab`,
and `Solo.out/Gene/filtered/barcodes.tsv` (the whitelist).

## How to run

From the repo root on c170:

```sh
bash validation/e2e/run_all.sh
```

Sequence:

1. **`run_r_pipeline.R`** — `splikit::make_junction_ab` + `make_m1` + `make_m2` on both samples with the Gene/filtered whitelist; exports CSC slots + eventdata + obs to `data/r_pipeline.h5`.
2. **`run_py_pipeline.py`** — `splk.io.read_starsolo(use_internal_whitelist=True)` + `splk.tl.make_m2(n_threads=8)`; writes `data/py_pipeline.h5ad`.
3. **`compare_outputs.py`** — aligns events on `(row_names_mtx, group_kind)` keys and obs on `obs_names`; bit-exact equality check on M1, M2, eventdata.

Expected runtime: 5–15 min, dominated by the R pipeline's eventdata build on
~600K junctions.

## Outputs

Both pipelines write under `validation/e2e/data/` (gitignored — files are
~hundreds of MB).

## Mission-critical agent checkpoints

The plan called for three coordination points (per the user's directive to
work with agents and ask their opinion before each mission-critical action):

1. ✓ Before writing `run_r_pipeline.R` — `R-rcpp-compuational-biologiest` agent confirmed the R splikit 2.2.1 API surface for `make_junction_ab` + `make_m1` + `make_m2`, including the `white_barcode_lists` semantics (a list of vectors, not paths) and the dense-int32 group_id remap.
2. (deferred) Before `compare_outputs.py` — tolerance bands chosen as bit-exact (`np.array_equal` everywhere). The kernel is documented bit-exact and the rest is integer re-encoding; no transcendentals or iterative solvers in this pipeline. If a band fails on actual run, `cross-language-numerical-equivalence-engineer` handles the diagnosis.
3. Before `test_gedi2py_integration.py` — `scverse-python-architect` will be re-engaged after step 2 produces the actual h5ad to confirm AnnData layout matches gedi2py's runtime expectations on real data, and to surface any layout improvements worth claiming. Findings get appended below as "Integration notes".

## E2E result on samples A01 + B01

| Field | Result |
|---|---|
| Events | 281,735 (R + Py identical) |
| Cells | 14,570 (5,604 A01 + 8,966 B01 after Gene/filtered whitelist) |
| M1 nnz | 26,412,664 — bit-exact (`np.array_equal`) |
| M2 nnz | 55,402,569 — bit-exact (`np.array_equal`) |
| `row_names_mtx` | equal (after stripping R's `_S`/`_E` suffix; see schema notes) |
| `group_kind` | equal |
| `group_id` partitions | equal (129,102 LJV groups) |
| `group_count` | equal post-filter; R's stored value differs in 83,879/281,735 events (it stores pre-filter) |
| `sample_id` | equal |

## Integration notes (gedi2py compatibility on real data)

The `scverse-python-architect` agent inspected the real-data h5ad
(`data/py_pipeline.h5ad`, 14,570 cells × 281,735 events) against gedi2py's
runtime contract (`gedi2py/src/gedi2py/_core/_model.py`). **Verdict: the
splikit-py AnnData is directly consumable by `gd.tl.gedi(layer="M1",
layer2="M2", batch_key="sample_id")` with no adapter.**

Contract checks (all pass):

* L155–164: layer membership in `adata.layers` — pass.
* L171–172, 201–211: per-sample CSC slicing handles the row-slice → CSR
  transient via the explicit `sp.csc_matrix(...)` cast.
* L177: `X = adata.X` is only read when `layer is None`. With both `layer`
  and `layer2` provided, `X = None` is **safe**. No other gedi2py codepath
  touches `adata.X` before the `M_paired` branch.
* L183: `obs[batch_key].astype(str)` works on splikit-py's categorical
  `sample_id` column.
* M1 / M2 must be shape-aligned (gedi2py asserts this); the `m2_valid`
  invariant on splikit-py's side guarantees it.

### HVE pre-filter sizing

gedi2py densifies M1 and M2 per-sample. For two samples with 5,604 + 8,966
cells, peak densification ≈ 260,016 × `n_events` bytes (retained `Yi_list`
+ momentary `M1i_dense + M2i_dense` for the largest sample).

| `n_top` | peak densify | budget |
|---|---|---|
| full (281,735) | ~73 GB | OOM |
| 50,000 | ~13 GB | tight |
| **20,000 (used)** | **~5.2 GB** | comfortable |
| 10,000 | ~2.6 GB | possibly under-powered |

`test_gedi2py_integration.py` uses `n_top=20_000`.

### Schema divergences from R splikit (claimed; up for discussion)

| field | R splikit | splikit-py | recommendation |
|---|---|---|---|
| `var['group_count']` | pre-filter (counts original LJV co-members at `make_m1` time) | post-filter (matches the matrix on disk after `min_counts`) | **keep splikit-py** |
| `var['row_names_mtx']` | suffixed with `_S` / `_E` | un-suffixed (suffix lives in `var_names` + `var['group_kind']`) | **keep splikit-py** |

Argument for both: `var` columns should describe the data on disk (atomic,
queryable, accurate). R's choices are upstream-data-table accidents that
don't carry well into the AnnData / scverse worldview. Both are reversible
(R's pre-filter `group_count` is deterministically recoverable from the
eventdata at ingest time; the suffix is recoverable from `var_names` or
`var['group_kind']`), so the claim costs nothing.

### Deferred to a future PR

Three nice-to-haves the architect surfaced for tightening multi-modal /
introspection ergonomics:

```python
adata.uns['splikit']['layers'] = {'M1': 'inclusion', 'M2': 'lvm_complement'}
adata.uns['splikit']['m1_nnz'] = int(M1.nnz)
adata.uns['splikit']['m2_nnz'] = int(M2.nnz)
```

And a `feature_type='splice_junction'` column on `var` so multi-modal
`MuData` embeds can disambiguate splicing from gene-expression modalities
by inspection rather than by naming convention.
