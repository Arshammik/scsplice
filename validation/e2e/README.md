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

## Integration notes (filled in after the actual gedi2py run)

_(empty until C.4 runs)_
