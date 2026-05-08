# R reference fixtures

Two HDF5 fixtures used by the cross-language equivalence tests:

| Fixture | Generator | Committed? | Tests |
|---|---|---|---|
| `tests/data/r_reference.h5` | `export_reference.R` (toy 2000×2000) | Yes (small) | `tests/test_make_m2.py::test_make_m2_bit_exact_vs_r` etc. |
| `tests/data/r_reference_merfish.h5` | `export_merfish_reference.R` (real, ~80 MB) | No (.gitignore) | `tests/test_r_equivalence_real.py` |

## Regenerating the MERFISH fixture (real data)

The MERFISH fixture is a single-sample slice of a trained MultiGEDI model (12,774 events × 835 cells from sample `SRR26528546`). The full M1/M2 RDS files plus the saved model are too large to ship; they live on the cluster only. So this fixture must be regenerated on a compute node with R + splikit + rhdf5 + access to those RDS files.

```sh
ssh c170    # or any compute node with the RDS files visible
cd /home/arsham79/projects/rrg-hsn/arsham79/splikitpy
module load r/4.4.0
Rscript tests/r_export/export_merfish_reference.R \
    --out tests/data/r_reference_merfish.h5
```

Override defaults via `--model PATH`, `--m1 PATH`, `--eventdata PATH`, `--sample SRR<id>`, `--min-row-sum N`. Run the script with `--help` for the full list.

The script:

1. Loads the MultiGEDI model and pulls `model$projections$ZDB(modalities="Splicing", samples=<sample>)`.
2. Slices the full M1 to those events × cells.
3. Recomputes M2 freshly via `splikit::make_m2(use_cpp=TRUE, n_threads=1)` on the *subset* eventdata. M2 is path-A (subset-derived), not the M2 the model originally trained on (which embeds full-data co-members no longer present).
4. Runs `find_variable_events(min_row_sum=1, n_threads=1)` and `get_pseudo_correlation` (CoxSnell + Nagelkerke) on the subset.
5. Remaps `eventdata$group_id` to dense int32 0..G-1 (mirrors `splikit/R/star_solo_processing.R:655-657`).
6. Writes everything to HDF5 with provenance attributes (R version, splikit version, BLAS, generated_at, sample_id, min_row_sum).

Expected runtime: 3–8 min on c170 (dominated by the M1 RDS load and the IRLS loop in `get_pseudo_correlation`).

## Why this fixture isn't committed

ZDB alone is 12,774 × 835 × 8 ≈ 81 MB. Plus M1/M2/eventdata + the metadata, the file lands at ~100 MB. Commit-via-LFS is workable but adds friction; for now the file lives only on c170 and CI regenerates on demand. Tests skip cleanly when the fixture is missing.

## Tolerance bands

Spelled out in the test docstrings; refer to `tests/test_r_equivalence_real.py`. Summary:

- `make_m2`: bit-exact via `np.array_equal`.
- `highly_variable_events`: `np.allclose(rtol=1e-10, atol=1e-12)` plus exact NaN-mask match.
- `pseudo_correlation`: `np.allclose(rtol=1e-7, atol=1e-9)` plus exact NaN-mask match (IRLS path-divergence tolerance band).

## Regenerating the toy fixture (synthetic)

Smaller, committable, gated behind `pytest.mark.r_required`:

```sh
module load r/4.4.0
Rscript tests/r_export/export_reference.R tests/data/r_reference.h5
```

Uses `splikit::load_toy_M1_M2_object()` so it needs only R splikit installed; no cluster RDS access required.
