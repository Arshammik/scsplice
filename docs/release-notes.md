# Release notes

## v0.1.0.dev0 (in development)

Initial pre-alpha development release. Not yet published to PyPI.

**Scope:** six public functions across `io`, `tl`, and `pp`.

**Cross-language parity:** M2 bit-exact vs. R splikit; HVE deviance `rtol=1e-10`; pseudo-correlation `rtol=1e-7`. The full regression suite lives on the [`validation` branch](https://github.com/Arshammik/splikitpy/tree/validation).

### New in commit `4a32cc2` (on `main`)

**`read_starsolo_gene`** (`src/splikit/io/_starsolo_gene.py`)

Reads `Solo.out/Gene/{raw,filtered}/` into a cell × gene AnnData with raw UMI counts in `X`. Drop-in for `scanpy.pp.normalize_total`, `scanpy.pp.highly_variable_genes`, and `scvi-tools`. Supports `tissue_positions=` for spatial samples with full squidpy `obsm["spatial"]` and `uns["spatial"]` population.

**`read_starsolo_velocyto`** (`src/splikit/io/_starsolo_velocyto.py`)

Reads `Solo.out/Velocyto/raw/` into a cell × gene AnnData with `layers["spliced"]`, `layers["unspliced"]`, `layers["ambiguous"]`. `X` is aliased to `layers["spliced"]` for scvelo drop-in. Handles both modern split-file layout (STARsolo 2.7.10b+: three sibling `.mtx` files) and legacy stacked `matrix.mtx` automatically.

**External whitelist / spatial whitelist refactor** (`src/splikit/io/_whitelist.py`)

Centralised per-sample whitelist resolution shared by all three readers. Strict four-level precedence: `tissue_positions > explicit barcode_whitelist > internal filtered/ > raw`. When `tissue_positions=` is given, the reader sources from `raw/` (not `filtered/`) and trims to the spatial spot set — necessary because Visium whitelists are derived from the spot grid, not the STARsolo knee-point algorithm.

All readers gained `tissue_positions=` and `spatial_library_ids=` kwargs with identical semantics. When `tissue_positions` is provided, the reader populates squidpy-compatible: `obs["in_tissue"]` (int8), `obs["array_row"]` / `obs["array_col"]` (int32), `obsm["spatial"]` ((n_obs, 2) float64), and `uns["spatial"][library_id]`. Space Ranger v1 (no-header `tissue_positions_list.csv`) and v2 (header `tissue_positions.csv`) are auto-detected.

For the full commit history, see the [GitHub repository](https://github.com/Arshammik/splikitpy/commits/main).

---

Formal versioned release notes will be published here starting with v1.0.
