# Data model

## Local junction variants (LJVs)

A **local junction variant** (LJV) is a group of splice junctions that compete for the same splice site. Two junctions belong to the same LJV group when they share either:

- The same donor site (chromosome + start coordinate + strand) — labelled `_S` (start).
- The same acceptor site (chromosome + end coordinate + strand) — labelled `_E` (end).

For example, consider three junctions on chr1 with strand `+`:

```
Junction A: chr1:1000-2000
Junction B: chr1:1000-2500   (same start as A → share an _S group)
Junction C: chr1:800-2000    (same end as A → share an _E group)
```

Junction A participates in two distinct LJV competitions:
- Against B for the start site at chr1:1000 → one `_S` event row.
- Against C for the end site at chr1:2000 → one `_E` event row.

This is the reason a single junction can appear **twice** in `adata.var` — once with suffix `_S`, once with `_E`. The two rows represent the same physical junction in different competitive contexts, and they will have different M2 values.

## The M1 and M2 layers

- **M1** (`layers["M1"]`, cells × events, sparse CSC float64): inclusion counts. For event `i`, `M1[j, i]` is the number of reads from cell `j` that crossed junction `i`.
- **M2** (`layers["M2"]`, same shape): exclusion counts. For event `i` and cell `j`:

```
M2[j, i] = sum(M1[j, k] for k in LJV group of i, k ≠ i)
```

M2 is the sum of all competing junction counts within the same LJV group, minus the event itself. It represents the "other side" of the splicing ratio.

Together, M1 and M2 define the **PSI** (percent spliced in) for each event:

```
PSI[j, i] = M1[j, i] / (M1[j, i] + M2[j, i])
```

Cells where `M1[j,i] + M2[j,i] == 0` are undefined (no coverage) and are excluded from all downstream calculations.

## The `_S` / `_E` suffix scheme

`var_names` are formatted as:

```
{chr}:{start}-{end}_{kind}
```

where `kind` is `S` (start-grouped) or `E` (end-grouped). Examples:

```
chr1:1000-2000_S   # Junction chr1:1000-2000 in its start-group competition
chr1:1000-2000_E   # Same junction in its end-group competition
chr1:1000-2500_S   # Competing junction (shares start with the above)
```

This naming scheme guarantees globally unique `var_names` by construction. **Never call `var_names_make_unique` on an AnnData produced by `read_starsolo`** — it would corrupt the names by appending numeric suffixes to the duplicate base names, breaking the correspondence between rows and the junction coordinates.

## The `group_id` column

`adata.var["group_id"]` is a dense integer array of shape `(n_vars,)` with values in `0..G-1`, where `G` is the number of distinct LJV groups. Events in the same group share the same `group_id`.

The `_S` and `_E` group-ID spaces are **disjoint** — a junction's start-group and end-group always receive different `group_id` values, even when the underlying junction coordinates overlap. This ensures that the M2 kernel sums only within the correct competitive context.

`group_id` is the sole input to the `make_m2` C++ kernel (besides M1). The kernel precondition is that IDs are dense (no gaps) and non-negative. Violating this precondition raises a `ValueError` before entering C++.

## Pseudo-correlation results

For the default `key_added="pseudo_correlation"`, the observed statistic and
event-wise inference remain aligned to the event axis:

- `var["pseudo_correlation"]`: observed signed pseudo-R².
- `var["pseudo_correlation_null_mean"]` and
  `var["pseudo_correlation_null_sd"]`: mean and sample standard deviation of
  each event's valid null draws.
- `var["pseudo_correlation_n_perm_valid"]`: valid draw count.
- `var["pseudo_correlation_emp_pvalue"]` and
  `var["pseudo_correlation_emp_padj"]`: two-sided empirical p-value and
  Benjamini-Hochberg adjustment.
- `varm["pseudo_correlation_null"]`: raw events × permutations null matrix.
- `uns["scsplice"]["pseudo_correlation"]["pseudo_correlation"]`: parameters
  and event/draw counts.

Custom `key_added` values replace the `pseudo_correlation` prefix in every
location. `get_pseudo_correlation_result()` uses these aligned fields to build
the R-compatible `statistics` and long `null_distribution` tables on demand,
so the long representation does not consume space in serialized AnnData.

## `m2_valid` sentinel

`adata.uns["scsplice"]["m2_valid"]` tracks whether M2 is consistent with the current state of M1 and `group_id`:

- Set to `False` by `read_starsolo` (M2 has not been computed yet).
- Set to `False` automatically when the var axis is subsetted (because M2 values are stale after group membership changes).
- Set to `True` only by `make_m2` after a successful computation.

Functions that require valid M2 (`highly_variable_events`, `pseudo_correlation`) check this flag and raise a `ValueError` rather than silently producing wrong results.

!!! note "Legacy `uns[\"splikit\"]` key (v1.0 compat shim)"
    AnnData objects created with `splikit-py` v1.0 carry `uns["splikit"]` instead of
    `uns["scsplice"]`. In scsplice v2.0, the `get_scsplice_ns()` and
    `setdefault_scsplice_ns()` helpers in `scsplice._core._validators` read the
    legacy key with a `FutureWarning` and migrate the value to `uns["scsplice"]`
    automatically on first access. The legacy `uns["splikit"]` key will be removed
    in v3.0. Re-save your AnnData after the first scsplice v2.0 call to persist the
    migrated key.

## Why `ljv_kind="start_end"` is the default

In typical scRNA-seq data, alternative splicing at both 5' and 3' ends is biologically meaningful. Using `ljv_kind="start_end"` (the default) captures both and matches the R splikit default. The `"start"` and `"end"` modes are provided for users who want to analyse only one class of alternative sites — for example, to reduce the event count in experiments where memory or compute is constrained.

## AnnData layout conventions

scsplice follows the standard AnnData layout:

- **Rows (`obs`) = cells.** All count matrices in `layers` are cells × events.
- **Columns (`var`) = events** (splice junctions after LJV expansion).
- No `X` matrix is set by `read_starsolo` or `make_m2`. All data lives in `layers`.

The C++ kernels internally transpose to events × cells (better memory access pattern for the column-sparse kernels), then transpose back before storing in `layers`. This is an implementation detail; callers always receive cells × events.
