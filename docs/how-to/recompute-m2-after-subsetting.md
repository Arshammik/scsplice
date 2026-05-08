# Recompute M2 after subsetting

## Problem

You have a fully processed splicing AnnData with valid M2 layers, and you want to subset it — for example, to keep only a specific cell type or to drop low-coverage events:

```python
# Subset to a single cell type
adata_t = adata[adata.obs["cell_type"] == "T cell", :].copy()

# Or subset to highly variable events only
adata_hve = adata[:, adata.var["highly_variable"]].copy()
```

After subsetting the **var axis** (events), M2 is no longer valid. The old M2 values were computed with the full LJV group membership; removing some members from the group changes the correct M2 values for the remaining members.

```python
print(adata_hve.uns["splikit"]["m2_valid"])  # False — invalidated automatically
```

## Solution

Call `make_m2` on the subset:

```python
import splikit as splk

# Subset to HVE
adata_hve = adata[:, adata.var["highly_variable"]].copy()

# M2 was invalidated; recompute from the subset's group structure
splk.tl.make_m2(adata_hve, n_threads=4)

print(adata_hve.uns["splikit"]["m2_valid"])  # True
```

M2 is now computed from the subset's `var["group_id"]`, which reflects only the events still present.

## Why subsetting cells does not require recomputation

Subsetting the **obs axis** (cells) does not change the LJV group structure — `group_id` is a property of the `var` axis, not the `obs` axis. M2 for the remaining cells is still the correct per-cell sum over each LJV group. The `m2_valid` flag remains `True` after a cell-only subset.

```python
# Cell subsetting: M2 remains valid
adata_cells = adata[adata.obs["sample_id"] == "s1", :].copy()
print(adata_cells.uns["splikit"]["m2_valid"])  # True — no recomputation needed
```

## How `m2_valid` is managed

The `m2_valid` flag in `adata.uns["splikit"]` is set to `False` by `read_starsolo` (M2 has not been computed yet) and by internal helpers whenever the var axis changes shape. It is set to `True` only by `make_m2`. Functions that require valid M2 — `highly_variable_events` and `pseudo_correlation` — check this flag and raise a `ValueError` if it is `False`.

!!! warning "Do not set `m2_valid` manually"
    Setting `adata.uns["splikit"]["m2_valid"] = True` without recomputing M2 will cause incorrect results downstream. Always call `make_m2` to revalidate.
