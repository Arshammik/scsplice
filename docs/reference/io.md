# scsplice.io

Input / output functions for STARsolo output. Three readers, one consistent API shape:

| Function | STARsolo source | Output |
|---|---|---|
| `read_starsolo` | `Solo.out/SJ/` | Splicing AnnData (`layers["M1"]`, `layers["M2"]`) |
| `read_starsolo_gene` | `Solo.out/Gene/` | Gene-expression AnnData (`X` = raw counts) |
| `read_starsolo_velocyto` | `Solo.out/Velocyto/` | Velocity AnnData (`layers["spliced/unspliced/ambiguous"]`) |

All three accept `tissue_positions=` for Visium / spatial samples and populate squidpy-compatible `obsm["spatial"]` and `uns["spatial"]`.

See [STARsolo readers and AnnData data layouts](../explanation/io-readers-and-data-layouts.md) for design rationale and the full AnnData schema.

---

::: scsplice.io
    options:
      show_root_heading: true
      show_root_full_path: false
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      members_order: source
      show_source: true
