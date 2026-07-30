# scsplice.tl

Tools that operate on a populated splicing AnnData. The computational functions require `layers["M1"]`; `pseudo_correlation` additionally requires valid M2 (`uns["scsplice"]["m2_valid"] == True`). `get_pseudo_correlation_result` materializes exportable tables from a completed computation without duplicating the long null table in the `.h5ad` file.

::: scsplice.tl
    options:
      show_root_heading: true
      show_root_full_path: false
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      members_order: source
      show_source: true
