#!/usr/bin/env Rscript
# Regenerate tests/data/r_reference.h5 — the golden file that splikit-py's
# numerical-equivalence tests compare against.
#
# Inputs: the toy dataset shipped with R splikit (load_toy_M1_M2_object).
# Outputs (HDF5):
#   /m1                               sparse_csc { indptr, indices, data, shape }
#   /m2                               sparse_csc { indptr, indices, data, shape }
#   /eventdata                        table { row_names_mtx, group_id, ... }
#   /find_variable_events/sum_deviance  numeric vector
#   /pseudo_correlation/zdb           dense matrix (Z draw, fixed seed)
#   /pseudo_correlation/coxsnell      numeric vector (point estimate)
#   /pseudo_correlation/null_coxsnell numeric vector (null distribution)
#   attributes on root: splikit_version, r_version, generated_at, blas_vendor
#
# Run: Rscript tests/r_export/export_reference.R [output_path]
#
# This script is gated behind pytest.mark.r_required in the Python suite; CI
# regenerates the fixture in a dedicated job and uploads it as an artifact
# for the rest of the matrix.

stop("export_reference.R is a stub for v1.0 — implementation lands with the kernels.")
