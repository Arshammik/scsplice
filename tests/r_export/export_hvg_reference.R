#!/usr/bin/env Rscript
# Regenerate tests/data/r_reference_hvg.h5 — the golden file that
# scsplice.pp.highly_variable_genes(method="vst")'s cross-language
# equivalence test compares against.
#
# Required R packages: splikit, Matrix, rhdf5
#
# Invocation:
#   Rscript tests/r_export/export_hvg_reference.R [output_path]
#
# Default output_path: tests/data/r_reference_hvg.h5
#
# The fixture captures R splikit's find_variable_genes(method = "vst")
# output on the toy gene-expression matrix, verbatim (same gene order, no
# filtering — the vst branch does not drop rows). The Python-side test
# (tests/test_highly_variable_genes_vs_r.py) feeds the exported matrix
# through scsplice.pp.highly_variable_genes(method="vst") and compares
# var["standardize_variance"] to this fixture.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1L && args[1] %in% c("--help", "-h")) {
    cat("Usage: Rscript export_hvg_reference.R [output_path]\n")
    cat("Default output_path: tests/data/r_reference_hvg.h5\n")
    quit(status = 0)
}

out_path <- if (length(args) >= 1L) args[1] else "tests/data/r_reference_hvg.h5"

suppressPackageStartupMessages({
    if (!requireNamespace("splikit", quietly = TRUE))
        stop("R package 'splikit' is required. Install via devtools::install_github('csglab/splikit').")
    if (!requireNamespace("Matrix", quietly = TRUE))
        stop("R package 'Matrix' is required.")
    if (!requireNamespace("rhdf5", quietly = TRUE))
        stop("R package 'rhdf5' is required (Bioconductor).")
    library(splikit)
    library(Matrix)
})

cat("Loading toy dataset...\n")
toy <- splikit::load_toy_M1_M2_object()

if (is.null(toy$gene_expression)) {
    stop("toy$gene_expression is NULL; the installed splikit toy bundle does not ",
         "carry a gene-expression matrix. Regenerate/update the toy RDS or point ",
         "this script at a different sparse gene x cell count matrix.")
}

ge <- as(toy$gene_expression, "CsparseMatrix")
if (is.null(rownames(ge))) {
    stop("toy$gene_expression has no rownames (gene ids); cannot export events.")
}

cat(sprintf("gene_expression: %d genes x %d cells, nnz=%d\n",
            nrow(ge), ncol(ge), length(ge@x)))

cat("Computing find_variable_genes(method = 'vst')...\n")
hvg_dt <- splikit::find_variable_genes(ge, method = "vst", verbose = FALSE)
# vst does not filter or reorder rows: verify the invariant this exporter and
# the Python test both rely on before writing anything.
if (!identical(as.character(hvg_dt$events), rownames(ge))) {
    stop("Internal error: find_variable_genes(method='vst') reordered or ",
         "filtered rows; the exporter's row-order assumption no longer holds.")
}

if (file.exists(out_path)) file.remove(out_path)
parent_dir <- dirname(out_path)
if (!dir.exists(parent_dir)) dir.create(parent_dir, recursive = TRUE)

cat("Writing", out_path, "...\n")
rhdf5::h5createFile(out_path)
rhdf5::h5createGroup(out_path, "gene_expression")
rhdf5::h5createGroup(out_path, "find_variable_genes")

rhdf5::h5write(as.numeric(ge@p), out_path, "gene_expression/indptr")
rhdf5::h5write(as.numeric(ge@i), out_path, "gene_expression/indices")
rhdf5::h5write(as.numeric(ge@x), out_path, "gene_expression/data")
rhdf5::h5write(as.integer(c(nrow(ge), ncol(ge))), out_path, "gene_expression/shape")
rhdf5::h5write(rownames(ge), out_path, "gene_expression/gene_names")

rhdf5::h5write(as.character(hvg_dt$events), out_path, "find_variable_genes/events")
rhdf5::h5write(as.numeric(hvg_dt$standardize_variance), out_path,
               "find_variable_genes/standardize_variance")

fid <- rhdf5::H5Fopen(out_path)
on.exit(rhdf5::H5Fclose(fid), add = TRUE)
rhdf5::h5writeAttribute(as.character(packageVersion("splikit")), fid, "splikit_version")
rhdf5::h5writeAttribute(paste(R.version$major, R.version$minor, sep = "."), fid, "r_version")
rhdf5::h5writeAttribute(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), fid, "generated_at")
blas <- tryCatch(sessionInfo()$BLAS, error = function(e) "unknown")
if (is.null(blas)) blas <- "unknown"
rhdf5::h5writeAttribute(as.character(blas), fid, "blas_vendor")
rhdf5::h5writeAttribute(0.3, fid, "loess_span")

cat(sprintf(
    "Wrote: gene_expression=%dx%d nnz=%d, standardize_variance range=[%.6g, %.6g]\n",
    nrow(ge), ncol(ge), length(ge@x),
    min(hvg_dt$standardize_variance), max(hvg_dt$standardize_variance)
))
