#!/usr/bin/env Rscript
# Regenerate tests/data/r_reference.h5 — the golden file that splikit-py's
# numerical-equivalence tests compare against.
#
# Required R packages: splikit, Matrix, rhdf5
#
# Invocation:
#   Rscript tests/r_export/export_reference.R [output_path]
#
# Default output_path: tests/data/r_reference.h5
#
# Gated behind pytest.mark.r_required in CI; one dedicated CI job regenerates
# the fixture against the latest CRAN/GitHub R splikit and uploads it as an
# artifact for the rest of the matrix.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1L && args[1] %in% c("--help", "-h")) {
    cat("Usage: Rscript export_reference.R [output_path]\n")
    cat("Default output_path: tests/data/r_reference.h5\n")
    quit(status = 0)
}

out_path <- if (length(args) >= 1L) args[1] else "tests/data/r_reference.h5"

# Loud failures on missing dependencies.
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

# Mirror the C++ wrapper's group_id remap (splikit/R/star_solo_processing.R:655-657).
# The dense 0..G-1 vector is what the C++ kernel actually receives; that is what
# we export so the Python test sees the identical input.
unique_groups <- unique(toy$eventdata$group_id)
group_map <- setNames(seq_along(unique_groups) - 1L, as.character(unique_groups))
group_ids_i32 <- as.integer(group_map[as.character(toy$eventdata$group_id)])

cat("Recomputing M2 via splikit::make_m2(use_cpp=TRUE, n_threads=1L)...\n")
m1 <- toy$m1
m2 <- splikit::make_m2(m1, toy$eventdata, use_cpp = TRUE, n_threads = 1L)

# Coerce to dgCMatrix to guarantee the @p / @i / @x slot semantics.
m1 <- as(m1, "CsparseMatrix")
m2 <- as(m2, "CsparseMatrix")

# Remove any stale file before writing.
if (file.exists(out_path)) file.remove(out_path)

# Ensure the parent directory exists.
parent_dir <- dirname(out_path)
if (!dir.exists(parent_dir)) dir.create(parent_dir, recursive = TRUE)

cat("Writing", out_path, "...\n")
rhdf5::h5createFile(out_path)
rhdf5::h5createGroup(out_path, "m1")
rhdf5::h5createGroup(out_path, "m2")
rhdf5::h5createGroup(out_path, "eventdata")

write_csc <- function(file, group, mat) {
    rhdf5::h5write(as.numeric(mat@p), file, paste0(group, "/indptr"))
    rhdf5::h5write(as.numeric(mat@i), file, paste0(group, "/indices"))
    rhdf5::h5write(as.numeric(mat@x), file, paste0(group, "/data"))
    rhdf5::h5write(as.integer(c(nrow(mat), ncol(mat))), file, paste0(group, "/shape"))
}

write_csc(out_path, "m1", m1)
write_csc(out_path, "m2", m2)

rhdf5::h5write(group_ids_i32, out_path, "eventdata/group_id")
rhdf5::h5write(as.character(toy$eventdata$row_names_mtx), out_path, "eventdata/row_names_mtx")

# group_kind / group_count are nice-to-have but the toy may not carry them
# under exactly those names; tolerate absence.
if ("group_kind" %in% names(toy$eventdata)) {
    rhdf5::h5write(as.character(toy$eventdata$group_kind), out_path, "eventdata/group_kind")
} else {
    warning("toy$eventdata$group_kind not present; emitting 'S' for every event")
    rhdf5::h5write(rep("S", nrow(toy$eventdata)), out_path, "eventdata/group_kind")
}
if ("group_count" %in% names(toy$eventdata)) {
    rhdf5::h5write(as.integer(toy$eventdata$group_count), out_path, "eventdata/group_count")
} else {
    counts <- as.integer(table(toy$eventdata$group_id)[as.character(toy$eventdata$group_id)])
    rhdf5::h5write(counts, out_path, "eventdata/group_count")
}

# Optional coordinate / strand columns; silent skip if missing.
for (col in c("chr", "start", "end", "strand")) {
    if (col %in% names(toy$eventdata)) {
        v <- toy$eventdata[[col]]
        if (is.numeric(v)) {
            rhdf5::h5write(as.numeric(v), out_path, paste0("eventdata/", col))
        } else {
            rhdf5::h5write(as.character(v), out_path, paste0("eventdata/", col))
        }
    } else {
        warning(sprintf("toy$eventdata$%s not present; skipping", col))
    }
}

# Root attributes for staleness detection.
fid <- rhdf5::H5Fopen(out_path)
on.exit(rhdf5::H5Fclose(fid), add = TRUE)
rhdf5::h5writeAttribute(as.character(packageVersion("splikit")), fid, "splikit_version")
rhdf5::h5writeAttribute(paste(R.version$major, R.version$minor, sep = "."), fid, "r_version")
rhdf5::h5writeAttribute(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), fid, "generated_at")
blas <- tryCatch(sessionInfo()$BLAS, error = function(e) "unknown")
if (is.null(blas)) blas <- "unknown"
rhdf5::h5writeAttribute(as.character(blas), fid, "blas_vendor")

cat(sprintf(
    "Wrote: m1=%dx%d nnz=%d, m2=%dx%d nnz=%d, n_groups=%d\n",
    nrow(m1), ncol(m1), length(m1@x),
    nrow(m2), ncol(m2), length(m2@x),
    length(unique_groups)
))
