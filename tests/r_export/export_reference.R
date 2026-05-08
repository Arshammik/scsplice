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

# Schema normalisation: older toy bundles use `event_id`, newer R splikit uses
# `row_names_mtx`. Materialise `row_names_mtx` from whichever exists so the
# rest of the script doesn't need to branch.
if (!"row_names_mtx" %in% names(toy$eventdata)) {
    if ("event_id" %in% names(toy$eventdata)) {
        toy$eventdata$row_names_mtx <- as.character(toy$eventdata$event_id)
    } else if (!is.null(rownames(toy$m1))) {
        toy$eventdata$row_names_mtx <- rownames(toy$m1)
    } else {
        stop("toy$eventdata has no event-id column and toy$m1 has no rownames.")
    }
}

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

cat("Computing find_variable_events reference (min_row_sum=50, n_threads=1L)...\n")
hve_dt <- tryCatch(
    splikit::find_variable_events(
        m1_matrix   = m1,
        m2_matrix   = m2,
        min_row_sum = 50,
        n_threads   = 1L,
        verbose     = FALSE
    ),
    error = function(e) {
        cat("[export]   find_variable_events errored: ", conditionMessage(e), "\n")
        data.table::data.table(events = character(0), sum_deviance = numeric(0))
    }
)
cat("HVE rows kept:", nrow(hve_dt), "/", nrow(m1), "\n")

# Generate a deterministic Z draw on the R side; we dump it to the fixture so
# the Python test loads identical input (R MT vs numpy PCG64 cannot be matched).
set.seed(42L)
zdb <- matrix(stats::rnorm(n = nrow(m1) * ncol(m1), sd = 7),
              nrow = nrow(m1), ncol = ncol(m1))
rownames(zdb) <- rownames(m1)
colnames(zdb) <- colnames(m1)

cat("Computing get_pseudo_correlation references (CoxSnell + Nagelkerke)...\n")
pcor_cox_dt <- splikit::get_pseudo_correlation(
    ZDB_matrix      = zdb,
    m1_inclusion    = m1,
    m2_exclusion    = m2,
    metric          = "CoxSnell",
    suppress_warnings = TRUE,
    verbose         = FALSE
)
pcor_nag_dt <- splikit::get_pseudo_correlation(
    ZDB_matrix      = zdb,
    m1_inclusion    = m1,
    m2_exclusion    = m2,
    metric          = "Nagelkerke",
    suppress_warnings = TRUE,
    verbose         = FALSE
)

# Re-align to the full 1..nrow(m1) order, NaN-padded for events R dropped via
# its internal na.omit. Python loads this and compares NaN-mask exactly.
align_pcor <- function(dt, all_events) {
    out <- rep(NA_real_, length(all_events))
    if (nrow(dt) == 0L) return(out)
    idx <- match(dt$event, all_events)
    keep <- !is.na(idx)
    out[idx[keep]] <- dt$pseudo_correlation[keep]
    out
}
all_events <- as.character(toy$eventdata$row_names_mtx)
pcor_cox_full <- align_pcor(pcor_cox_dt, all_events)
pcor_nag_full <- align_pcor(pcor_nag_dt, all_events)
cat(sprintf("CoxSnell  NaN: %d/%d\n", sum(is.na(pcor_cox_full)), length(all_events)))
cat(sprintf("Nagelkerke NaN: %d/%d\n", sum(is.na(pcor_nag_full)), length(all_events)))

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
rhdf5::h5createGroup(out_path, "obs")
rhdf5::h5createGroup(out_path, "find_variable_events")
rhdf5::h5createGroup(out_path, "pseudo_correlation")

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

# obs (barcode + sample_id parsed via R splikit's regex `^.{16}-(.*$)`).
m1_colnames <- colnames(toy$m1)
if (is.null(m1_colnames)) m1_colnames <- paste0("bc", seq_len(ncol(toy$m1)))
sample_ids <- sub("^.{16}-(.*$)", "\\1", m1_colnames)
# Fallback: if no `-` present in colnames, treat all cells as one sample.
if (all(sample_ids == m1_colnames)) sample_ids <- rep("s1", length(m1_colnames))
rhdf5::h5write(as.character(m1_colnames), out_path, "obs/obs_names")
rhdf5::h5write(as.character(sample_ids),  out_path, "obs/sample_id")

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

# find_variable_events reference (kept events only; Python re-aligns).
rhdf5::h5write(as.character(hve_dt$events), out_path, "find_variable_events/events")
rhdf5::h5write(as.numeric(hve_dt$sum_deviance), out_path, "find_variable_events/sum_deviance")

# pseudo_correlation reference (full event order, NaN where R early-returned).
rhdf5::h5write(all_events, out_path, "pseudo_correlation/events")
rhdf5::h5write(as.numeric(pcor_cox_full), out_path, "pseudo_correlation/coxsnell")
rhdf5::h5write(as.numeric(pcor_nag_full), out_path, "pseudo_correlation/nagelkerke")
rhdf5::h5write(as.numeric(zdb), out_path, "pseudo_correlation/zdb")
rhdf5::h5write(as.integer(c(nrow(zdb), ncol(zdb))), out_path,
               "pseudo_correlation/zdb_shape")

# Root attributes for staleness detection.
fid <- rhdf5::H5Fopen(out_path)
on.exit(rhdf5::H5Fclose(fid), add = TRUE)
rhdf5::h5writeAttribute(as.character(packageVersion("splikit")), fid, "splikit_version")
rhdf5::h5writeAttribute(paste(R.version$major, R.version$minor, sep = "."), fid, "r_version")
rhdf5::h5writeAttribute(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), fid, "generated_at")
blas <- tryCatch(sessionInfo()$BLAS, error = function(e) "unknown")
if (is.null(blas)) blas <- "unknown"
rhdf5::h5writeAttribute(as.character(blas), fid, "blas_vendor")

rhdf5::h5writeAttribute(50L, fid, "min_row_sum")
rhdf5::h5writeAttribute(42L, fid, "zdb_seed")

cat(sprintf(
    "Wrote: m1=%dx%d nnz=%d, m2=%dx%d nnz=%d, n_groups=%d, hve=%d, pcor_cox_NaN=%d\n",
    nrow(m1), ncol(m1), length(m1@x),
    nrow(m2), ncol(m2), length(m2@x),
    length(unique_groups),
    nrow(hve_dt),
    sum(is.na(pcor_cox_full))
))
