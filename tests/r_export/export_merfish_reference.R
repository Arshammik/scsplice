#!/usr/bin/env Rscript
# Generate the real-data R reference fixture for splikit-py validation.
#
# Loads the saved MultiGEDI model, slices to one sample (smallest one,
# SRR26528546 by default), recomputes M2 / find_variable_events /
# get_pseudo_correlation on the subset, extracts ZDB from the model, and
# writes everything to HDF5.
#
# Usage on c170:
#   module load r/4.4.0
#   Rscript tests/r_export/export_merfish_reference.R \
#       --model /home/arsham79/projects/rrg-hsn/arsham79/model_merfish_apr_30.rds \
#       --m1    /home/arsham79/scratch/new-midbrain/m1_with_CB.rds \
#       --eventdata /home/arsham79/scratch/new-midbrain/event_data_with_CB.rds \
#       --sample SRR26528546 \
#       --out   tests/data/r_reference_merfish.h5
#
# All args are optional; defaults match the canonical paths.

suppressPackageStartupMessages({
    library(Matrix)
    library(data.table)
    library(splikit)
    library(rhdf5)
    library(R6)  # for MultiGEDI introspection
})

# --- argument parsing ---------------------------------------------------
parse_args <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    out <- list(
        model     = "/home/arsham79/projects/rrg-hsn/arsham79/model_merfish_apr_30.rds",
        m1        = "/home/arsham79/scratch/new-midbrain/m1_with_CB.rds",
        eventdata = "/home/arsham79/scratch/new-midbrain/event_data_with_CB.rds",
        sample    = "SRR26528546",
        out       = "tests/data/r_reference_merfish.h5",
        min_row_sum = 1
    )
    i <- 1L
    while (i <= length(args)) {
        a <- args[i]
        v <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
        if (a == "--model") out$model <- v
        else if (a == "--m1") out$m1 <- v
        else if (a == "--eventdata") out$eventdata <- v
        else if (a == "--sample") out$sample <- v
        else if (a == "--out") out$out <- v
        else if (a == "--min-row-sum") out$min_row_sum <- as.integer(v)
        else if (a %in% c("--help", "-h")) {
            cat("Usage: Rscript export_merfish_reference.R [--model PATH] [--m1 PATH]\n",
                "  [--eventdata PATH] [--sample ID] [--out PATH] [--min-row-sum N]\n")
            quit(status = 0L)
        }
        else stop("Unknown argument: ", a)
        i <- i + 2L
    }
    out
}

opt <- parse_args()
cat("[export] model     :", opt$model, "\n")
cat("[export] m1        :", opt$m1, "\n")
cat("[export] eventdata :", opt$eventdata, "\n")
cat("[export] sample    :", opt$sample, "\n")
cat("[export] out       :", opt$out, "\n")
cat("[export] min_row_sum:", opt$min_row_sum, "\n\n")

# --- load model ---------------------------------------------------------
cat("[export] Loading model RDS...\n")
model <- readRDS(opt$model)
stopifnot(inherits(model, "R6"))

# Resolve the sample's events and cells.
cd <- model$colData("Splicing")
events <- cd$geneIDs
all_samples <- model$samples
sample_idx <- which(all_samples == opt$sample)
if (length(sample_idx) != 1L)
    stop("Sample ", opt$sample, " not in model$samples (",
         paste(head(all_samples, 5), collapse=", "), "...)")
cells <- cd$cellIDs_per_sample[[opt$sample]]
cat("[export] Events:", length(events),
    "| Cells in sample:", length(cells), "\n")

# --- ZDB ---------------------------------------------------------------
cat("[export] Extracting ZDB...\n")
zdb <- model$projections$ZDB(modalities = "Splicing",
                              samples    = opt$sample)
storage.mode(zdb) <- "double"
stopifnot(identical(rownames(zdb), events))
stopifnot(identical(colnames(zdb), cells))
cat("[export] ZDB:", nrow(zdb), "x", ncol(zdb),
    "| size:", format(object.size(zdb), units = "auto"), "\n")

# --- slice M1 ----------------------------------------------------------
cat("[export] Loading full M1 RDS (large)...\n")
m1_full <- readRDS(opt$m1)
m1_subset <- m1_full[events, cells]
m1_subset <- as(m1_subset, "CsparseMatrix")
rm(m1_full); invisible(gc())
cat("[export] M1 subset:", nrow(m1_subset), "x", ncol(m1_subset),
    "| nnz:", length(m1_subset@x), "\n")

# --- slice eventdata ---------------------------------------------------
cat("[export] Loading eventdata RDS...\n")
ev_full <- readRDS(opt$eventdata)
if (!data.table::is.data.table(ev_full)) ev_full <- data.table::as.data.table(ev_full)
id_col <- if ("row_names_mtx" %in% names(ev_full)) "row_names_mtx" else
          stop("Could not find event-id column in eventdata.")
match_idx <- match(events, ev_full[[id_col]])
if (any(is.na(match_idx)))
    stop(sum(is.na(match_idx)), " events not found in eventdata; cannot continue.")
ev_subset <- ev_full[match_idx, ]
stopifnot(identical(ev_subset[[id_col]], events))
cat("[export] Eventdata subset cols:", paste(names(ev_subset), collapse = ", "), "\n")

# Mirror the make_m2 wrapper's group_id remap to dense 0..G-1 int32.
unique_groups <- unique(ev_subset$group_id)
group_map <- setNames(seq_along(unique_groups) - 1L, as.character(unique_groups))
group_id_dense <- as.integer(group_map[as.character(ev_subset$group_id)])
n_groups <- length(unique_groups)
cat("[export] LJV groups (post-subset):", n_groups, "\n")

# --- recompute M2 on the subset ---------------------------------------
cat("[export] make_m2 (use_cpp=TRUE, n_threads=1)...\n")
m2_subset <- splikit::make_m2(
    m1_inclusion_matrix = m1_subset,
    eventdata           = ev_subset,
    n_threads           = 1L,
    use_cpp             = TRUE,
    verbose             = FALSE
)
m2_subset <- as(m2_subset, "CsparseMatrix")
cat("[export] M2 subset:", nrow(m2_subset), "x", ncol(m2_subset),
    "| nnz:", length(m2_subset@x), "\n")

# --- find_variable_events on the subset --------------------------------
cat("[export] find_variable_events (min_row_sum=", opt$min_row_sum,
    ", n_threads=1)...\n", sep = "")
hve_dt <- tryCatch(
    splikit::find_variable_events(
        m1_matrix   = m1_subset,
        m2_matrix   = m2_subset,
        min_row_sum = opt$min_row_sum,
        n_threads   = 1L,
        verbose     = FALSE
    ),
    error = function(e) {
        cat("[export]   find_variable_events errored: ", conditionMessage(e), "\n")
        cat("[export]   continuing with empty HVE table\n")
        data.table::data.table(events = character(0), sum_deviance = numeric(0))
    }
)
cat("[export] HVE rows kept:", nrow(hve_dt), "\n")

# --- get_pseudo_correlation on the subset ------------------------------
# Returns events that didn't trip an na.omit; we manually re-align to the
# 12774-event order, filling NaN for dropped events.
cat("[export] get_pseudo_correlation (CoxSnell)...\n")
pcor_cox_dt <- splikit::get_pseudo_correlation(
    ZDB_matrix      = zdb,
    m1_inclusion    = m1_subset,
    m2_exclusion    = m2_subset,
    metric          = "CoxSnell",
    suppress_warnings = TRUE,
    verbose         = FALSE
)
cat("[export] get_pseudo_correlation (Nagelkerke)...\n")
pcor_nag_dt <- splikit::get_pseudo_correlation(
    ZDB_matrix      = zdb,
    m1_inclusion    = m1_subset,
    m2_exclusion    = m2_subset,
    metric          = "Nagelkerke",
    suppress_warnings = TRUE,
    verbose         = FALSE
)

align_pcor <- function(dt, all_events) {
    out <- rep(NA_real_, length(all_events))
    if (nrow(dt) == 0L) return(out)
    idx <- match(dt$event, all_events)
    keep <- !is.na(idx)
    out[idx[keep]] <- dt$pseudo_correlation[keep]
    out
}
pcor_cox_full <- align_pcor(pcor_cox_dt, events)
pcor_nag_full <- align_pcor(pcor_nag_dt, events)
cat("[export] CoxSnell    NaN count:", sum(is.na(pcor_cox_full)), "/", length(events), "\n")
cat("[export] Nagelkerke  NaN count:", sum(is.na(pcor_nag_full)), "/", length(events), "\n")

# --- write HDF5 --------------------------------------------------------
out_path <- opt$out
if (file.exists(out_path)) file.remove(out_path)
parent <- dirname(out_path)
if (!dir.exists(parent)) dir.create(parent, recursive = TRUE)
cat("[export] Writing HDF5 to", out_path, "...\n")

rhdf5::h5createFile(out_path)
rhdf5::h5createGroup(out_path, "m1")
rhdf5::h5createGroup(out_path, "m2")
rhdf5::h5createGroup(out_path, "eventdata")
rhdf5::h5createGroup(out_path, "find_variable_events")
rhdf5::h5createGroup(out_path, "pseudo_correlation")

write_csc <- function(group, mat) {
    rhdf5::h5write(as.numeric(mat@p), out_path, paste0(group, "/indptr"))
    rhdf5::h5write(as.numeric(mat@i), out_path, paste0(group, "/indices"))
    rhdf5::h5write(as.numeric(mat@x), out_path, paste0(group, "/data"))
    rhdf5::h5write(as.integer(c(nrow(mat), ncol(mat))), out_path,
                   paste0(group, "/shape"))
}
write_csc("m1", m1_subset)
write_csc("m2", m2_subset)

# eventdata
rhdf5::h5write(as.character(ev_subset$row_names_mtx), out_path,
               "eventdata/row_names_mtx")
rhdf5::h5write(group_id_dense, out_path, "eventdata/group_id_dense")
# group_kind / group_count
extract_kind <- function(rn) {
    s <- substr(rn, nchar(rn), nchar(rn))
    if (any(!s %in% c("S", "E"))) {
        bad <- head(rn[!s %in% c("S", "E")], 5)
        warning("Non-S/E suffix encountered: ", paste(bad, collapse = ", "))
    }
    s
}
group_kind <- extract_kind(ev_subset$row_names_mtx)
group_count <- if ("group_count" %in% names(ev_subset))
                   as.integer(ev_subset$group_count) else
                   as.integer(table(group_id_dense)[as.character(group_id_dense)])
rhdf5::h5write(group_kind, out_path, "eventdata/group_kind")
rhdf5::h5write(group_count, out_path, "eventdata/group_count")
rhdf5::h5write(as.character(ev_subset$chr), out_path, "eventdata/chr")
rhdf5::h5write(as.integer(ev_subset$start), out_path, "eventdata/start")
rhdf5::h5write(as.integer(ev_subset$end), out_path, "eventdata/end")
rhdf5::h5write(as.character(ev_subset$strand), out_path, "eventdata/strand")

# ZDB
rhdf5::h5write(as.numeric(zdb), out_path, "zdb")
rhdf5::h5write(rownames(zdb), out_path, "zdb_rownames")
rhdf5::h5write(colnames(zdb), out_path, "zdb_colnames")
# Persist ZDB shape so the Python loader doesn't have to infer it from
# rownames / colnames separately (rhdf5 flattens dense matrices to 1D).
rhdf5::h5write(as.integer(c(nrow(zdb), ncol(zdb))), out_path, "zdb_shape")

# find_variable_events
rhdf5::h5write(as.character(hve_dt$events), out_path, "find_variable_events/events")
rhdf5::h5write(as.numeric(hve_dt$sum_deviance), out_path, "find_variable_events/sum_deviance")

# pseudo_correlation (one float64 vector per metric, length = length(events))
rhdf5::h5write(as.character(events), out_path, "pseudo_correlation/events")
rhdf5::h5write(pcor_cox_full, out_path, "pseudo_correlation/coxsnell")
rhdf5::h5write(pcor_nag_full, out_path, "pseudo_correlation/nagelkerke")

# Root attributes via H5Fopen / h5writeAttribute.
fid <- rhdf5::H5Fopen(out_path)
on.exit(rhdf5::H5Fclose(fid), add = TRUE)
rhdf5::h5writeAttribute(opt$sample, fid, "sample_id")
rhdf5::h5writeAttribute(as.character(packageVersion("splikit")), fid, "splikit_version")
rhdf5::h5writeAttribute(paste(R.version$major, R.version$minor, sep = "."), fid, "r_version")
rhdf5::h5writeAttribute(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), fid, "generated_at")
blas <- tryCatch(sessionInfo()$BLAS, error = function(e) "unknown")
if (is.null(blas)) blas <- "unknown"
rhdf5::h5writeAttribute(as.character(blas), fid, "blas_vendor")
rhdf5::h5writeAttribute(as.integer(opt$min_row_sum), fid, "min_row_sum")

cat("[export] DONE. ",
    "M1=", nrow(m1_subset), "x", ncol(m1_subset), " (nnz=", length(m1_subset@x), "), ",
    "M2=", nrow(m2_subset), "x", ncol(m2_subset), " (nnz=", length(m2_subset@x), "), ",
    "ZDB=", nrow(zdb), "x", ncol(zdb), ", ",
    "HVE_kept=", nrow(hve_dt), "\n", sep = "")
