#!/usr/bin/env Rscript
# run_r_pipeline.R
# -----------------------------------------------------------------------
# Runs the full splikit R pipeline (make_junction_ab -> make_m1 -> make_m2)
# on two real STARsolo samples and exports the result to HDF5 for
# comparison with the Python (splikitpy) reimplementation.
#
# Invoke:  Rscript run_r_pipeline.R
# Output:  data/r_pipeline.h5
# -----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(splikit)
  library(rhdf5)
  library(Matrix)
  library(data.table)
})

# -----------------------------------------------------------------------
# 0.  Hardcoded paths
# -----------------------------------------------------------------------

SJ_DIRS <- list(
  "/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_A01_S01_L003/Solo.out/SJ",
  "/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_B01_S01_L003/Solo.out/SJ"
)
SAMPLE_IDS <- list("A01", "B01")

WL_PATHS <- lapply(SJ_DIRS, function(sj_dir) {
  file.path(sj_dir, "..", "Gene", "filtered", "barcodes.tsv")
})

OUT_H5 <- "/home/arsham79/projects/rrg-hsn/arsham79/splikitpy/validation/e2e/data/r_pipeline.h5"

# -----------------------------------------------------------------------
# 1.  Validate inputs exist
# -----------------------------------------------------------------------

for (i in seq_along(SJ_DIRS)) {
  stopifnot("SJ dir missing" = dir.exists(SJ_DIRS[[i]]))
  stopifnot("whitelist missing" = file.exists(WL_PATHS[[i]]))
  cat("[check] sample", SAMPLE_IDS[[i]], "paths OK\n")
}

# -----------------------------------------------------------------------
# 2.  Read whitelists (character vectors, one per sample)
# -----------------------------------------------------------------------

cat("[step 1] reading per-sample whitelists ...\n")
white_barcode_lists <- lapply(seq_along(WL_PATHS), function(i) {
  bc <- data.table::fread(WL_PATHS[[i]], header = FALSE, showProgress = FALSE)$V1
  cat("         sample", SAMPLE_IDS[[i]], ":", length(bc), "whitelisted barcodes\n")
  bc
})

# -----------------------------------------------------------------------
# 3.  make_junction_ab
# -----------------------------------------------------------------------

cat("[step 2] make_junction_ab ...\n")
junction_ab_object <- make_junction_ab(
  STARsolo_SJ_dirs     = SJ_DIRS,
  white_barcode_lists  = white_barcode_lists,
  sample_ids           = SAMPLE_IDS,
  use_internal_whitelist = FALSE,   # we supply external lists
  keep_multi_mapped_junctions = FALSE,
  verbose              = TRUE
)
cat("         done:", length(junction_ab_object), "samples in junction_ab_object\n")

# -----------------------------------------------------------------------
# 4.  make_m1
# -----------------------------------------------------------------------

cat("[step 3] make_m1 ...\n")
m1_obj <- make_m1(
  junction_ab_object = junction_ab_object,
  min_counts         = 1,
  verbose            = TRUE
)

m1   <- m1_obj$m1_inclusion_matrix   # dgCMatrix events x cells
evdt <- m1_obj$event_data             # data.table

cat("         M1 shape:", nrow(m1), "events x", ncol(m1), "cells\n")
stopifnot("M1 row count matches eventdata" = nrow(m1) == nrow(evdt))
stopifnot("M1 rownames match eventdata row_names_mtx" =
            identical(rownames(m1), evdt$row_names_mtx))

# -----------------------------------------------------------------------
# 5.  make_m2
# -----------------------------------------------------------------------

cat("[step 4] make_m2 (C++ path, n_threads=1) ...\n")
m2 <- make_m2(
  m1_inclusion_matrix = m1,
  eventdata           = evdt,
  n_threads           = 1L,
  use_cpp             = TRUE,
  verbose             = TRUE
)
cat("         M2 shape:", nrow(m2), "events x", ncol(m2), "cells\n")
stopifnot("M2 dims match M1" = identical(dim(m2), dim(m1)))

# -----------------------------------------------------------------------
# 6.  Derived columns for export
# -----------------------------------------------------------------------

# Dense 0-based group_id — mirrors what make_m2's C++ kernel receives.
# Replicates the remapping in make_m2() lines 655-657 of star_solo_processing.R:
#   unique_groups <- unique(eventdata$group_id)
#   group_map     <- setNames(seq_along(unique_groups) - 1L, unique_groups)
unique_groups   <- unique(evdt$group_id)
group_map       <- setNames(seq_along(unique_groups) - 1L, as.character(unique_groups))
group_id_dense  <- as.integer(group_map[as.character(evdt$group_id)])

# group_kind: trailing letter of the group_id string ("S" or "E")
group_kind <- sub("^.*_([SE])$", "\\1", evdt$group_id)

# obs vectors — colnames are "<16-char barcode>-<sample_id>"
obs_names <- colnames(m1)
barcode   <- substr(obs_names, 1, 16)
sample_id <- sub("^.{16}-", "", obs_names)

cat("[check] obs_names sample:", head(obs_names, 3), "\n")
cat("[check] barcodes  sample:", head(barcode,   3), "\n")
cat("[check] sample_id sample:", head(sample_id, 3), "\n")

# -----------------------------------------------------------------------
# 7.  Export to HDF5
# -----------------------------------------------------------------------

cat("[step 5] writing", OUT_H5, "...\n")

if (file.exists(OUT_H5)) file.remove(OUT_H5)
rhdf5::h5createFile(OUT_H5)

# Helper: cast sparse matrix slots to numeric before writing to avoid rhdf5
# silently mangling integer types.
write_csc <- function(h5f, group, mat) {
  dgc <- as(mat, "dgCMatrix")
  rhdf5::h5createGroup(h5f, group)
  rhdf5::h5write(as.numeric(dgc@p),    h5f, paste0(group, "/indptr"))
  rhdf5::h5write(as.numeric(dgc@i),    h5f, paste0(group, "/indices"))
  rhdf5::h5write(as.numeric(dgc@x),    h5f, paste0(group, "/data"))
  rhdf5::h5write(as.numeric(dim(dgc)), h5f, paste0(group, "/shape"))
}

# --- /m1, /m2 ---
write_csc(OUT_H5, "m1", m1)
write_csc(OUT_H5, "m2", m2)
cat("         /m1 and /m2 written\n")

# --- /eventdata ---
rhdf5::h5createGroup(OUT_H5, "eventdata")
rhdf5::h5write(evdt$row_names_mtx,   OUT_H5, "eventdata/row_names_mtx")
rhdf5::h5write(group_id_dense,       OUT_H5, "eventdata/group_id")
rhdf5::h5write(group_kind,           OUT_H5, "eventdata/group_kind")
rhdf5::h5write(as.integer(evdt$group_count), OUT_H5, "eventdata/group_count")

# Genomic coordinate columns — emit zero-length placeholders if absent
for (col in c("chr", "start", "end", "strand")) {
  if (col %in% colnames(evdt)) {
    rhdf5::h5write(evdt[[col]], OUT_H5, paste0("eventdata/", col))
  } else {
    warning("eventdata column '", col, "' missing — writing empty placeholder")
    rhdf5::h5write(character(0), OUT_H5, paste0("eventdata/", col))
  }
}
cat("         /eventdata written\n")

# --- /obs ---
rhdf5::h5createGroup(OUT_H5, "obs")
rhdf5::h5write(obs_names, OUT_H5, "obs/obs_names")
rhdf5::h5write(sample_id, OUT_H5, "obs/sample_id")
rhdf5::h5write(barcode,   OUT_H5, "obs/barcode")
cat("         /obs written\n")

# --- root attributes (open handle once, write all, then close) ---
h5fid <- rhdf5::H5Fopen(OUT_H5)
rhdf5::h5writeAttribute(as.character(packageVersion("splikit")),
                        h5fid, "splikit_version")
rhdf5::h5writeAttribute(R.version$version.string, h5fid, "r_version")
rhdf5::h5writeAttribute(format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
                        h5fid, "generated_at")
rhdf5::h5writeAttribute(
  tryCatch(as.character(extSoftVersion()["BLAS"]), error = function(e) "unknown"),
  h5fid, "blas_vendor"
)
rhdf5::h5writeAttribute(as.integer(length(SAMPLE_IDS)), h5fid, "n_samples")
rhdf5::h5writeAttribute(paste(unlist(SAMPLE_IDS), collapse = ","), h5fid, "samples")
rhdf5::H5Fclose(h5fid)

rhdf5::H5close()
cat("[done]  HDF5 written to", OUT_H5, "\n")
cat("        M1:", nrow(m1), "events x", ncol(m1), "cells  |",
    "M2:", nrow(m2), "events x", ncol(m2), "cells\n")
