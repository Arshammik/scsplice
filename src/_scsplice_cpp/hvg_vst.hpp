#pragma once

#include <Eigen/SparseCore>
#include <tuple>

namespace scsplice {

// Per-row (gene) mean and population variance (including zero entries) of a
// sparse genes x cells matrix:
//   mean[i] = sum_c(X[i,c]) / n_cols
//   var[i]  = sum_c(X[i,c]^2) / n_cols - mean[i]^2
//
// Port of R splikit src/hvf_gene_expression.cpp
// (standardizeSparse_variance_vst), first pass. np.allclose-tight vs R, not
// bit-exact: floating-point summation order differs (this walks each row's
// nonzeros in row-major order; R accumulates in a single column-major sweep).
//
// Deterministic across n_threads (disjoint per-row writes, no reductions).
std::tuple<Eigen::VectorXd, Eigen::VectorXd>
hvg_row_mean_var(const Eigen::SparseMatrix<double, Eigen::ColMajor>& X, int n_threads);

// Seurat/R-splikit VST standardization pass. For each row i:
//   z(i,c)    = clamp((X(i,c) - mean[i]) / sd[i], -vmax, vmax), vmax = sqrt(n_cols)
//   result[i] = (sum_c z(i,c)^2) / (n_cols - 1)          [0 if n_cols <= 1]
// Zero entries contribute (n_cols - nnz[i]) * clamp((0 - mean[i]) / sd[i])^2;
// nnz[i] is derived from the row's stored nonzeros, not passed in.
//
// Rows for which the caller could not fit a mean-variance trend (see the
// Python wrapper) should be passed sd[i] = 1.0 — matching R's default
// (`sd(nrow, 1.0)` in hvf_gene_expression.cpp) so those rows still get a
// (non-NaN) value rather than special-cased output. This is a faithful port
// of R's behaviour, not a bug: replicate it exactly.
//
// Port of R splikit src/hvf_gene_expression.cpp, steps 4-5 (the
// standardization pass; the loess fit itself happens in Python via
// skmisc.loess against the same underlying Fortran/C loess library R calls).
//
// Deterministic across n_threads (disjoint per-row writes, no reductions).
Eigen::VectorXd
hvg_standardize_variance(const Eigen::SparseMatrix<double, Eigen::ColMajor>& X,
                          const Eigen::Ref<const Eigen::VectorXd>& mean,
                          const Eigen::Ref<const Eigen::VectorXd>& sd,
                          int n_threads);

}  // namespace scsplice
