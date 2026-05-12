#pragma once

#include <Eigen/SparseCore>

namespace scsplice {

// Per-event ratio binomial deviance for one library.
//
// For each row i (event):
//   p_hat = sum_k(M1[i,k]) / sum_k(M1[i,k] + M2[i,k])
//   dev[i] = sum_k 2 * (M1[i,k]*log(M1[i,k] / (n_k*p_hat))
//                        + M2[i,k]*log(M2[i,k] / (n_k*(1-p_hat))))
// where n_k = M1[i,k] + M2[i,k] and the M*log(M) term is skipped (= 0)
// whenever the corresponding count is zero. Rows with sum_n == 0 or
// p_hat == 0 / 1 return 0.
//
// Bit-identical across thread counts (per-row results, disjoint scalar writes).
// Equivalence with R splikit::calcDeviances_ratio is np.allclose-tight, not
// bit-exact, because std::log differs across libm/BLAS stacks.
//
// Port of R splikit src/calcDeviances.cpp.
//
// Default `int` StorageIndex matches scipy.sparse.csc_matrix on the
// pybind11 boundary.
Eigen::VectorXd
calc_deviances_ratio(const Eigen::SparseMatrix<double, Eigen::ColMajor>& M1,
                     const Eigen::SparseMatrix<double, Eigen::ColMajor>& M2,
                     int n_threads);

}  // namespace scsplice
