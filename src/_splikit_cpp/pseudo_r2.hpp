#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>

namespace splikit {

// Per-event signed pseudo-correlation between an external Z predictor and the
// M1/(M1+M2) ratio.
//
// For each event i, fit a binomial GLM (logistic link) via IRLS with design
// matrix [intercept | Z[i, valid]], where 'valid' selects cells where
// (M1+M2) > 0. Return signed_r = sqrt(R^2) * sign(slope), where R^2 is
// CoxSnell or Nagelkerke pseudo-R^2 from the deviance ratio. NaN is written
// for events with n_valid<2, all-zero M1 or M2, singular Hessian, or R^2<0.
//
// Z orientation contract: Z is **events x cells** (n_events rows, n_cells
// columns); the kernel indexes Z(i, c) for event i, cell c. The Python
// wrapper enforces this on its side (see splikit.tl.pseudo_correlation
// shape check) and the binding docstring documents the contract.
//
// Cross-language tolerance vs R splikit::cppBetabinPseudoR2:
//   - converged events: np.allclose(rtol=1e-9, atol=1e-7)
//   - events near tol=1e-6 IRLS convergence may diverge by 1 iter — same band
//   - bit-identical across thread counts (per-event disjoint scalar writes)
//
// Port of R splikit src/cpp_pseudoR2.cpp; the four R dispatch wrappers are
// collapsed into this single sparse x sparse path because AnnData.layers are
// always sparse in our schema.
//
// Default `int` StorageIndex matches scipy.sparse.csc_matrix on the
// pybind11 boundary.
Eigen::VectorXd
pseudo_correlation(const Eigen::Ref<const Eigen::MatrixXd>& Z,
                   const Eigen::SparseMatrix<double, Eigen::ColMajor>& M1,
                   const Eigen::SparseMatrix<double, Eigen::ColMajor>& M2,
                   const std::string& metric,
                   int n_threads);

}  // namespace splikit
