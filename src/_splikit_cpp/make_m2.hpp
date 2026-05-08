#pragma once

#include <Eigen/SparseCore>

namespace splikit {

// Build the exclusion matrix M2 from inclusion counts M1 and an LJV grouping.
// For each (event i, cell j): M2[i,j] = sum(M1[k,j] for k in same group as i) - M1[i,j].
// Entries that are exactly zero are dropped (no epsilon).
//
// Bit-exact equivalence with R splikit::make_m2 is guaranteed by:
//   - column-disjoint OpenMP partitioning (no reductions, no critical sections)
//   - thread-local indexed dense workspace (not a hash map)
//   - explicit per-column std::sort by row index before write
//
// Port of /project/6007998/arsham79/splikit/src/make_m2_cpp.cpp.
Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>
make_m2(const Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>& M1,
        const Eigen::Ref<const Eigen::VectorXi>& group_ids,
        int n_threads);

}  // namespace splikit
