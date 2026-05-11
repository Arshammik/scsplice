#include "deviance.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef SPLIKIT_USE_OPENMP
#include <omp.h>
#endif

namespace splikit {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace {

// Compute deviance for a single row, given the row-i sparse views of M1 and M2.
//
// Mathematically: zero (M1[i,k], M2[i,k]) cells contribute zero to both
// sum_y / sum_n (y + f = 0 -> skipped in the inner branches) so we only
// need to enumerate the union of (M1, M2) nonzero columns in row i.
// We do this with a synchronised walk over the two row-i InnerIterators
// (both row-major-CSC -> sorted column indices).
inline double compute_row_deviance(const SpMatRM& M1_rm,
                                   const SpMatRM& M2_rm,
                                   Eigen::Index i) {
    double sum_y = 0.0;
    double sum_n = 0.0;
    // Pass 1: aggregate sums via union-walk.
    SpMatRM::InnerIterator it1(M1_rm, i);
    SpMatRM::InnerIterator it2(M2_rm, i);
    while (it1 || it2) {
        const Eigen::Index c1 = it1 ? it1.col() : std::numeric_limits<Eigen::Index>::max();
        const Eigen::Index c2 = it2 ? it2.col() : std::numeric_limits<Eigen::Index>::max();
        double y = 0.0;
        double f = 0.0;
        if (c1 == c2) {
            y = it1.value();
            f = it2.value();
            ++it1; ++it2;
        } else if (c1 < c2) {
            y = it1.value();
            ++it1;
        } else {
            f = it2.value();
            ++it2;
        }
        sum_y += y;
        sum_n += (y + f);
    }
    if (sum_n <= 0.0) return 0.0;
    const double p_hat = sum_y / sum_n;
    // Hard clamp matches R splikit/src/calcDeviances.cpp:38 (no epsilon).
    if (p_hat <= 0.0 || p_hat >= 1.0) return 0.0;

    // Pass 2: accumulate deviance via the same union-walk.
    double dev_row = 0.0;
    SpMatRM::InnerIterator jt1(M1_rm, i);
    SpMatRM::InnerIterator jt2(M2_rm, i);
    while (jt1 || jt2) {
        const Eigen::Index c1 = jt1 ? jt1.col() : std::numeric_limits<Eigen::Index>::max();
        const Eigen::Index c2 = jt2 ? jt2.col() : std::numeric_limits<Eigen::Index>::max();
        double y = 0.0;
        double f = 0.0;
        if (c1 == c2) {
            y = jt1.value();
            f = jt2.value();
            ++jt1; ++jt2;
        } else if (c1 < c2) {
            y = jt1.value();
            ++jt1;
        } else {
            f = jt2.value();
            ++jt2;
        }
        const double n_i = y + f;
        if (n_i <= 0.0) continue;
        // Skip M*log(M) when the count is zero (M2 written as n_i - y to
        // match the source verbatim).
        if (y > 0.0) {
            dev_row += 2.0 * y * std::log(y / (n_i * p_hat));
        }
        if (n_i - y > 0.0) {
            dev_row += 2.0 * (n_i - y) * std::log((n_i - y) / (n_i * (1.0 - p_hat)));
        }
    }
    return dev_row;
}

}  // namespace

Eigen::VectorXd
calc_deviances_ratio(const SpMat& M1, const SpMat& M2, int n_threads) {
    if (M1.rows() != M2.rows() || M1.cols() != M2.cols()) {
        throw std::invalid_argument(
            "calc_deviances_ratio: M1 and M2 must have identical shapes");
    }

    const Eigen::Index n_rows = M1.rows();
    const Eigen::Index n_cols = M1.cols();
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(n_rows);

    if (n_rows == 0 || n_cols == 0) {
        return dev;
    }

    // Transpose to row-major once up front. This converts O(n_events*n_cells)
    // coeff() lookups into a single linear pass per row via InnerIterator.
    // The input SparseMatrix must be compressed for the transpose copy; the
    // Python wrapper already passes compressed CSC so we only materialise a
    // new compressed copy when needed.
    SpMatRM M1_rm;
    SpMatRM M2_rm;
    if (M1.isCompressed()) {
        M1_rm = M1;
    } else {
        SpMat tmp = M1;
        tmp.makeCompressed();
        M1_rm = tmp;
    }
    if (M2.isCompressed()) {
        M2_rm = M2;
    } else {
        SpMat tmp = M2;
        tmp.makeCompressed();
        M2_rm = tmp;
    }
    M1_rm.makeCompressed();
    M2_rm.makeCompressed();

#ifdef SPLIKIT_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            dev[i] = compute_row_deviance(M1_rm, M2_rm, i);
        }
    } else
#endif
    {
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            dev[i] = compute_row_deviance(M1_rm, M2_rm, i);
        }
    }

    return dev;
}

}  // namespace splikit
