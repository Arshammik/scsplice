#include "deviance.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifdef SPLIKIT_USE_OPENMP
#include <omp.h>
#endif

namespace splikit {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;

Eigen::VectorXd
calc_deviances_ratio(const SpMat& M1, const SpMat& M2, int n_threads) {
    if (M1.rows() != M2.rows() || M1.cols() != M2.cols()) {
        throw std::invalid_argument(
            "calc_deviances_ratio: M1 and M2 must have identical shapes");
    }

    const std::int64_t n_rows = M1.rows();
    const std::int64_t n_cols = M1.cols();
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(n_rows);

    if (n_rows == 0 || n_cols == 0) {
        return dev;
    }

    // CSC coeff() requires compressed storage.
    SpMat M1c = M1;
    SpMat M2c = M2;
    if (!M1c.isCompressed()) M1c.makeCompressed();
    if (!M2c.isCompressed()) M2c.makeCompressed();

    auto compute_row = [&](std::int64_t i) -> double {
        double sum_y = 0.0;
        double sum_n = 0.0;
        for (std::int64_t k = 0; k < n_cols; ++k) {
            const double y = M1c.coeff(i, k);
            const double f = M2c.coeff(i, k);
            sum_y += y;
            sum_n += (y + f);
        }
        if (sum_n <= 0.0) return 0.0;
        const double p_hat = sum_y / sum_n;
        // Hard clamp matches R splikit/src/calcDeviances.cpp:38 (no epsilon).
        if (p_hat <= 0.0 || p_hat >= 1.0) return 0.0;

        double dev_row = 0.0;
        for (std::int64_t k = 0; k < n_cols; ++k) {
            const double y = M1c.coeff(i, k);
            const double f = M2c.coeff(i, k);
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
    };

#ifdef SPLIKIT_USE_OPENMP
    if (n_threads > 1) {
        omp_set_num_threads(n_threads);
        #pragma omp parallel for schedule(dynamic)
        for (std::int64_t i = 0; i < n_rows; ++i) {
            dev[i] = compute_row(i);
        }
    } else
#endif
    {
        for (std::int64_t i = 0; i < n_rows; ++i) {
            dev[i] = compute_row(i);
        }
    }

    return dev;
}

}  // namespace splikit
