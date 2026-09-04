#include "hvg_vst.hpp"

#include <cmath>
#include <stdexcept>

#ifdef SCSPLICE_USE_OPENMP
#include <omp.h>
#endif

namespace scsplice {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace {

SpMatRM to_row_major(const SpMat& X) {
    SpMatRM out;
    if (X.isCompressed()) {
        out = X;
    } else {
        SpMat tmp = X;
        tmp.makeCompressed();
        out = tmp;
    }
    out.makeCompressed();
    return out;
}

inline void compute_row_mean_var(const SpMatRM& X_rm, Eigen::Index i, Eigen::Index n_cols,
                                 double& mean_out, double& var_out) {
    double sum = 0.0;
    double sum2 = 0.0;
    for (SpMatRM::InnerIterator it(X_rm, i); it; ++it) {
        const double v = it.value();
        sum += v;
        sum2 += v * v;
    }
    const double n = static_cast<double>(n_cols);
    const double mu = sum / n;
    // Zeros contribute 0 to sum2, so dividing by the full column count (not
    // nnz) already yields the "including zeros" mean-of-squares R computes.
    const double mean_of_squares = sum2 / n;
    mean_out = mu;
    var_out = mean_of_squares - mu * mu;
}

inline double clamp_sym(double z, double vmax) {
    if (z > vmax) return vmax;
    if (z < -vmax) return -vmax;
    return z;
}

inline double compute_row_standardized_variance(const SpMatRM& X_rm, Eigen::Index i,
                                                 Eigen::Index n_cols, double mean_i,
                                                 double sd_i, double vmax) {
    double acc = 0.0;
    Eigen::Index nnz = 0;
    for (SpMatRM::InnerIterator it(X_rm, i); it; ++it) {
        const double z = clamp_sym((it.value() - mean_i) / sd_i, vmax);
        acc += z * z;
        ++nnz;
    }
    const Eigen::Index n_zero = n_cols - nnz;
    if (n_zero > 0) {
        const double z0 = clamp_sym((0.0 - mean_i) / sd_i, vmax);
        acc += static_cast<double>(n_zero) * z0 * z0;
    }
    return (n_cols > 1) ? (acc / static_cast<double>(n_cols - 1)) : 0.0;
}

}  // namespace

std::tuple<Eigen::VectorXd, Eigen::VectorXd>
hvg_row_mean_var(const SpMat& X, int n_threads) {
    const Eigen::Index n_rows = X.rows();
    const Eigen::Index n_cols = X.cols();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_rows);
    Eigen::VectorXd var = Eigen::VectorXd::Zero(n_rows);
    if (n_rows == 0 || n_cols == 0) {
        return {mean, var};
    }

    SpMatRM X_rm = to_row_major(X);

#ifdef SCSPLICE_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            compute_row_mean_var(X_rm, i, n_cols, mean[i], var[i]);
        }
    } else
#endif
    {
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            compute_row_mean_var(X_rm, i, n_cols, mean[i], var[i]);
        }
    }

    return {mean, var};
}

Eigen::VectorXd
hvg_standardize_variance(const SpMat& X, const Eigen::Ref<const Eigen::VectorXd>& mean,
                         const Eigen::Ref<const Eigen::VectorXd>& sd, int n_threads) {
    const Eigen::Index n_rows = X.rows();
    const Eigen::Index n_cols = X.cols();
    if (mean.size() != n_rows || sd.size() != n_rows) {
        throw std::invalid_argument(
            "hvg_standardize_variance: mean/sd must have length X.rows()");
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(n_rows);
    if (n_rows == 0 || n_cols == 0) {
        return result;
    }

    SpMatRM X_rm = to_row_major(X);
    const double vmax = std::sqrt(static_cast<double>(n_cols));

#ifdef SCSPLICE_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            result[i] = compute_row_standardized_variance(X_rm, i, n_cols, mean[i], sd[i], vmax);
        }
    } else
#endif
    {
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            result[i] = compute_row_standardized_variance(X_rm, i, n_cols, mean[i], sd[i], vmax);
        }
    }

    return result;
}

}  // namespace scsplice
