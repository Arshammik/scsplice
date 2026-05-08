#include "pseudo_r2.hpp"

#include <Eigen/QR>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPLIKIT_USE_OPENMP
#include <omp.h>
#endif

namespace splikit {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;

namespace {

// Match splikit/src/cpp_pseudoR2.cpp:7,48-49 exactly.
constexpr double EPS = 1e-8;
constexpr int MAX_ITER = 100;
constexpr double TOL = 1e-6;

inline double clamp_p(double p) {
    if (std::isnan(p)) return std::numeric_limits<double>::quiet_NaN();
    if (p < EPS) return EPS;
    if (p > 1.0 - EPS) return 1.0 - EPS;
    return p;
}

inline double sigmoid_naive(double e) {
    // Match R kernel verbatim (line ~55 in cpp_pseudoR2.cpp). Overflow at
    // |e| >> 0 produces NaN which is then clamped to EPS / (1-EPS) by clamp_p,
    // matching the R behaviour where the clamp also runs after the sigmoid.
    const double ee = std::exp(e);
    return ee / (1.0 + ee);
}

double per_event(const Eigen::Ref<const Eigen::MatrixXd>& Z,
                 const SpMat& M1, const SpMat& M2,
                 std::int64_t i, const std::string& metric) {
    const std::int64_t n_cells = M1.cols();
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    // Collect cells with (M1 + M2) > 0.
    std::vector<std::int64_t> idx;
    idx.reserve(static_cast<std::size_t>(n_cells));
    for (std::int64_t j = 0; j < n_cells; ++j) {
        const double y = M1.coeff(i, j);
        const double f = M2.coeff(i, j);
        if (y + f > 0.0) idx.push_back(j);
    }
    const int n_valid = static_cast<int>(idx.size());
    if (n_valid < 2) return NaN;

    Eigen::MatrixXd X(n_valid, 2);
    Eigen::VectorXd y(n_valid);
    Eigen::VectorXd n_trials(n_valid);
    double sum_m1 = 0.0;
    double sum_m2 = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const std::int64_t c = idx[j];
        X(j, 0) = 1.0;
        X(j, 1) = Z(i, c);
        const double m1c = M1.coeff(i, c);
        const double m2c = M2.coeff(i, c);
        y[j] = m1c;
        n_trials[j] = m1c + m2c;
        sum_m1 += m1c;
        sum_m2 += m2c;
    }
    if (sum_m1 <= 0.0 || sum_m2 <= 0.0) return NaN;

    Eigen::Vector2d beta = Eigen::Vector2d::Zero();
    Eigen::Vector2d beta_new;

    for (int it = 0; it < MAX_ITER; ++it) {
        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd p(n_valid);
        Eigen::VectorXd w(n_valid);
        Eigen::VectorXd z_work(n_valid);
        for (int j = 0; j < n_valid; ++j) {
            const double p_j = clamp_p(sigmoid_naive(eta[j]));
            p[j] = p_j;
            const double mu_j = n_trials[j] * p_j;
            w[j] = n_trials[j] * p_j * (1.0 - p_j);
            z_work[j] = eta[j] + (y[j] - mu_j) / (w[j] + EPS);
        }

        Eigen::Matrix2d XtWX = Eigen::Matrix2d::Zero();
        Eigen::Vector2d XtWz = Eigen::Vector2d::Zero();
        for (int j = 0; j < n_valid; ++j) {
            const double w_j = w[j];
            const double x0 = X(j, 0);
            const double x1 = X(j, 1);
            XtWX(0, 0) += w_j * x0 * x0;
            XtWX(0, 1) += w_j * x0 * x1;
            XtWX(1, 0) += w_j * x1 * x0;
            XtWX(1, 1) += w_j * x1 * x1;
            XtWz[0] += w_j * x0 * z_work[j];
            XtWz[1] += w_j * x1 * z_work[j];
        }

        Eigen::ColPivHouseholderQR<Eigen::Matrix2d> qr(XtWX);
        if (qr.rank() < 2) return NaN;
        beta_new = qr.solve(XtWz);
        if ((beta_new - beta).cwiseAbs().maxCoeff() < TOL) {
            beta = beta_new;
            break;
        }
        beta = beta_new;
    }

    // Final-beta deviance (R does NOT NaN on non-convergence; uses iter-100 beta).
    Eigen::VectorXd eta_final = X * beta;
    double D_full = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const double p_j = clamp_p(sigmoid_naive(eta_final[j]));
        if (std::isnan(p_j)) return NaN;
        const double mu_j = n_trials[j] * p_j;
        const double m1_j = y[j];
        const double m2_j = n_trials[j] - m1_j;
        if (m1_j > 0.0) {
            D_full += 2.0 * m1_j * std::log(m1_j / (mu_j + EPS));
        }
        if (m2_j > 0.0) {
            D_full += 2.0 * m2_j * std::log(m2_j / ((n_trials[j] - mu_j) + EPS));
        }
    }

    // Null deviance.
    double p_hat = sum_m1 / (sum_m1 + sum_m2);
    p_hat = clamp_p(p_hat);
    double D_null = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const double mu_j = n_trials[j] * p_hat;
        const double m1_j = y[j];
        const double m2_j = n_trials[j] - m1_j;
        if (m1_j > 0.0) {
            D_null += 2.0 * m1_j * std::log(m1_j / (mu_j + EPS));
        }
        if (m2_j > 0.0) {
            D_null += 2.0 * m2_j * std::log(m2_j / ((n_trials[j] - mu_j) + EPS));
        }
    }

    const double n_v = static_cast<double>(n_valid);
    double R2 = 1.0 - std::exp((D_full - D_null) / n_v);
    if (metric == "Nagelkerke") {
        const double denom = 1.0 - std::exp(-D_null / n_v);
        if (denom == 0.0) return NaN;
        R2 /= denom;
    }
    if (R2 < 0.0 || std::isnan(R2)) return NaN;

    const double r = std::sqrt(R2);
    return (beta[1] >= 0.0 ? r : -r);
}

}  // namespace

Eigen::VectorXd
pseudo_correlation(const Eigen::Ref<const Eigen::MatrixXd>& Z,
                   const SpMat& M1, const SpMat& M2,
                   const std::string& metric, int n_threads) {
    if (M1.rows() != M2.rows() || M1.cols() != M2.cols()) {
        throw std::invalid_argument(
            "pseudo_correlation: M1 and M2 must have identical shapes");
    }
    if (Z.rows() != M1.rows() || Z.cols() != M1.cols()) {
        throw std::invalid_argument(
            "pseudo_correlation: Z shape must equal M1 shape (events x cells)");
    }
    if (metric != "CoxSnell" && metric != "Nagelkerke") {
        throw std::invalid_argument(
            "pseudo_correlation: metric must be 'CoxSnell' or 'Nagelkerke'");
    }

    const std::int64_t n_events = M1.rows();
    Eigen::VectorXd result(n_events);
    if (n_events == 0) return result;

    SpMat M1c = M1;
    SpMat M2c = M2;
    if (!M1c.isCompressed()) M1c.makeCompressed();
    if (!M2c.isCompressed()) M2c.makeCompressed();

#ifdef SPLIKIT_USE_OPENMP
    if (n_threads > 1) {
        omp_set_num_threads(n_threads);
        #pragma omp parallel for schedule(dynamic)
        for (std::int64_t i = 0; i < n_events; ++i) {
            result[i] = per_event(Z, M1c, M2c, i, metric);
        }
    } else
#endif
    {
        for (std::int64_t i = 0; i < n_events; ++i) {
            result[i] = per_event(Z, M1c, M2c, i, metric);
        }
    }

    return result;
}

}  // namespace splikit
