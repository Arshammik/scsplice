#include "pseudo_r2.hpp"

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

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace {

// Match R splikit src/cpp_pseudoR2.cpp:7,48-49 exactly. EPS is load-bearing
// per CLAUDE.md and must not be replaced by std::numeric_limits epsilon.
constexpr double EPS = 1e-8;
constexpr int MAX_ITER = 100;
constexpr double TOL = 1e-6;

inline double clamp_p(double p) {
    if (std::isnan(p)) return std::numeric_limits<double>::quiet_NaN();
    if (p < EPS) return EPS;
    if (p > 1.0 - EPS) return 1.0 - EPS;
    return p;
}

inline double sigmoid_stable(double e) {
    // Numerically stable form: 1/(1+exp(-e)) is bounded in [0,1] for all
    // finite e (no inf/inf overflow). Matches R splikit's behaviour: R uses
    // the same stable form via base R's plogis(); the IRLS clamp then
    // pulls borderline values to EPS / (1-EPS). Naive exp(e)/(1+exp(e))
    // overflows past |e| ~= 710 and silently NaN-propagates, NaN-ing 16
    // separating-hyperplane events that R fits cleanly.
    return 1.0 / (1.0 + std::exp(-e));
}

// Per-event IRLS scratch hoisted out of the IRLS loop. Sized once per event
// (resize is a no-op when capacity is sufficient).
struct EventScratch {
    Eigen::MatrixXd X;            // n_valid x 2
    Eigen::VectorXd y;            // n_valid
    Eigen::VectorXd n_trials;     // n_valid
    Eigen::VectorXd eta;          // n_valid
    Eigen::VectorXd p_vec;        // n_valid
    Eigen::VectorXd w;            // n_valid
    Eigen::VectorXd z_work;       // n_valid
};

double per_event(const Eigen::Ref<const Eigen::MatrixXd>& Z,
                 const SpMatRM& M1_rm, const SpMatRM& M2_rm,
                 Eigen::Index i, const std::string& metric,
                 EventScratch& s) {
    const Eigen::Index n_cells = M1_rm.cols();
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    (void)n_cells;

    // Collect cells with (M1[i,k] + M2[i,k]) > 0 by walking the row-i
    // iterators jointly. Zero-zero cells contribute nothing to (m1+m2 > 0)
    // selection so they can be skipped entirely.
    struct Cell { Eigen::Index c; double m1; double m2; };
    std::vector<Cell> cells;
    cells.reserve(64);
    SpMatRM::InnerIterator it1(M1_rm, i);
    SpMatRM::InnerIterator it2(M2_rm, i);
    while (it1 || it2) {
        const Eigen::Index c1 = it1 ? it1.col() : std::numeric_limits<Eigen::Index>::max();
        const Eigen::Index c2 = it2 ? it2.col() : std::numeric_limits<Eigen::Index>::max();
        double m1v = 0.0;
        double m2v = 0.0;
        Eigen::Index c;
        if (c1 == c2) {
            c = c1;
            m1v = it1.value();
            m2v = it2.value();
            ++it1; ++it2;
        } else if (c1 < c2) {
            c = c1;
            m1v = it1.value();
            ++it1;
        } else {
            c = c2;
            m2v = it2.value();
            ++it2;
        }
        if (m1v + m2v > 0.0) {
            cells.push_back({c, m1v, m2v});
        }
    }

    const int n_valid = static_cast<int>(cells.size());
    if (n_valid < 2) return NaN;

    // Resize scratch (cheap once capacity is reached).
    s.X.resize(n_valid, 2);
    s.y.resize(n_valid);
    s.n_trials.resize(n_valid);
    s.eta.resize(n_valid);
    s.p_vec.resize(n_valid);
    s.w.resize(n_valid);
    s.z_work.resize(n_valid);

    double sum_m1 = 0.0;
    double sum_m2 = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const Eigen::Index c = cells[j].c;
        s.X(j, 0) = 1.0;
        // Z is documented as events x cells (see header): Z(event, cell).
        s.X(j, 1) = Z(i, c);
        const double m1c = cells[j].m1;
        const double m2c = cells[j].m2;
        s.y[j] = m1c;
        s.n_trials[j] = m1c + m2c;
        sum_m1 += m1c;
        sum_m2 += m2c;
    }
    if (sum_m1 <= 0.0 || sum_m2 <= 0.0) return NaN;

    Eigen::Vector2d beta = Eigen::Vector2d::Zero();
    Eigen::Vector2d beta_new;

    for (int it = 0; it < MAX_ITER; ++it) {
        // eta = X * beta
        s.eta.noalias() = s.X * beta;
        for (int j = 0; j < n_valid; ++j) {
            const double p_j = clamp_p(sigmoid_stable(s.eta[j]));
            s.p_vec[j] = p_j;
            const double mu_j = s.n_trials[j] * p_j;
            s.w[j] = s.n_trials[j] * p_j * (1.0 - p_j);
            s.z_work[j] = s.eta[j] + (s.y[j] - mu_j) / (s.w[j] + EPS);
        }

        Eigen::Matrix2d XtWX = Eigen::Matrix2d::Zero();
        Eigen::Vector2d XtWz = Eigen::Vector2d::Zero();
        for (int j = 0; j < n_valid; ++j) {
            const double w_j = s.w[j];
            const double x0 = s.X(j, 0);
            const double x1 = s.X(j, 1);
            XtWX(0, 0) += w_j * x0 * x0;
            XtWX(0, 1) += w_j * x0 * x1;
            XtWX(1, 0) += w_j * x1 * x0;
            XtWX(1, 1) += w_j * x1 * x1;
            XtWz[0] += w_j * x0 * s.z_work[j];
            XtWz[1] += w_j * x1 * s.z_work[j];
        }

        // Closed-form 2x2 inverse via Cramer's rule. Bit-equivalent to LAPACK
        // dgesv on a 2x2 system (which Armadillo's solve(no_approx) calls).
        // Eigen's ColPivHouseholderQR rejects near-singular matrices via a
        // relative rank threshold; that threshold differs from dgesv's
        // exactly-zero-pivot test and produced 16 spurious NaN events vs R.
        const double a11 = XtWX(0, 0);
        const double a12 = XtWX(0, 1);
        const double a21 = XtWX(1, 0);
        const double a22 = XtWX(1, 1);
        const double det = a11 * a22 - a12 * a21;
        // Documented exact-zero pivot check to match R's dgesv. Below we
        // also guard against det values so close to zero that 1/det is
        // non-finite (magnitude ~ 1e-300 -> inf), which the exact-zero
        // check alone would miss.
        if (!std::isfinite(det) || det == 0.0) return NaN;
        const double inv_det = 1.0 / det;
        beta_new[0] = inv_det * (a22 * XtWz[0] - a12 * XtWz[1]);
        beta_new[1] = inv_det * (a11 * XtWz[1] - a21 * XtWz[0]);
        if (!std::isfinite(beta_new[0]) || !std::isfinite(beta_new[1])) {
            return NaN;
        }
        if ((beta_new - beta).cwiseAbs().maxCoeff() < TOL) {
            beta = beta_new;
            break;
        }
        beta = beta_new;
    }

    // Final-beta deviance (R does NOT NaN on non-convergence; uses iter-100 beta).
    // No `+ EPS` inside the log: clamp_p already bounds p in [EPS, 1-EPS], so
    // mu = n_trials * p >= n_trials * EPS > 0 whenever n_trials > 0. The
    // `+ EPS` would introduce a ~1e-8 systematic bias and violate the
    // published rtol=1e-9 tolerance band.
    s.eta.noalias() = s.X * beta;
    double D_full = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const double p_j = clamp_p(sigmoid_stable(s.eta[j]));
        const double mu_j = s.n_trials[j] * p_j;
        const double m1_j = s.y[j];
        const double m2_j = s.n_trials[j] - m1_j;
        if (m1_j > 0.0) {
            D_full += 2.0 * m1_j * std::log(m1_j / mu_j);
        }
        if (m2_j > 0.0) {
            D_full += 2.0 * m2_j * std::log(m2_j / (s.n_trials[j] - mu_j));
        }
    }

    // Null deviance.
    double p_hat = sum_m1 / (sum_m1 + sum_m2);
    p_hat = clamp_p(p_hat);
    double D_null = 0.0;
    for (int j = 0; j < n_valid; ++j) {
        const double mu_j = s.n_trials[j] * p_hat;
        const double m1_j = s.y[j];
        const double m2_j = s.n_trials[j] - m1_j;
        if (m1_j > 0.0) {
            D_null += 2.0 * m1_j * std::log(m1_j / mu_j);
        }
        if (m2_j > 0.0) {
            D_null += 2.0 * m2_j * std::log(m2_j / (s.n_trials[j] - mu_j));
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
    // Z is events x cells: Z.rows() == n_events, Z.cols() == n_cells.
    // See header for the orientation contract.
    if (Z.rows() != M1.rows() || Z.cols() != M1.cols()) {
        throw std::invalid_argument(
            "pseudo_correlation: Z must be shape (n_events, n_cells) — same "
            "shape as M1/M2 (events x cells layout). Got mismatched dims.");
    }
    if (metric != "CoxSnell" && metric != "Nagelkerke") {
        throw std::invalid_argument(
            "pseudo_correlation: metric must be 'CoxSnell' or 'Nagelkerke'");
    }

    const Eigen::Index n_events = M1.rows();
    Eigen::VectorXd result(n_events);
    if (n_events == 0) return result;

    // Build row-major views once. Zero coeff() calls inside the per-event
    // hot loop. Aliasing the input when it is already compressed.
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
        #pragma omp parallel num_threads(n_threads)
        {
            EventScratch s;
            #pragma omp for schedule(dynamic)
            for (Eigen::Index i = 0; i < n_events; ++i) {
                result[i] = per_event(Z, M1_rm, M2_rm, i, metric, s);
            }
        }
    } else
#endif
    {
        EventScratch s;
        for (Eigen::Index i = 0; i < n_events; ++i) {
            result[i] = per_event(Z, M1_rm, M2_rm, i, metric, s);
        }
    }

    return result;
}

}  // namespace splikit
