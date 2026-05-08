#include "make_m2.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef SPLIKIT_USE_OPENMP
#include <omp.h>
#endif

namespace splikit {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;

namespace {

// Per-column work, shared between Pass 1 (count) and Pass 2 (fill).
// Re-evaluating M1.coeff(i, j) in both passes (rather than caching) matches the
// R kernel and avoids an O(nnz_output) auxiliary buffer.
inline void
accumulate_group_sums(const SpMat& M1, const Eigen::Ref<const Eigen::VectorXi>& group_ids,
                      std::int64_t j, std::vector<double>& group_sum) {
    std::fill(group_sum.begin(), group_sum.end(), 0.0);
    for (SpMat::InnerIterator it(M1, j); it; ++it) {
        const auto row = static_cast<std::int64_t>(it.row());
        group_sum[group_ids[row]] += it.value();
    }
}

}  // namespace

SpMat make_m2(const SpMat& M1_in,
              const Eigen::Ref<const Eigen::VectorXi>& group_ids,
              int n_threads) {
    if (group_ids.size() != M1_in.rows()) {
        throw std::invalid_argument(
            "make_m2: group_ids.size() must equal M1.rows()");
    }

    const std::int64_t n_events = M1_in.rows();
    const std::int64_t n_cells = M1_in.cols();

    if (n_events == 0 || n_cells == 0) {
        return SpMat(n_events, n_cells);
    }

    // CSC InnerIterator + coeff() require compressed storage.
    SpMat M1 = M1_in;
    if (!M1.isCompressed()) {
        M1.makeCompressed();
    }

    // Pre-loop: build group_members in encounter order (NOT sorted).
    // group_ids are dense 0..G-1 (precondition; the Python wrapper validates).
    int max_group = 0;
    for (Eigen::Index i = 0; i < group_ids.size(); ++i) {
        if (group_ids[i] < 0) {
            throw std::invalid_argument(
                "make_m2: group_ids must be non-negative");
        }
        if (group_ids[i] > max_group) {
            max_group = group_ids[i];
        }
    }
    const int n_groups = max_group + 1;

    std::vector<std::vector<std::int64_t>> group_members(n_groups);
    for (Eigen::Index i = 0; i < group_ids.size(); ++i) {
        group_members[group_ids[i]].push_back(static_cast<std::int64_t>(i));
    }

    // Pass 1: count nnz per column.
    std::vector<std::int64_t> col_nnz(static_cast<std::size_t>(n_cells), 0);

#ifdef SPLIKIT_USE_OPENMP
    if (n_threads > 1) {
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            std::vector<double> group_sum(n_groups);
            #pragma omp for schedule(dynamic)
            for (std::int64_t j = 0; j < n_cells; ++j) {
                accumulate_group_sums(M1, group_ids, j, group_sum);
                std::int64_t cnt = 0;
                for (int g = 0; g < n_groups; ++g) {
                    if (group_sum[g] == 0.0) continue;
                    for (std::int64_t i : group_members[g]) {
                        const double m2_val = group_sum[g] - M1.coeff(i, j);
                        if (m2_val != 0.0) ++cnt;
                    }
                }
                col_nnz[j] = cnt;
            }
        }
    } else
#endif
    {
        std::vector<double> group_sum(n_groups);
        for (std::int64_t j = 0; j < n_cells; ++j) {
            accumulate_group_sums(M1, group_ids, j, group_sum);
            std::int64_t cnt = 0;
            for (int g = 0; g < n_groups; ++g) {
                if (group_sum[g] == 0.0) continue;
                for (std::int64_t i : group_members[g]) {
                    const double m2_val = group_sum[g] - M1.coeff(i, j);
                    if (m2_val != 0.0) ++cnt;
                }
            }
            col_nnz[j] = cnt;
        }
    }

    // Sequential prefix-sum: fixes per-column write offsets so Pass 2 can write
    // to disjoint slices of the shared output without any cross-thread sync.
    std::vector<std::int64_t> col_ptr(static_cast<std::size_t>(n_cells + 1), 0);
    for (std::int64_t j = 0; j < n_cells; ++j) {
        col_ptr[j + 1] = col_ptr[j] + col_nnz[j];
    }
    const std::int64_t total_nnz = col_ptr[n_cells];

    std::vector<std::int64_t> row_indices(static_cast<std::size_t>(total_nnz));
    std::vector<double> values(static_cast<std::size_t>(total_nnz));

    // Pass 2: fill values, sort per column, write to disjoint output slices.
#ifdef SPLIKIT_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel
        {
            std::vector<double> group_sum(n_groups);
            #pragma omp for schedule(dynamic)
            for (std::int64_t j = 0; j < n_cells; ++j) {
                accumulate_group_sums(M1, group_ids, j, group_sum);
                std::vector<std::pair<std::int64_t, double>> col_entries;
                col_entries.reserve(static_cast<std::size_t>(col_nnz[j]));
                for (int g = 0; g < n_groups; ++g) {
                    if (group_sum[g] == 0.0) continue;
                    for (std::int64_t i : group_members[g]) {
                        const double m2_val = group_sum[g] - M1.coeff(i, j);
                        if (m2_val != 0.0) col_entries.emplace_back(i, m2_val);
                    }
                }
                std::sort(col_entries.begin(), col_entries.end());
                std::int64_t write_pos = col_ptr[j];
                for (const auto& entry : col_entries) {
                    row_indices[write_pos] = entry.first;
                    values[write_pos] = entry.second;
                    ++write_pos;
                }
            }
        }
    } else
#endif
    {
        std::vector<double> group_sum(n_groups);
        for (std::int64_t j = 0; j < n_cells; ++j) {
            accumulate_group_sums(M1, group_ids, j, group_sum);
            std::vector<std::pair<std::int64_t, double>> col_entries;
            col_entries.reserve(static_cast<std::size_t>(col_nnz[j]));
            for (int g = 0; g < n_groups; ++g) {
                if (group_sum[g] == 0.0) continue;
                for (std::int64_t i : group_members[g]) {
                    const double m2_val = group_sum[g] - M1.coeff(i, j);
                    if (m2_val != 0.0) col_entries.emplace_back(i, m2_val);
                }
            }
            std::sort(col_entries.begin(), col_entries.end());
            std::int64_t write_pos = col_ptr[j];
            for (const auto& entry : col_entries) {
                row_indices[write_pos] = entry.first;
                values[write_pos] = entry.second;
                ++write_pos;
            }
        }
    }

    // Manual CSC construction preserves our deterministic per-column ordering.
    // setFromTriplets would re-aggregate and could re-order; not used here.
    SpMat result(n_events, n_cells);
    Eigen::VectorXi nnz_per_col(static_cast<Eigen::Index>(n_cells));
    for (std::int64_t j = 0; j < n_cells; ++j) {
        nnz_per_col[j] = static_cast<int>(col_nnz[j]);
    }
    result.reserve(nnz_per_col);
    for (std::int64_t j = 0; j < n_cells; ++j) {
        const std::int64_t start = col_ptr[j];
        const std::int64_t end = col_ptr[j + 1];
        for (std::int64_t k = start; k < end; ++k) {
            result.insert(row_indices[k], j) = values[k];
        }
    }
    result.makeCompressed();
    return result;
}

}  // namespace splikit
