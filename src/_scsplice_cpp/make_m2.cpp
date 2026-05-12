#include "make_m2.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef SCSPLICE_USE_OPENMP
#include <omp.h>
#endif

namespace scsplice {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using StorageIndex = SpMat::StorageIndex;  // == int (default)

namespace {

// Per-column work, shared between Pass 1 (count) and Pass 2 (fill).
//
// We walk the column with InnerIterator once and stash a thread-local
// (row -> value) scratch keyed by row index. Avoiding M1.coeff(i, j) here
// removes an O(log nnz_col) binary search per access; with the doubly-nested
// per-group / per-member loops below the prior coeff() pattern was
// ~10^10 searches per call on realistic input.
//
// `m1_col_value` is a dense scratch of length n_events shared across columns
// within a thread: we record which row indices we touched in `touched_rows`
// (size O(nnz in column)) so the per-column reset cost stays in O(nnz_col).
struct ColumnScratch {
    std::vector<double> group_sum;     // size n_groups
    std::vector<double> m1_col_value;  // size n_events; sparse: only touched rows are non-zero
    std::vector<Eigen::Index> touched_rows;
};

inline void
accumulate_group_sums(const SpMat& M1, const Eigen::Ref<const Eigen::VectorXi>& group_ids,
                      Eigen::Index j, ColumnScratch& s) {
    // Reset only the slots we touched on the previous call.
    for (Eigen::Index r : s.touched_rows) {
        s.m1_col_value[r] = 0.0;
    }
    s.touched_rows.clear();
    std::fill(s.group_sum.begin(), s.group_sum.end(), 0.0);

    for (SpMat::InnerIterator it(M1, j); it; ++it) {
        const Eigen::Index row = it.row();
        const double v = it.value();
        s.group_sum[group_ids[row]] += v;
        s.m1_col_value[row] = v;
        s.touched_rows.push_back(row);
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

    const Eigen::Index n_events = M1_in.rows();
    const Eigen::Index n_cells = M1_in.cols();

    if (n_events == 0 || n_cells == 0) {
        return SpMat(n_events, n_cells);
    }

    // Alias if already compressed (the Python wrapper passes compressed CSC);
    // copy only when forced to call makeCompressed().
    const SpMat* M1_ptr = &M1_in;
    SpMat M1_compressed_copy;
    if (!M1_in.isCompressed()) {
        M1_compressed_copy = M1_in;
        M1_compressed_copy.makeCompressed();
        M1_ptr = &M1_compressed_copy;
    }
    const SpMat& M1 = *M1_ptr;

    // Pre-loop: validate group_ids and build group_members in encounter order.
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

    std::vector<std::vector<Eigen::Index>> group_members(n_groups);
    for (Eigen::Index i = 0; i < group_ids.size(); ++i) {
        group_members[group_ids[i]].push_back(i);
    }

    // Pass 1: count nnz per column.
    std::vector<Eigen::Index> col_nnz(static_cast<std::size_t>(n_cells), 0);

    auto count_column = [&](Eigen::Index j, ColumnScratch& s) {
        accumulate_group_sums(M1, group_ids, j, s);
        Eigen::Index cnt = 0;
        for (int g = 0; g < n_groups; ++g) {
            if (s.group_sum[g] == 0.0) continue;
            for (Eigen::Index i : group_members[g]) {
                const double m2_val = s.group_sum[g] - s.m1_col_value[i];
                if (m2_val != 0.0) ++cnt;
            }
        }
        col_nnz[j] = cnt;
    };

#ifdef SCSPLICE_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            ColumnScratch s;
            s.group_sum.assign(n_groups, 0.0);
            s.m1_col_value.assign(static_cast<std::size_t>(n_events), 0.0);
            s.touched_rows.reserve(64);
            #pragma omp for schedule(dynamic)
            for (Eigen::Index j = 0; j < n_cells; ++j) {
                count_column(j, s);
            }
        }
    } else
#endif
    {
        ColumnScratch s;
        s.group_sum.assign(n_groups, 0.0);
        s.m1_col_value.assign(static_cast<std::size_t>(n_events), 0.0);
        s.touched_rows.reserve(64);
        for (Eigen::Index j = 0; j < n_cells; ++j) {
            count_column(j, s);
        }
    }

    // Sequential prefix-sum: fixes per-column write offsets so Pass 2 can write
    // to disjoint slices of the shared output without any cross-thread sync.
    std::vector<Eigen::Index> col_ptr(static_cast<std::size_t>(n_cells + 1), 0);
    for (Eigen::Index j = 0; j < n_cells; ++j) {
        col_ptr[j + 1] = col_ptr[j] + col_nnz[j];
    }
    const Eigen::Index total_nnz = col_ptr[n_cells];

    std::vector<Eigen::Index> row_indices(static_cast<std::size_t>(total_nnz));
    std::vector<double> values(static_cast<std::size_t>(total_nnz));

    auto fill_column = [&](Eigen::Index j, ColumnScratch& s) {
        accumulate_group_sums(M1, group_ids, j, s);
        std::vector<std::pair<Eigen::Index, double>> col_entries;
        col_entries.reserve(static_cast<std::size_t>(col_nnz[j]));
        for (int g = 0; g < n_groups; ++g) {
            if (s.group_sum[g] == 0.0) continue;
            for (Eigen::Index i : group_members[g]) {
                const double m2_val = s.group_sum[g] - s.m1_col_value[i];
                if (m2_val != 0.0) col_entries.emplace_back(i, m2_val);
            }
        }
        std::sort(col_entries.begin(), col_entries.end());
        Eigen::Index write_pos = col_ptr[j];
        for (const auto& entry : col_entries) {
            row_indices[write_pos] = entry.first;
            values[write_pos] = entry.second;
            ++write_pos;
        }
    };

    // Pass 2: fill values, sort per column, write to disjoint output slices.
#ifdef SCSPLICE_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            ColumnScratch s;
            s.group_sum.assign(n_groups, 0.0);
            s.m1_col_value.assign(static_cast<std::size_t>(n_events), 0.0);
            s.touched_rows.reserve(64);
            #pragma omp for schedule(dynamic)
            for (Eigen::Index j = 0; j < n_cells; ++j) {
                fill_column(j, s);
            }
        }
    } else
#endif
    {
        ColumnScratch s;
        s.group_sum.assign(n_groups, 0.0);
        s.m1_col_value.assign(static_cast<std::size_t>(n_events), 0.0);
        s.touched_rows.reserve(64);
        for (Eigen::Index j = 0; j < n_cells; ++j) {
            fill_column(j, s);
        }
    }

    // Manual CSC construction preserves our deterministic per-column ordering.
    // setFromTriplets would re-aggregate and could re-order; not used here.
    SpMat result(n_events, n_cells);
    // Use the matrix's own StorageIndex type so reserve() matches without
    // a narrowing cast.
    Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> nnz_per_col(
        static_cast<Eigen::Index>(n_cells));
    for (Eigen::Index j = 0; j < n_cells; ++j) {
        nnz_per_col[j] = static_cast<StorageIndex>(col_nnz[j]);
    }
    result.reserve(nnz_per_col);
    for (Eigen::Index j = 0; j < n_cells; ++j) {
        const Eigen::Index start = col_ptr[j];
        const Eigen::Index end = col_ptr[j + 1];
        for (Eigen::Index k = start; k < end; ++k) {
            result.insert(static_cast<StorageIndex>(row_indices[k]),
                          static_cast<StorageIndex>(j)) = values[k];
        }
    }
    result.makeCompressed();
    return result;
}

}  // namespace scsplice
