// scsplice C++ extension — pybind11 module entry point.
//
// All long-running kernels release the GIL via py::call_guard<py::gil_scoped_release>().

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "deviance.hpp"
#include "hvg_vst.hpp"
#include "make_m2.hpp"
#include "pseudo_r2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_scsplice_cpp, m) {
    m.doc() = "scsplice C++ kernels (Eigen + OpenMP)";

#ifdef SCSPLICE_USE_OPENMP
    m.attr("__openmp__") = true;
#else
    m.attr("__openmp__") = false;
#endif

    m.def("make_m2", &scsplice::make_m2,
          py::arg("M1"), py::arg("group_ids"), py::arg("n_threads") = 1,
          "Build the LJV exclusion matrix M2 from M1 and a dense 0..G-1 group_ids vector.\n"
          "M1 is scipy.sparse.csc_matrix (events x cells layout), group_ids is int32.",
          py::call_guard<py::gil_scoped_release>());

    m.def("calc_deviances_ratio", &scsplice::calc_deviances_ratio,
          py::arg("M1"), py::arg("M2"), py::arg("n_threads") = 1,
          "Per-event ratio binomial deviance for one library (M1 + M2 sub-matrices, "
          "events x cells layout, scipy.sparse.csc_matrix float64).",
          py::call_guard<py::gil_scoped_release>());

    m.def("pseudo_correlation", &scsplice::pseudo_correlation,
          py::arg("Z"), py::arg("M1"), py::arg("M2"), py::arg("metric"),
          py::arg("n_threads") = 1,
          "Per-event signed pseudo-correlation (Cox-Snell or Nagelkerke) via "
          "logistic IRLS.\n\n"
          "Z orientation contract: Z is (n_events, n_cells) — the kernel reads "
          "Z(i, c) for event i, cell c. M1 and M2 must use the same events x "
          "cells layout (scipy.sparse.csc_matrix, float64) and match Z's shape. "
          "The Python wrapper enforces this on its side.\n\n"
          "Returns one scalar per event in [-1, 1] or NaN.",
          py::call_guard<py::gil_scoped_release>());

    m.def("hvg_row_mean_var", &scsplice::hvg_row_mean_var,
          py::arg("X"), py::arg("n_threads") = 1,
          "Per-row (gene) mean and population variance (including zeros) of a "
          "sparse genes x cells matrix, scipy.sparse.csc_matrix float64.\n"
          "Returns (mean, var), each length X.shape[0].",
          py::call_guard<py::gil_scoped_release>());

    m.def("hvg_standardize_variance", &scsplice::hvg_standardize_variance,
          py::arg("X"), py::arg("mean"), py::arg("sd"), py::arg("n_threads") = 1,
          "Seurat/R-splikit VST standardization pass: per-row clamped "
          "z-score sum of squares, normalized by (n_cells - 1).\n"
          "X is genes x cells (scipy.sparse.csc_matrix float64); mean/sd are "
          "length X.shape[0] (pass sd=1.0 for rows with no fitted trend).",
          py::call_guard<py::gil_scoped_release>());
}
