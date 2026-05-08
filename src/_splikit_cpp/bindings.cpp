// splikit-py C++ extension — pybind11 module entry point.
//
// All long-running kernels release the GIL via py::call_guard<py::gil_scoped_release>().

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "deviance.hpp"
#include "make_m2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_splikit_cpp, m) {
    m.doc() = "splikit-py C++ kernels (Eigen + OpenMP)";

#ifdef SPLIKIT_USE_OPENMP
    m.attr("__openmp__") = true;
#else
    m.attr("__openmp__") = false;
#endif

    m.def("make_m2", &splikit::make_m2,
          py::arg("M1"), py::arg("group_ids"), py::arg("n_threads") = 1,
          "Build the LJV exclusion matrix M2 from M1 and a dense 0..G-1 group_ids vector.",
          py::call_guard<py::gil_scoped_release>());

    m.def("calc_deviances_ratio", &splikit::calc_deviances_ratio,
          py::arg("M1"), py::arg("M2"), py::arg("n_threads") = 1,
          "Per-event ratio binomial deviance for one library (M1 + M2 sub-matrices).",
          py::call_guard<py::gil_scoped_release>());
}
