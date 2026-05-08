// splikit-py C++ extension — pybind11 module entry point.
//
// v1.0 will expose:
//   - SplikitKernel.set_M1i_M2i(M1i, M2i)
//   - SplikitKernel.find_variable_events(min_row_sum)
//   - SplikitKernel.pseudo_correlation(zdb, metric)
//   - make_m2(M1, group_ids, n_threads)
//
// All long-running methods are wrapped in py::call_guard<py::gil_scoped_release>().

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_splikit_cpp, m) {
    m.doc() = "splikit-py C++ kernels (Eigen + OpenMP)";

#ifdef SPLIKIT_USE_OPENMP
    m.attr("__openmp__") = true;
#else
    m.attr("__openmp__") = false;
#endif

    // Kernels land here in subsequent commits.
}
