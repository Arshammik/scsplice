# Compiler flags. On x86_64 we match R's CRAN baseline (-march=nocona
# -mtune=haswell -ftree-vectorize) so floating-point reduction order is
# deterministic across our build vs the R reference. On other ISAs (notably
# macOS-arm64, since GitHub's macos-latest is Apple Silicon since late 2024)
# clang rejects the x86-only flags; let the compiler pick its own defaults
# there. Vectorisation is enabled by default at -O2/-O3 on every supported
# compiler regardless of these flags.
set(SCSPLICE_CXX_OPTS "")
if(NOT MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
        list(APPEND SCSPLICE_CXX_OPTS
            -march=nocona
            -mtune=haswell
            -ftree-vectorize
        )
    endif()
    # On non-x86 (notably macOS-arm64) we leave -march unset; -O3 already
    # enables auto-vectorisation on every supported compiler.
endif()

pybind11_add_module(_scsplice_cpp
    bindings.cpp
    make_m2.cpp
    deviance.cpp
    pseudo_r2.cpp
)

target_link_libraries(_scsplice_cpp PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_scsplice_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_scsplice_cpp PRIVATE SCSPLICE_USE_OPENMP)
endif()

target_compile_options(_scsplice_cpp PRIVATE ${SCSPLICE_CXX_OPTS})

# Hidden visibility + sibling-.so RPATH; small wheels, no symbol clashes
set_target_properties(_scsplice_cpp PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
)
if(APPLE)
    set_target_properties(_scsplice_cpp PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(_scsplice_cpp PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# DESTINATION is the Python package directory (still `splikit/` at this step;
# renamed to `scsplice/` in the follow-up Python rename commit).
install(TARGETS _scsplice_cpp DESTINATION splikit)
