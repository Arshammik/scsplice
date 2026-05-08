# Match R's CRAN baseline so floating-point reduction order is deterministic
# across our build vs the R reference. Override per-platform if needed.
set(SPLIKIT_CXX_OPTS "")
if(NOT MSVC)
    list(APPEND SPLIKIT_CXX_OPTS
        -march=nocona
        -mtune=haswell
        -ftree-vectorize
    )
endif()

pybind11_add_module(_splikit_cpp
    bindings.cpp
)

target_link_libraries(_splikit_cpp PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_splikit_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_splikit_cpp PRIVATE SPLIKIT_USE_OPENMP)
endif()

target_compile_options(_splikit_cpp PRIVATE ${SPLIKIT_CXX_OPTS})

# Hidden visibility + sibling-.so RPATH; small wheels, no symbol clashes
set_target_properties(_splikit_cpp PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
)
if(APPLE)
    set_target_properties(_splikit_cpp PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(_splikit_cpp PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _splikit_cpp DESTINATION splikit)
