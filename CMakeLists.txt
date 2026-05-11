cmake_minimum_required(VERSION 3.18)

project(splikit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Optional build switch: skip OpenMP entirely (useful on macOS where libomp
# discovery from cibuildwheel is fragile — see wheels.yml).
option(SPLIKIT_DISABLE_OPENMP "Disable OpenMP even when CMake can find it" OFF)

# Find Python (Development.Module is the right component for pybind11_add_module)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Make `cmake -S . -B build` work outside scikit-build-core by pointing
# pybind11's CMake config at the build-environment site-packages install.
# scikit-build-core sets pybind11_DIR itself, so a non-empty hint here is
# only used when invoking CMake directly.
if(NOT DEFINED pybind11_DIR OR pybind11_DIR STREQUAL "")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_lookup_result
    )
    if(_pybind11_lookup_result EQUAL 0 AND _pybind11_cmake_dir)
        set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake config dir")
        message(STATUS "Discovered pybind11_DIR from Python: ${pybind11_DIR}")
    endif()
endif()

# pybind11 from the Python site-packages of the build environment (scikit-build-core
# installs it at build time via [build-system].requires)
find_package(pybind11 CONFIG REQUIRED)

# Eigen3 (header-only); fail loudly if not found
find_package(Eigen3 NO_MODULE REQUIRED)

# OpenMP (optional); kernels degrade gracefully when absent.
if(NOT SPLIKIT_DISABLE_OPENMP)
    find_package(OpenMP)
endif()

add_subdirectory(src/_splikit_cpp)
