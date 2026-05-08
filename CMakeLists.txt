cmake_minimum_required(VERSION 3.18)

project(splikit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Find Python (Development.Module is the right component for pybind11_add_module)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11 from the Python site-packages of the build environment (scikit-build-core
# installs it at build time via [build-system].requires)
find_package(pybind11 CONFIG REQUIRED)

# Eigen3 (header-only); fail loudly if not found
find_package(Eigen3 NO_MODULE REQUIRED)

# OpenMP (optional); kernels degrade gracefully when absent
find_package(OpenMP)

add_subdirectory(src/_splikit_cpp)
