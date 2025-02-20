cmake_minimum_required(VERSION 3.20)
project(LinearAlgebra VERSION 1.0 LANGUAGES CXX)

# Enable C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the library
add_subdirectory(src)

# Add tests (if enabled)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Add benchmarks (if enabled)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
