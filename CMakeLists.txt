cmake_minimum_required(VERSION 3.20)
project(LinearAlgebra VERSION 1.0 LANGUAGES CXX)

# Enable C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile_commands.json for clang lsp server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup google tests
find_package(benchmark REQUIRED)

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

# Install targets (all from the root CMakeLists.txt)
install(TARGETS linear_algebra
    EXPORT linear_algebra-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/linear_algebra
    DESTINATION include
)
