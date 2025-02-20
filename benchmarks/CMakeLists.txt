find_package(benchmark REQUIRED)

add_executable(benchmark_vector benchmark_vector.cpp)

target_link_libraries(benchmark_vector PRIVATE linear_algebra benchmark::benchmark)
