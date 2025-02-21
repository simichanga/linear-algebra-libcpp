add_executable(vector_matrix_benchmarks
    vector_matrix_benchmark.cpp # Path relative to benchmarks/
)

target_link_libraries(vector_matrix_benchmarks
    PRIVATE benchmark::benchmark
    linear_algebra # Link to library
)

target_include_directories(vector_matrix_benchmarks
    PRIVATE ${CMAKE_SOURCE_DIR}/include  # Include from the project root
)
