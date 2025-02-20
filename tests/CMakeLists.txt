find_package(GTest REQUIRED)

add_executable(test_static_vector test_static_vector.cpp)
add_executable(test_block_matrix test_block_matrix.cpp)
add_executable(test_hybrid_vector test_hybrid_vector.cpp)

target_link_libraries(test_static_vector PRIVATE linear_algebra GTest::GTest GTest::Main)
target_link_libraries(test_block_matrix PRIVATE linear_algebra GTest::GTest GTest::Main)
target_link_libraries(test_hybrid_vector PRIVATE linear_algebra GTest::GTest GTest::Main)

add_test(NAME test_static_vector COMMAND test_static_vector)
add_test(NAME test_block_matrix COMMAND test_block_matrix)
add_test(NAME test_hybrid_vector COMMAND test_block_matrix)
