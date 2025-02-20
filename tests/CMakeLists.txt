find_package(GTest REQUIRED)

add_executable(test_vector test_vector.cpp)
add_executable(test_block_matrix test_block_matrix.cpp)

target_link_libraries(test_vector PRIVATE linear_algebra GTest::GTest GTest::Main)
target_link_libraries(test_block_matrix PRIVATE linear_algebra GTest::GTest GTest::Main)

add_test(NAME test_vector COMMAND test_vector)
add_test(NAME test_block_matrix COMMAND test_block_matrix)
