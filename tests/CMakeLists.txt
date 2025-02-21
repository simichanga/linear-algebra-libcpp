find_package(GTest REQUIRED)

set(TEST_DIRECTORIES block_matrix static_vector) # TODO: add hybrid_vector later on

foreach(TEST_DIR ${TEST_DIRECTORIES})
    file(GLOB TEST_FILES "${TEST_DIR}/*.cpp")

    foreach(TEST_FILE ${TEST_FILES})
        # Extract the test name (without the path and extension)
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)  # Use CMake's built-in function

        # Construct the executable name (e.g., test_block_matrix_initializer_list)
        set(EXECUTABLE_NAME test_${TEST_DIR}_${TEST_NAME})

        add_executable(${EXECUTABLE_NAME} ${TEST_FILE})
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE linear_algebra GTest::GTest GTest::Main)
        add_test(NAME ${EXECUTABLE_NAME} COMMAND ${EXECUTABLE_NAME})
    endforeach()
endforeach()