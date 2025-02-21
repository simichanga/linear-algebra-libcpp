add_library(linear_algebra
    INTERFACE
    # Source files (if any)
)

target_include_directories(linear_algebra
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional: SIMD instructions
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_options(linear_algebra INTERFACE -mavx2)
endif()
