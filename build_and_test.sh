#!/usr/bin/fish

# Exit on error
set -e

# Variables
set BUILD_DIR "build"
set TEST_EXECUTABLE "tests/test_vector"
set BENCHMARK_EXECUTABLE "benchmarks/vector_matrix_benchmarks"

# Clean the build directory
echo "Cleaning build directory..."
rm -rf $BUILD_DIR

# Configure the project
echo "Configuring the project with CMake..."
cmake -S . -B $BUILD_DIR -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

# Build the project
echo "Building the project..."
cmake --build $BUILD_DIR

# Link the compilation database
ln -s build/compile_commands.json compile_commands.json

# Run the tests
echo "Running tests..."
cd $BUILD_DIR
ctest --output-on-failure

# Go back to the root directory (optional)
cd ..
