#!/usr/bin/fish

# Exit on error
set -e

# Variables
set BUILD_DIR "build"
set TEST_EXECUTABLE "tests/test_vector"

# Clean the build directory
echo "Cleaning build directory..."
rm -rf $BUILD_DIR

# Configure the project
echo "Configuring the project with CMake..."
cmake -S . -B $BUILD_DIR -DBUILD_TESTS=ON

# Build the project
echo "Building the project..."
cmake --build $BUILD_DIR

# Run the tests
echo "Running tests..."
cd $BUILD_DIR
ctest --output-on-failure

echo "Build and test process completed successfully!"
