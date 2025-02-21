#include <gtest/gtest.h>
#include <linear_algebra/block_matrix.hpp>

using namespace linear_algebra;

TEST(BlockMatrixTest, MatrixMultiplication) {
    // Test case: 2x3 matrix multiplied by 3x2 matrix
    BlockMatrix<int> a(2, 3);
    BlockMatrix<int> b(3, 2);

    // Initialize matrix a with all 1s
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            a[i, j] = 1;
        }
    }

    // Initialize matrix b with all 2s
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            b[i, j] = 2;
        }
    }

    // Perform multiplication
    auto c = a * b;

    // Assert the result: expected 2x2 matrix with all 6s
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ((c[i, j]), 6);
        }
    }
}

TEST(BlockMatrixTest, IncompatibleMultiplicationThrows) {
    // Test case: Attempt to multiply matrices with incompatible dimensions
    const BlockMatrix<int> a(2, 3);
    const BlockMatrix<int> b(2, 3);

    // Expect an invalid_argument exception to be thrown
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(BlockMatrixTest, MatrixTranspose) {
    // Test case: Transpose a 3x4 matrix
    BlockMatrix<int> a(3, 4);
    a[0, 1] = 5;  // Set some specific values
    a[2, 3] = 10;

    // Perform transpose
    auto b = a.transpose();

    // Assert the dimensions of the transposed matrix
    EXPECT_EQ(b.get_rows(), 4);
    EXPECT_EQ(b.get_cols(), 3);

    // Assert the transposed values are in the correct positions
    EXPECT_EQ((b[1, 0]), 5);
    EXPECT_EQ((b[3, 2]), 10);
}

TEST(BlockMatrixTest, SingleElementBlocks) {
    // Test case: Create a matrix with block size 1 and transpose it
    BlockMatrix<int, 1> m(3, 3);
    m[0, 0] = 1;
    m[1, 1] = 2;
    m[2, 2] = 3;

    // Perform transpose
    auto t = m.transpose();

    // Assert the transposed values
    EXPECT_EQ((t[1, 1]), 2);
    EXPECT_EQ((t[0, 0]), 1);
    EXPECT_EQ((t[2, 2]), 3);
}

TEST(BlockMatrixTest, EmptyMatrixOperations) {
    // Test case: Perform addition and multiplication on empty matrices
    const BlockMatrix<int> a(0, 0);
    const BlockMatrix<int> b(0, 0);

    // Perform addition
    const auto c = a + b;
    EXPECT_EQ(c.get_rows(), 0);
    EXPECT_EQ(c.get_cols(), 0);

    // Perform multiplication
    const auto d = a * b;
    EXPECT_EQ(d.get_rows(), 0);
    EXPECT_EQ(d.get_cols(), 0);
}

TEST(BlockMatrixTest, BoundaryElementHandling) {
    // Test case: Access elements at the boundary of blocks and test out-of-bounds access
    BlockMatrix<int, 32> m(33, 34);
    m[32, 33] = 100; // Access element at the edge of a block

    // Assert the value at the boundary
    EXPECT_EQ((m[32, 33]), 100);

    // Expect out-of-bounds access to throw an exception
    EXPECT_THROW((m[33, 33]), std::out_of_range);
    EXPECT_THROW((m[32, 34]), std::out_of_range);
}