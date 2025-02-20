#include <gtest/gtest.h>
#include <linear_algebra/block_matrix.hpp>

// Test fixture for BlockMatrix
class BlockMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        m1 = linear_algebra::BlockMatrix<double>(2, 2);
        m2 = linear_algebra::BlockMatrix<double>(2, 2);

        // Initialize matrices
        m1[0, 0] = 1.0; m1[0, 1] = 2.0;
        m1[1, 0] = 3.0; m1[1, 1] = 4.0;

        m2[0, 0] = 5.0; m2[0, 1] = 6.0;
        m2[1, 0] = 7.0; m2[1, 1] = 8.0;
    }

    linear_algebra::BlockMatrix<double> m1;
    linear_algebra::BlockMatrix<double> m2;
};

// Test multidimensional subscript operator
TEST_F(BlockMatrixTest, SubscriptOperator) {
    EXPECT_EQ((m1[0, 0]), 1.0);
    EXPECT_EQ((m1[0, 1]), 2.0);
    EXPECT_EQ((m1[1, 0]), 3.0);
    EXPECT_EQ((m1[1, 1]), 4.0);

    // Test modification
    m1[1, 1] = 10.0;
    EXPECT_EQ((m1[1, 1]), 10.0);
}

// Test matrix addition
TEST_F(BlockMatrixTest, Addition) {
    auto m3 = m1 + m2;
    EXPECT_EQ((m3[0, 0]), 6.0);
    EXPECT_EQ((m3[0, 1]), 8.0);
    EXPECT_EQ((m3[1, 0]), 10.0);
    EXPECT_EQ((m3[1, 1]), 12.0);
}

// Test matrix multiplication
TEST_F(BlockMatrixTest, Multiplication) {
    linear_algebra::BlockMatrix<double> m3(2, 3);
    linear_algebra::BlockMatrix<double> m4(3, 2);

    // Initialize matrices
    m3[0, 0] = 1.0; m3[0, 1] = 2.0; m3[0, 2] = 3.0;
    m3[1, 0] = 4.0; m3[1, 1] = 5.0; m3[1, 2] = 6.0;

    m4[0, 0] = 7.0; m4[0, 1] = 8.0;
    m4[1, 0] = 9.0; m4[1, 1] = 10.0;
    m4[2, 0] = 11.0; m4[2, 1] = 12.0;

    auto m5 = m3 * m4;
    EXPECT_EQ((m5[0, 0]), 58.0);
    EXPECT_EQ((m5[0, 1]), 64.0);
    EXPECT_EQ((m5[1, 0]), 139.0);
    EXPECT_EQ((m5[1, 1]), 154.0);
}

// Test transpose
TEST_F(BlockMatrixTest, Transpose) {
    auto m3 = m1.transpose();
    EXPECT_EQ((m3[0, 0]), 1.0);
    EXPECT_EQ((m3[0, 1]), 3.0);
    EXPECT_EQ((m3[1, 0]), 2.0);
    EXPECT_EQ((m3[1, 1]), 4.0);
}

// Test out-of-range access
TEST_F(BlockMatrixTest, OutOfRangeAccess) {
    EXPECT_THROW((m1[2, 0]), std::out_of_range);
    EXPECT_THROW((m1[0, 2]), std::out_of_range);
}
