#include <gtest/gtest.h>
#include <linear_algebra/block_matrix.hpp>

using namespace linear_algebra;

TEST(BlockMatrixTest, DefaultConstructorCreatesEmptyMatrix) {
    const BlockMatrix<int> m;
    EXPECT_EQ(m.get_rows(), 0);
    EXPECT_EQ(m.get_cols(), 0);
}

TEST(BlockMatrixTest, ConstructorInitializesCorrectDimensions) {
    const BlockMatrix<int, 32> m(35, 40);
    EXPECT_EQ(m.get_rows(), 35);
    EXPECT_EQ(m.get_cols(), 40);
}