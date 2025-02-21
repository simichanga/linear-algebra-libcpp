#include <gtest/gtest.h>
#include <linear_algebra/block_matrix.hpp>

using namespace linear_algebra;

TEST(BlockMatrixTest, ElementAccessAndModification) {
    BlockMatrix<int, 32> m(32, 32);
    m[0, 0] = 42;
    EXPECT_EQ((m[0, 0]), 42);

    m[31, 31] = 100;
    EXPECT_EQ((m[31, 31]), 100);
}

TEST(BlockMatrixTest, CrossBlockElementAccess) {
    BlockMatrix<int, 32> m(33, 34);
    m[32, 33] = 5;
    EXPECT_EQ((m[32, 33]), 5);
}

TEST(BlockMatrixTest, OutOfBoundsAccessThrows) {
    BlockMatrix<int> m(10, 10);
    EXPECT_THROW((m[10, 0]), std::out_of_range);
    EXPECT_THROW((m[0, 10]), std::out_of_range);
    EXPECT_THROW((m[10, 10]), std::out_of_range);
}