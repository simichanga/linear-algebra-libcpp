#include <gtest/gtest.h>
#include <linear_algebra/block_matrix.hpp>

using namespace linear_algebra;

TEST(BlockMatrixTest, InitializerListConstructor) {
    BlockMatrix<int, 2> matrix = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    EXPECT_EQ(matrix.get_rows(), 3);
    EXPECT_EQ(matrix.get_cols(), 4);

    EXPECT_EQ((matrix[0, 0]), 1);
    EXPECT_EQ((matrix[0, 1]), 2);
    EXPECT_EQ((matrix[0, 2]), 3);
    EXPECT_EQ((matrix[0, 3]), 4);
    EXPECT_EQ((matrix[1, 0]), 5);
    EXPECT_EQ((matrix[1, 1]), 6);
    EXPECT_EQ((matrix[1, 2]), 7);
    EXPECT_EQ((matrix[1, 3]), 8);
    EXPECT_EQ((matrix[2, 0]), 9);
    EXPECT_EQ((matrix[2, 1]), 10);
    EXPECT_EQ((matrix[2, 2]), 11);
    EXPECT_EQ((matrix[2, 3]), 12);
}

TEST(BlockMatrixTest, InitializerListConstructor_NonSquare_Throws) {
    EXPECT_THROW((BlockMatrix<int, 2>({
        {1, 2},
        {3, 4, 5},
        {6}
    })), std::invalid_argument);
}

TEST(BlockMatrixTest, InitializerListConstructor_NonSquare_Valid) {
    BlockMatrix<int, 2> matrix = {
        {1, 2, 0},
        {3, 4, 5},
        {6, 0, 0}
    };

    EXPECT_EQ(matrix.get_rows(), 3);
    EXPECT_EQ(matrix.get_cols(), 3);

    EXPECT_EQ((matrix[0, 0]), 1);
    EXPECT_EQ((matrix[0, 1]), 2);
    EXPECT_EQ((matrix[0, 2]), 0);
    EXPECT_EQ((matrix[1, 0]), 3);
    EXPECT_EQ((matrix[1, 1]), 4);
    EXPECT_EQ((matrix[1, 2]), 5);
    EXPECT_EQ((matrix[2, 0]), 6);
    EXPECT_EQ((matrix[2, 1]), 0);
    EXPECT_EQ((matrix[2, 2]), 0);
}

TEST(BlockMatrixTest, InitializerListConstructor_Empty) {
    BlockMatrix<int> matrix = {};
    EXPECT_EQ(matrix.get_rows(), 0);
    EXPECT_EQ(matrix.get_cols(), 0);
}

TEST(BlockMatrixTest, InitializerListConstructor_InconsistentRows) {
    EXPECT_THROW((BlockMatrix<int>({
        {1, 2},
        {3, 4, 5}
    })), std::invalid_argument);
}

TEST(BlockMatrixTest, InitializerListConstructor_DifferentBlockSizes)
{
    BlockMatrix<int, 1> matrix1 = {
            {1, 2},
            {3, 4}
    };
    EXPECT_EQ((matrix1[0, 0]), 1);
    EXPECT_EQ((matrix1[0, 1]), 2);
    EXPECT_EQ((matrix1[1, 0]), 3);
    EXPECT_EQ((matrix1[1, 1]), 4);

    BlockMatrix<int, 2> matrix2 = {
            {1, 2, 3, 4},
            {5, 6, 7, 8}
    };
    EXPECT_EQ((matrix2[0, 0]), 1);
    EXPECT_EQ((matrix2[0, 1]), 2);
    EXPECT_EQ((matrix2[0, 2]), 3);
    EXPECT_EQ((matrix2[0, 3]), 4);
    EXPECT_EQ((matrix2[1, 0]), 5);
    EXPECT_EQ((matrix2[1, 1]), 6);
    EXPECT_EQ((matrix2[1, 2]), 7);
    EXPECT_EQ((matrix2[1, 3]), 8);
}