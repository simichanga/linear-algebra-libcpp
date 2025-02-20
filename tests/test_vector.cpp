#include <gtest/gtest.h>
#include <linear_algebra/vector.hpp>

// Test fixture for Vector
class VectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        v1 = linear_algebra::Vector<double, 3>{1.0, 2.0, 3.0};
        v2 = linear_algebra::Vector<double, 3>{4.0, 5.0, 6.0};
    }

    linear_algebra::Vector<double, 3> v1;
    linear_algebra::Vector<double, 3> v2;
};

// Test default constructor and operator[]
TEST_F(VectorTest, AccessOperator) {
    EXPECT_EQ(v1[0], 1.0);
    EXPECT_EQ(v1[1], 2.0);
    EXPECT_EQ(v1[2], 3.0);

    // Test modification
    v1[1] = 4.0;
    EXPECT_EQ(v1[1], 4.0);
}

// Test vector addition
TEST_F(VectorTest, Addition) {
    auto v3 = v1 + v2;
    EXPECT_EQ(v3[0], 5.0);
    EXPECT_EQ(v3[1], 7.0);
    EXPECT_EQ(v3[2], 9.0);
}

// Test dot product
TEST_F(VectorTest, DotProduct) {
    double dot_result = v1.dot(v2);
    EXPECT_EQ(dot_result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
}

// Test const operator[]
TEST_F(VectorTest, ConstAccessOperator) {
    const linear_algebra::Vector<double, 3> v4{7.0, 8.0, 9.0};
    EXPECT_EQ(v4[0], 7.0);
    EXPECT_EQ(v4[1], 8.0);
    EXPECT_EQ(v4[2], 9.0);
}

// Test with a different size
TEST(VectorTestDifferentSize, Operations) {
    linear_algebra::Vector<double, 2> v5{1.0, 2.0};
    linear_algebra::Vector<double, 2> v6{3.0, 4.0};

    // Test addition
    auto v7 = v5 + v6;
    EXPECT_EQ(v7[0], 4.0);
    EXPECT_EQ(v7[1], 6.0);

    // Test dot product
    double dot_result2 = v5.dot(v6);
    EXPECT_EQ(dot_result2, 1.0 * 3.0 + 2.0 * 4.0);
}
