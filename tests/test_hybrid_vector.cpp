#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

TEST(HybridVectorTest, SmallVector) {
    linear_algebra::HybridVector<float, 32> v1(10);
    linear_algebra::HybridVector<float, 32> v2(10);

    for (std::size_t i = 0; i < 10; ++i) {
        v1[i] = i;
        v2[i] = 2 * i;
    }

    auto v3 = v1 + v2;
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(v3[i], 3 * i);
    }

    EXPECT_EQ(v1.dot(v2), 570);        // Call dot on v1 (or v2)
    EXPECT_EQ(linear_algebra::dot_simd(v1, v2), 570);
}

TEST(HybridVectorTest, LargeVector) {
    linear_algebra::HybridVector<float, 32> v1(100);
    linear_algebra::HybridVector<float, 32> v2(100);

    for (std::size_t i = 0; i < 100; ++i) {
        v1[i] = i;
        v2[i] = 2 * i;
    }

    auto v3 = v1 + v2;
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(v3[i], 3 * i);
    }

    EXPECT_EQ(v1.dot(v2), 2 * 99 * 100 * 199 / 6);  // Call dot on v1 (or v2)
    EXPECT_EQ(linear_algebra::dot_simd(v1, v2), 2 * 99 * 100 * 199 / 6);
}

TEST(HybridVectorTest, SIMDAlignment) {
    linear_algebra::HybridVector<float, 32> v1(16);
    linear_algebra::HybridVector<float, 32> v2(16);

    for (std::size_t i = 0; i < 16; ++i) {
        v1[i] = i + 1;
        v2[i] = (i + 1) * 2;
    }

    float expected_dot = v1.dot(v2);   // Call dot on v1 (or v2)
    float simd_result = linear_algebra::dot_simd(v1, v2);

    EXPECT_NEAR(simd_result, expected_dot, 1e-5);
}

TEST(HybridVectorTest, UnevenSizeVector) {
    linear_algebra::HybridVector<float, 32> v1(15);
    linear_algebra::HybridVector<float, 32> v2(15);

    for (std::size_t i = 0; i < 15; ++i) {
        v1[i] = i;
        v2[i] = i * 3;
    }

    float expected_dot = v1.dot(v2);   // Call dot on v1 (or v2)
    float simd_result = linear_algebra::dot_simd(v1, v2);

    EXPECT_NEAR(simd_result, expected_dot, 1e-5);
}
