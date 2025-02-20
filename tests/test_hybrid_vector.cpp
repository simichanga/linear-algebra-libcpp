#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

TEST(HybridVectorTest, SmallVector) {
    linear_algebra::HybridVector<double, 32> v1(10);
    linear_algebra::HybridVector<double, 32> v2(10);

    for (std::size_t i = 0; i < 10; ++i) {
        v1[i] = i;
        v2[i] = 2 * i;
    }

    auto v3 = v1 + v2;
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(v3[i], 3 * i);
    }

    EXPECT_EQ(dot(v1, v2), 570); // 0*0 + 1*2 + 2*4 + ... + 9*18 = 570
}

TEST(HybridVectorTest, LargeVector) {
    linear_algebra::HybridVector<double, 32> v1(100);
    linear_algebra::HybridVector<double, 32> v2(100);

    for (std::size_t i = 0; i < 100; ++i) {
        v1[i] = i;
        v2[i] = 2 * i;
    }

    auto v3 = v1 + v2;
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(v3[i], 3 * i);
    }

    EXPECT_EQ(dot(v1, v2), 2 * 99 * 100 * 199 / 6); // Sum of 2*i^2 for i=0 to 99
}
