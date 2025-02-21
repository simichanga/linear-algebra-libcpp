#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

namespace linear_algebra {

    // Use a larger SmallSize for SIMD tests
    using LargeHybridVector = HybridVector<float, 256>;

    TEST(SimdTest, FloatDotProduct) {
        LargeHybridVector a(256);
        LargeHybridVector b(256);

        for (std::size_t i = 0; i < 256; ++i) {
            a[i] = 1.0f;
            b[i] = 2.0f;
        }

        float result = dot_simd(a, b);
        EXPECT_FLOAT_EQ(result, 512.0f);
    }

    TEST(SimdTest, UnalignedAccess) {
        LargeHybridVector a(127);
        LargeHybridVector b(127);

        for (std::size_t i = 0; i < 127; ++i) {
            a[i] = 1.0f;
            b[i] = 1.0f;
        }

        float result = dot_simd(a, b);
        EXPECT_FLOAT_EQ(result, 127.0f);
    }

    TEST(SimdTest, NonFloatFallback) {
        HybridVector<int, 256> a(256, 2);
        HybridVector<int, 256> b(256, 3);

        int result = dot_simd(a, b);
        EXPECT_EQ(result, 1536);
    }

} // namespace linear_algebra