#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

namespace linear_algebra {

    TEST(AccessTest, ElementOperations) {
        HybridVector<int, 32> v(10);

        // Test write/read
        v[0] = 42;
        v[9] = 100;
        EXPECT_EQ(v[0], 42);
        EXPECT_EQ(v[9], 100);

        // Test const access
        const auto& cv = v;
        EXPECT_EQ(cv[0], 42);
    }

    TEST(AccessTest, BoundaryChecks) {
        HybridVector<int, 32> v(10);

        EXPECT_THROW(v[10], std::out_of_range);
        EXPECT_THROW(v[static_cast<size_t>(-1)], std::out_of_range);
    }

    TEST(AccessTest, LargeVectorAccess) {
        HybridVector<int, 33> v(1000);
        v[999] = 123;
        EXPECT_EQ(v[999], 123);
        EXPECT_THROW(v[1000], std::out_of_range);
    }

}