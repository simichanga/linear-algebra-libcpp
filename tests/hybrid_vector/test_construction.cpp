#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

namespace linear_algebra {

    TEST(ConstructionTest, DefaultSmallStorage) {
        HybridVector<int, 32> v(30);
        EXPECT_EQ(v.size(), 30);

        // Test resize within small bounds
        v.resize(32);
        EXPECT_EQ(v.size(), 32);

        // Test overflow
        EXPECT_THROW(v.resize(33), std::length_error);
    }

    TEST(ConstructionTest, LargeStorageInitialization) {
        HybridVector<int, 33> v(100);  // Uses vector storage
        EXPECT_EQ(v.size(), 100);

        // Test large resize
        v.resize(200);
        EXPECT_EQ(v.size(), 200);
    }

    TEST(ConstructionTest, EdgeCases) {
        // Empty vector
        HybridVector<int, 32> empty(0);
        EXPECT_EQ(empty.size(), 0);

        // Exact small size
        HybridVector<int, 32> exact(32);
        EXPECT_NO_THROW(exact.resize(32));

        // Invalid initialization
        EXPECT_THROW((HybridVector<int, 32>(33)), std::out_of_range);
    }

}