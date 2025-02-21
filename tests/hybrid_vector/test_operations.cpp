#include <gtest/gtest.h>
#include <linear_algebra/hybrid_vector.hpp>

namespace linear_algebra {

    TEST(OperationsTest, VectorAddition) {
        HybridVector<int, 32> a(10, 2);
        HybridVector<int, 32> b(10, 3);
        auto c = a + b;

        for(size_t i = 0; i < c.size(); ++i) {
            EXPECT_EQ(c[i], 5);
        }
    }

    TEST(OperationsTest, DotProduct) {
        HybridVector<int, 32> a({1, 2, 3});
        HybridVector<int, 32> b({4, 5, 6});

        EXPECT_EQ(a.dot(b), 32);
    }

    TEST(OperationsTest, MixedTypeOperations) {
        HybridVector<int, 32> a(10, 2);       // Vector of 10 int elements, all 2
        HybridVector<double, 32> b(10, 3.5);  // Vector of 10 double elements, all 3.5

        auto c = a + b;
        EXPECT_DOUBLE_EQ(c[0], 5.5);
        EXPECT_DOUBLE_EQ(c[1], 5.5);
        EXPECT_DOUBLE_EQ(c[9], 5.5);

        // Verify the type of the result
        static_assert(std::is_same_v<decltype(c)::value_type, double>);
    }

    TEST(OperationsTest, MixedTypeAddition) {
        // int + double
        HybridVector<int, 32> a = {1, 2, 3};
        HybridVector<double, 32> b = {1.5, 2.5, 3.5};
        auto c = a + b;
        EXPECT_DOUBLE_EQ(c[0], 2.5);
        EXPECT_DOUBLE_EQ(c[1], 4.5);
        EXPECT_DOUBLE_EQ(c[2], 6.5);

        // double + int
        auto d = b + a;
        EXPECT_DOUBLE_EQ(d[0], 2.5);
        EXPECT_DOUBLE_EQ(d[1], 4.5);
        EXPECT_DOUBLE_EQ(d[2], 6.5);

        // Verify result types
        static_assert(std::is_same_v<decltype(c)::value_type, double>);
        static_assert(std::is_same_v<decltype(d)::value_type, double>);
    }

    TEST(OperationsTest, MixedTypeEdgeCases) {
        // Empty vectors
        HybridVector<int, 32> empty_int(0);
        HybridVector<double, 32> empty_double(0);
        auto result_empty = empty_int + empty_double;
        EXPECT_EQ(result_empty.size(), 0);

        // Different sizes should still throw
        HybridVector<int, 32> a(5);
        HybridVector<double, 32> b(6);
        EXPECT_THROW(a + b, std::invalid_argument);
    }

    TEST(OperationsTest, InvalidOperations) {
        HybridVector<int, 32> a(5);
        HybridVector<int, 32> b(6);

        EXPECT_THROW(a + b, std::invalid_argument);
        EXPECT_THROW(a.dot(b), std::invalid_argument);
    }

} // namespace linear_algebra