#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace linear_algebra {

template <typename T, std::size_t N>
class StaticVector {
public:
    // Default constructor
    constexpr StaticVector() = default;

    // Constructor for brace initialization
    template <typename... Args>
    constexpr StaticVector(Args... args) : data_{args...} {
        static_assert(sizeof...(Args) == N, "Number of arguments must match vector size.");
    }

    // Array index operator with deducing this and perfect forwarding
    template <typename Self>
    constexpr auto& operator[](this Self&& self, std::size_t i) {
        return std::forward<Self>(self).data_[i];
    }

    // Arithmetic operations
    constexpr StaticVector operator+(const StaticVector& other) const {
        StaticVector result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    constexpr T dot(const StaticVector& other) const {
        T result{};
        for (std::size_t i = 0; i < N; ++i) {
            result += (*this)[i] * other[i];
        }
        return result;
    }

private:
    std::array<T, N> data_{};
};

} // namespace linear_algebra
