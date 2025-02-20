#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace linear_algebra {

template <typename T, std::size_t N>
class Vector {
public:
    // Default constructor
    constexpr Vector() = default;

    // Constructor for brace initialization
    template <typename... Args>
    constexpr Vector(Args... args) : data_{args...} {
        static_assert(sizeof...(Args) == N, "Number of arguments must match vector size.");
    }

    // Array index operator with deducing this and perfect forwarding
    template <typename Self>
    constexpr auto& operator[](this Self&& self, std::size_t i) {
        return std::forward<Self>(self).data_[i];
    }

    // Arithmetic operations
    constexpr Vector operator+(const Vector& other) const {
        Vector result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    constexpr T dot(const Vector& other) const {
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
