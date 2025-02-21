#pragma once

#include <array>
#include <concepts>
#include <utility>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <immintrin.h> // simd instructions

namespace linear_algebra {
  template <typename T, std::size_t SmallSize = 32>
  class HybridVector {
  private:
    // Determine the storage type based on size
    using StorageType = typename std::conditional<
      (SmallSize) <= 32,
      std::array<T, SmallSize>,
      std::vector<T>
    >::type;

    StorageType data_;
    std::size_t size_;

  public:
    HybridVector(std::size_t size) : size_(size) {
      if constexpr (std::is_same_v<StorageType, std::vector<T>>)
        data_.resize(size);
      else if (size > SmallSize)
        data_ = std::vector<T>(size);
      else
        data_ = std::array<T, SmallSize>();
    }

    HybridVector(std::size_t size, const T& initial_value)
      : HybridVector(size) { data_.fill(initial_value); }

    HybridVector(std::initializer_list<T> init)
      : HybridVector(init.size())
    {
      std::size_t i = 0;
      for (const auto& value : init) {
        data_[i++] = value;
      }
    }

    template <typename Self>
    constexpr auto&& operator[](this Self&& self, std::size_t index) {
      if (index >= std::forward<Self>(self).size_) [[unlikely]]
        throw std::out_of_range("Index out of bounds");
      return std::forward<Self>(self).data_[index];
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

    constexpr void resize(std::size_t new_size) {
      if constexpr (requires { data_.resize(new_size); }) {
        data_.resize(new_size);
        size_ = new_size;
      } else {
        if (new_size > SmallSize) [[unlikely]] {
          throw std::length_error("Resize beyond small buffer capacity");
        }
        [[assume(new_size <= SmallSize)]];
        size_ = new_size;
      }
    }

    // Add a type alias for value_type
    using value_type = T;

    template <typename V> // More generic
    constexpr auto dot(const HybridVector<V>& other) const {
      if (size() != other.size()) {
          throw std::invalid_argument("Vector sizes must match for dot product.");
      }

      using ResultType = std::common_type_t<T, V>; // Handle different types

      ResultType result{};
      for (std::size_t i = 0; i < size(); ++i) {
        result += (*this)[i] * other[i];
      }
      return result;
    }
  };

  template <typename T1, std::size_t S1, typename T2, std::size_t S2>
  constexpr auto operator+(const HybridVector<T1, S1>& v1, const HybridVector<T2, S2>& v2) {
    using ResultType = std::common_type_t<T1, T2>; // Determine the result type
    constexpr std::size_t ResultSmallSize = (S1 <= S2) ? S1 : S2; // Use the smaller block size

    if (v1.size() != v2.size()) {
      throw std::invalid_argument("Vector sizes must match for addition.");
    }

    HybridVector<ResultType, ResultSmallSize> result(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
      result[i] = v1[i] + v2[i];
    }
    return result;
  }

  template <typename T, std::size_t SmallSize>
T dot_simd(const HybridVector<T, SmallSize>& a, const HybridVector<T, SmallSize>& b) {
    if (a.size() != b.size()) {
      throw std::invalid_argument("Vector sizes must match for dot product.");
    }

    T result{};
    std::size_t i = 0;

    // Process 8 elements at a time (AVX2)
    if constexpr (std::is_same_v<T, float>) {
      __m256 sum = _mm256_setzero_ps();
      for (; i + 8 <= a.size(); i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
      }
      alignas(32) float temp[8];
      _mm256_store_ps(temp, sum);
      for (std::size_t j = 0; j < 8; ++j) {
        result += temp[j];
      }
    }

    // Process remaining elements
    for (; i < a.size(); ++i) {
      result += a[i] * b[i];
    }

    return result;
  }
}