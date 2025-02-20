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
        throw std::out_of_range("Vector size exceeds small size limit.");
    }

    template <typename Self>
    constexpr auto&& operator[](this Self&& self, std::size_t index) {
      if (index >= std::forward<Self>(self).size_) [[unlikely]]
        throw std::out_of_range("Index out of bounds");
      return std::forward<Self>(self).data_[index];
    }

    constexpr std::size_t size() const noexcept { return size_; }
    
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

    // Forwarding references in arithmetic operations
    template <typename V1, typename V2>
    friend constexpr auto operator+(V1&& v1, V2&& v2)
      requires  std::same_as<std::remove_cvref_t<V1>, HybridVector> &&
                std::same_as<std::remove_cvref_t<V2>, HybridVector> 
    {
      HybridVector result(v1.size());
      for (std::size_t i = 0; i < v1.size(); ++i)
        result[i] = std::forward<V1>(v1)[i] + std::forward<V2>(v2)[i];
      return result;
    }

    template <typename V1, typename V2>
    friend constexpr auto dot(V1&& v1, V2&& v2)
      requires  std::same_as<std::remove_cvref_t<V1>, HybridVector> &&
                std::same_as<std::remove_cvref_t<V2>, HybridVector> 
    {
      using ResultType = decltype(std::declval<T>() * std::declval<T>());
      ResultType result{};
      for (std::size_t i = 0; i < v1.size(); ++i)
        result += std::forward<V1>(v1)[i] * std::forward<V2>(v2)[i];
      return result;
    }
    
  };

  template <typename T>
  T dot_simd(const HybridVector<T>& a, const HybridVector<T>& b) {
    if (a.size() != b.size())
      throw std::invalid_argument("Vector sizes must match for dot product.");
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
      for (std::size_t j = 0; j < 8; ++j)
        result += temp[j];
    }
    for (; i < a.size(); ++i)
      result += a[i] * b[i];
    return result;
  }
}
