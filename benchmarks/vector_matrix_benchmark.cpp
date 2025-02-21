#include <benchmark/benchmark.h>
#include <linear_algebra/static_vector.hpp>
#include <linear_algebra/hybrid_vector.hpp>

// Benchmark for StaticVector addition
static void BM_StaticVectorAdd(benchmark::State& state) {
    constexpr size_t N = 1024; // Example size
    linear_algebra::StaticVector<int, N> v1{};
    linear_algebra::StaticVector<int, N> v2{};
    for (int i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        linear_algebra::StaticVector<int, N> result = v1 + v2;
        benchmark::DoNotOptimize(result); // Prevent unwanted optimizations
    }
}
BENCHMARK(BM_StaticVectorAdd);

// Benchmark for StaticVector dot product
static void BM_StaticVectorDot(benchmark::State& state) {
    constexpr size_t N = 1024; // Example size
    linear_algebra::StaticVector<int, N> v1{};
    linear_algebra::StaticVector<int, N> v2{};
    for (int i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        int result = v1.dot(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_StaticVectorDot);



// Benchmark for HybridVector addition (small size)
static void BM_HybridVectorAddSmall(benchmark::State& state) {
    constexpr size_t N = 32; // Small size
    linear_algebra::HybridVector<int> v1(N);
    linear_algebra::HybridVector<int> v2(N);
    for (std::size_t i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        linear_algebra::HybridVector<int> result = v1 + v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_HybridVectorAddSmall);

// Benchmark for HybridVector addition (large size)
static void BM_HybridVectorAddLarge(benchmark::State& state) {
    constexpr size_t N = 1024; // Large size
    linear_algebra::HybridVector<int, N> v1(N);
    linear_algebra::HybridVector<int, N> v2(N);
    for (std::size_t i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        linear_algebra::HybridVector<int, N> result = v1 + v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_HybridVectorAddLarge);

// Benchmark for HybridVector dot product (small size)
static void BM_HybridVectorDotSmall(benchmark::State& state) {
    constexpr size_t N = 32; // Small size
    linear_algebra::HybridVector<int> v1(N);
    linear_algebra::HybridVector<int> v2(N);
    for (std::size_t i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        int result = v1.dot(v2);  // Corrected: Call dot on v1 (or v2)
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_HybridVectorDotSmall);


// Benchmark for HybridVector dot product (large size)
static void BM_HybridVectorDotLarge(benchmark::State& state) {
    constexpr size_t N = 1024; // Large size
    linear_algebra::HybridVector<int, N> v1(N);
    linear_algebra::HybridVector<int, N> v2(N);
    for (std::size_t i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        int result = v1.dot(v2);  // Corrected: Call dot on v1 (or v2)
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_HybridVectorDotLarge);

//Benchmark for HybridVector dot product with SIMD (large size)
static void BM_HybridVectorDotSimdLarge(benchmark::State& state) {
    constexpr size_t N = 1024; // Large size
    linear_algebra::HybridVector<float, N> v1(N);
    linear_algebra::HybridVector<float, N> v2(N);
    for (std::size_t i = 0; i < N; ++i) {
        v1[i] = i;
        v2[i] = N - i;
    }

    for (auto _ : state) {
        float result = linear_algebra::dot_simd(v1, v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_HybridVectorDotSimdLarge);

BENCHMARK_MAIN();
