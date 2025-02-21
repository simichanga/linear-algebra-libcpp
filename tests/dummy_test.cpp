#include "../include/linear_algebra/block_matrix.hpp"
#include <iostream>
#include "../include/linear_algebra/hybrid_vector.hpp"

using namespace linear_algebra;

template <typename T>
void print(const BlockMatrix<T>& input) {
  for (int i = 0; i < input.get_rows(); ++i) {
    for (int j = 0; j < input.get_cols(); ++j) {
      std::cout << input[i, j] << ' ';
    }
    std::cout << '\n';
  }
}

int main() {
  HybridVector<double, 1024> numbers(1024);
  std::cout << "LOL!\n";
}
