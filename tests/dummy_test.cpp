#include "../include/linear_algebra/block_matrix.hpp"
#include <iostream>

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
  BlockMatrix<double> m1;
  std::cout << "Rows: " << m1.get_rows() << ", Cols: " << m1.get_cols() << '\n';
  m1 = BlockMatrix<double>(2, 2);
  std::cout << "Rows: " << m1.get_rows() << ", Cols: " << m1.get_cols() << '\n';
  
  m1[0, 0] = 1.0; m1[0, 1] = 2.0;
  m1[1, 0] = 3.0; m1[1, 1] = 4.0;
  
  print(m1);
  
  auto m2 = m1.transpose();
  
  print(m2);
}
