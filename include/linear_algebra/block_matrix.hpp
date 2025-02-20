#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <utility>

namespace linear_algebra {

template <typename T, std::size_t BlockSize = 32>
class BlockMatrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::vector<std::unique_ptr<T[]>>> blocks_;

    constexpr std::pair<std::size_t, std::size_t> get_block_indices(std::size_t i, std::size_t j) const {
        return {i / BlockSize, j / BlockSize};
    }

    constexpr std::pair<std::size_t, std::size_t> get_in_block_indices(std::size_t i, std::size_t j) const {
        return {i % BlockSize, j % BlockSize};
    }
    
    void allocate_blocks() {
        if (rows_ == 0 && cols_ == 0) return; // empty object

        std::size_t block_rows = (rows_ + BlockSize - 1) / BlockSize;
        std::size_t block_cols = (cols_ + BlockSize - 1) / BlockSize;

        blocks_.resize(block_rows);
        for (auto& row : blocks_) {
            row.resize(block_cols);
            for (auto& block : row)
                block = std::make_unique<T[]>(BlockSize * BlockSize);
        }
    }


public:
    BlockMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols) { allocate_blocks(); }

    BlockMatrix()
        : BlockMatrix(0, 0) {}

    T get_rows() const { return rows_; }
    T get_cols() const { return cols_; }

    template <typename Self>
    constexpr decltype(auto) operator[](this Self&& self, std::size_t i, std::size_t j) {
        if (i >= self.rows_ || j >= self.cols_) {
            throw std::out_of_range("Matrix indices out of range.");
        }
        auto [block_row, block_col] = self.get_block_indices(i, j);
        auto [in_block_row, in_block_col] = self.get_in_block_indices(i, j);
        return std::forward<Self>(self).blocks_[block_row][block_col][in_block_row * BlockSize + in_block_col];
    }

    BlockMatrix operator+(const BlockMatrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }
        BlockMatrix result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result[i, j] = (*this)[i, j] + other[i, j];
            }
        }
        return result;
    }

    BlockMatrix operator*(const BlockMatrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication.");
        }
        BlockMatrix result(rows_, other.cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < other.cols_; ++j) {
                result[i, j] = T{};
                for (std::size_t k = 0; k < cols_; ++k) {
                    result[i, j] += (*this)[i, k] * other[k, j];
                }
            }
        }
        return result;
    }

    BlockMatrix transpose() const {
        BlockMatrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result[j, i] = (*this)[i, j];
            }
        }
        return result;
    }
};

} // namespace linear_algebra

