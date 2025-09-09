#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

namespace lynalg {

template<typename Type>
class Matrix {
  size_t cols;
  size_t rows;

public:
  std::vector<Type> data;
  std::tuple<size_t, size_t> shape;
  int numel;

  Matrix(size_t rows, size_t cols)
    : cols(cols), rows(rows), data({}), numel(rows * cols) {
    data.resize(cols * rows, Type());
    shape = std::make_tuple(rows, cols);
  }
  Matrix() : cols(0), rows(0), data({}), numel(0) { shape = {rows, cols}; };

  Type &operator()(size_t row, size_t col) {
    assert (row < rows);
    assert (col < cols);
    return data[row * cols + col];
  }

  Matrix matmul(Matrix &target) {
    assert(cols == target.rows);
    Matrix output(rows, target.cols);
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        for (size_t k = 0; k < target.rows; ++k)
          output(r, c) += (*this)(r, k) * target(k, c);
      }
    }
    return output;
  }

  Matrix multiply_scalar(Type scalar) {
    Matrix output((*this));
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = scalar * (*this)(r, c);
      }
    }
    return output;
  }

  Matrix multiply_elementwise(Matrix &target) {
    assert(shape == target.shape);
    Matrix output((*this));
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = target(r, c) * (*this)(r, c);
      }
    }
    return output;
  }

  Matrix square() {
    Matrix output((*this));
    output = multiply_elementwise(output);
    return output;
  }

  Matrix add(Matrix &target) {
    assert(shape == target.shape);
    Matrix output(rows, target.cols);
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = (*this)(r, c) + target(r, c);
      }
    }
    return output;
  }
  Matrix operator+(Matrix &target) { return add(target); }

  Matrix operator-() {
    Matrix output(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(r, c) = -(*this)(r, c);
      }
    }
    return output;
  }
  
  Matrix sub(Matrix &target) {
    Matrix neg_target = -target;
    return add(neg_target);
  }
  Matrix operator-(Matrix &target) { return sub(target); }

  Matrix T() {
    size_t new_rows{cols}, new_cols{rows};
    Matrix transposed(new_rows, new_cols);
    for (size_t r = 0; r < new_rows; ++r) {
      for (size_t c = 0; c < new_cols; ++c) {
        transposed(r, c) = (*this)(c, r);
      }
    }
    return transposed;
  }

  Matrix sum() {
    Matrix output{1, 1};
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(0, 0) += (*this)(r, c);
      }
    }
    return output;
  }

  Matrix sum(size_t dim) {
    assert (dim < 2);
    auto output = (dim == 0) ? Matrix{1, cols} : Matrix{rows, 1};
    if (dim == 0) {
      for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < rows; ++r)
          output(0, c) += (*this)(r, c);
    } else {
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
          output(r, 0) += (*this)(r, c);
    }
    return output;
  }

  Matrix mean() {
    auto n = Type(numel);
    return sum().multiply_scalar(1 / n);
  }
  
  Matrix mean(size_t dim) {
    auto n = (dim == 0) ? Type(rows) : Type(cols);
    return sum(dim).multiply_scalar(1 / n);
  }

  Matrix cat(Matrix target, size_t dim) {
    (dim == 0) ? assert(cols == target.cols) : assert(rows == target.rows);
    auto output = (dim == 0) ? Matrix{rows, cols + target.cols} : Matrix{rows + target.rows, cols};
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c)
        output(r, c) = (*this)(r, c);
    if (dim == 0) {
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < target.cols; ++c)
          output(r, c + cols) = target(r, c);
    } else {
      for (size_t r = 0; r < target.rows; ++r)
        for (size_t c = 0; c < cols; ++c)
          output(r + rows, c) = target(r, c);
    }
    return output;
  }

  Matrix diag() {
    assert((rows == 1 || cols == 1) || (rows == cols));
    if (rows == 1 || cols == 1) {
      size_t n = (rows > cols) ? rows : cols;
      Matrix output{n, n};
      for (size_t i = 0; i < n; ++i)
        output(i, i) = (rows == 1) ? (*this)(0, i) : (*this)(i, 0);
      return output;
    } else {
      Matrix output{rows, 1};
      for (size_t i = 0; i < rows; ++i)
        output(i, 0) = (*this)(i, i);
      return output;
    }
  }

  Matrix apply_function(const std::function<Type(const Type &)> &function) {
    Matrix output((*this));
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(r, c) = function((*this)(r, c));
      }
    }
    return output;
  }

  void print_shape() {
    std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
  }

  void print() {
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << (*this)(r, c) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  void fill_(Type val) {
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        (*this)(r, c) = val;
      }
    }
  }
};


// The mtx struct goes AFTER the Matrix class definition
template<typename T>
struct mtx {
private:
  static std::mt19937& get_generator() {
    static std::random_device rd{};
    static std::mt19937 gen{rd()};;
    return gen;
  }

public:
  static Matrix<T> zeros(size_t rows, size_t cols) {
    Matrix<T> M{rows, cols};
    M.fill_(T(0));
    return M;
  }

  static Matrix<T> ones(size_t rows, size_t cols) {
    Matrix<T> M{rows, cols};
    M.fill_(T(1));
    return M;
  }

  static Matrix<T> randn(size_t rows, size_t cols) {
    Matrix<T> M{rows, cols};
    T n(M.numel);
    T stdev{1 / sqrt(n)};
    std::normal_distribution<T> d{0, stdev};
    auto& gen = get_generator();
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        M(r, c) = d(gen);
      }
    }
    return M;
  }

  static Matrix<T> rand(size_t rows, size_t cols) {
    Matrix<T> M{rows, cols};
    std::uniform_real_distribution<T> d{0, 1};
    auto& gen = get_generator();
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        M(r, c) = d(gen);
      }
    }
    return M;
  }
};

} // namespace lynalg